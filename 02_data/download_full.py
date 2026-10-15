"""
Download the FULL CholecT50 (all 100,863 PNGs) from the Voxel51/cholect50 dataset through hf-mirror.com,
bypassing any local HTTP proxy (hf-mirror is reachable directly from China).

Reads  : <raw_dir>/samples.json (already present), config data.hf_mirror / data.hf_dataset / data.raw_dir
Writes : <raw_dir>/data/<shard>/*.png, <raw_dir>/download_full.log, <raw_dir>/download_failed.txt
Resume : re-run; complete PNGs (ending in IEND) are skipped.

    work/.venv/Scripts/python.exe 02_data/download_full.py [--workers 128] [--shards 000 001 ...]
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import urllib3
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "_shared" / "config.yaml", encoding="utf-8"))
# Same relocation rule as surgvista_mini.load_cfg (not imported here so this script needs no timm/torch):
# with NPJ_WORK_ROOT set, every path under work_root moves there.
_new = os.environ.get("NPJ_WORK_ROOT")
if not _new and not Path(CFG["work_root"]).exists() and (ROOT.parent / "npj-dm").is_dir():
    _new = str((ROOT.parent / "npj-dm").resolve())  # bundle layout: work folder next to the project
if _new:
    _old = CFG["work_root"]
    CFG["data"] = {k: (_new + v[len(_old):] if isinstance(v, str) and v.startswith(_old) else v) for k, v in CFG["data"].items()}
    CFG["work_root"] = _new

# urllib3 never reads HTTP(S)_PROXY or the Windows proxy, so traffic never goes through a local proxy.
# One keep-alive pool per thread: each frame otherwise costs two fresh TLS connections (hf-mirror, then its CDN
# redirect). Measured 2026-10-09, 128 threads: new connection per file 2.32 MB/s, keep-alive 3.50 MB/s; 256 threads
# gave no gain (the mirror serves ~5.5 files/s per IP).
_TL = threading.local()


def _pool():
    if not hasattr(_TL, "pm"):
        _TL.pm = urllib3.PoolManager(maxsize=2, retries=urllib3.Retry(total=3, redirect=5, backoff_factor=1),
                                     timeout=urllib3.Timeout(connect=20, read=120))
    return _TL.pm


def complete_png(p):
    """Same check as build_index.py: a complete PNG ends with the IEND chunk."""
    try:
        if p.stat().st_size < 1024:
            return False
        with open(p, "rb") as f:
            f.seek(-12, os.SEEK_END)
            return f.read()[4:8] == b"IEND"
    except OSError:
        return False


def fetch(rel, raw_dir, base, tries=6):
    dst = raw_dir / rel
    if complete_png(dst):
        return rel, "skip", 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    for k in range(tries):
        try:
            r = _pool().request("GET", f"{base}/{rel}", preload_content=False)
            try:
                if r.status != 200:
                    raise IOError(f"HTTP {r.status}")
                n = 0
                with open(tmp, "wb") as f:
                    for chunk in r.stream(1 << 16):
                        f.write(chunk)
                        n += len(chunk)
            finally:
                r.release_conn()
            if complete_png(tmp):
                os.replace(tmp, dst)
                return rel, "ok", n
        except Exception:
            pass
        time.sleep(min(60, 2 ** k))
    tmp.unlink(missing_ok=True)
    return rel, "fail", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--shards", nargs="*", help="only these shard folders, e.g. 000 001")
    a = ap.parse_args()

    raw_dir = Path(CFG["data"]["raw_dir"])
    base = f"{CFG['data']['hf_mirror']}/datasets/{CFG['data']['hf_dataset']}/resolve/main"
    samples = json.load(open(raw_dir / "samples.json", encoding="utf-8"))["samples"]
    rels = [s["filepath"][2:] for s in samples]  # './data/000/000000.png' -> 'data/000/000000.png'
    if a.shards:
        rels = [r for r in rels if r.split("/")[1] in set(a.shards)]
    todo = [r for r in rels if not complete_png(raw_dir / r)]

    log = open(raw_dir / "download_full.log", "a", encoding="utf-8")
    say = lambda m: (print(m, flush=True), log.write(m + "\n"), log.flush())
    say(f"{time.strftime('%F %T')} start: {len(rels)} files, {len(rels) - len(todo)} already complete, "
        f"{len(todo)} to fetch via {base} (no proxy), {a.workers} workers")

    # Fixed-size batches: queuing all ~97k futures at once starved the main thread (no progress was ever logged).
    t0, done, nbytes, failed = time.time(), 0, 0, []
    chunk = 16 * a.workers
    with ThreadPoolExecutor(a.workers) as ex:
        for c in range(0, len(todo), chunk):
            for rel, status, n in ex.map(lambda r: fetch(r, raw_dir, base), todo[c:c + chunk]):
                done += 1
                nbytes += n
                if status == "fail":
                    failed.append(rel)
            dt = time.time() - t0
            eta = (len(todo) - done) * (dt / done) / 3600
            say(f"{time.strftime('%F %T')} {done}/{len(todo)}  {nbytes / 1e9:.2f} GB  {nbytes / dt / 1e6:.2f} MB/s  "
                f"fail {len(failed)}  eta {eta:.1f} h")

    (raw_dir / "download_failed.txt").write_text("\n".join(failed), encoding="utf-8")
    say(f"{time.strftime('%F %T')} done: {len(todo) - len(failed)} fetched, {len(failed)} failed "
        f"(re-run to retry)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
