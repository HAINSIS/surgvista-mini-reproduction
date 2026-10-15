"""02_data, phase 2: full CholecT50 index + frame cache, as the official SurgVISTA pipeline prepares data.

  * classes in the OFFICIAL CholecT50 id order (parsed from the official generate_pkl_cholect50.py header),
    not alphabetical as in phase 1 (audit section 3)
  * official challenge split (config phase2.split): train 40 / val 5 / test 5 videos
  * frames resized to 512x320 with PIL bilinear, as datasets/data_processing/extractframes/resize_frame.py does,
    stored in one uint8 memmap so clips load without PNG decoding

Reads  : <raw_dir>/samples.json, <raw_dir>/data/*/*.png, official clone (class maps)
Writes : output/index_full.csv, output/classes_official.json, output/data_summary_full.md,
         <cache>/labels_full.npz (ivt 100 | i 6 | v 10 | t 15, multi-hot),
         <cache>/frames_512x320.u8 (N x 320 x 512 x 3 memmap) + frames_512x320_filled.npy
Incremental: re-run while the download is in progress; only newly completed PNGs are processed.

    work/.venv/Scripts/python.exe 02_data/build_full.py [--workers 6]
"""
import argparse
import ast
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
from surgvista_mini import load_cfg  # noqa: E402
from download_full import complete_png  # noqa: E402  (same IEND check)

cfg = load_cfg(); p2 = cfg["phase2"]
RAW = Path(cfg["data"]["raw_dir"])
CACHE = Path(cfg["data"]["cache"])
H, W = p2["frame_hw"]
MM = CACHE / f"frames_{W}x{H}.u8"
FILLED = CACHE / f"frames_{W}x{H}_filled.npy"


def official_classes():
    """{'instrument': {id: name}, 'verb': ..., 'target': ..., 'triplet': ...} from the official script's header."""
    src = Path(cfg["work_root"]) / "ref" / "SurgVISTA" / "datasets" / "data_processing" / "finetune_preprosses" / "generate_pkl_cholect50.py"
    maps = {}
    for line in open(src, encoding="utf8"):
        if line.startswith("# ") and ": {" in line:
            key, d = line[2:].split(": ", 1)
            maps[key.strip()] = {int(k): v for k, v in ast.literal_eval(d.strip()).items()}
    assert [len(maps[k]) for k in ["instrument", "verb", "target", "triplet"]] == [6, 10, 15, 100], maps.keys()
    return maps


def _fill(args):
    rows, paths = args
    mm = np.memmap(MM, np.uint8, "r+", shape=(N_TOTAL, H, W, 3))
    done = []
    for r, p in zip(rows, paths):
        with Image.open(p) as im:
            mm[r] = np.asarray(im.convert("RGB").resize((W, H), Image.BILINEAR))
        done.append(r)
    mm.flush()
    return done


def labels(s, key):
    f = s.get(key) or {}
    return [c["label"] for c in (f.get("classifications") or [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = HERE / "output"; out.mkdir(exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)

    maps = official_classes()
    json.dump({k: [v[i] for i in range(len(v))] for k, v in maps.items()}, open(out / "classes_official.json", "w"), indent=1)
    idx = {k: {n: i for i, n in v.items()} for k, v in maps.items()}

    samples = json.load(open(RAW / "samples.json", encoding="utf8"))["samples"]
    rows = [{"video": int(s["video_id"][3:]), "frame": int(s["frame_number"]), "rel": s["filepath"][2:],
             "ivt": labels(s, "triplets"), "i": labels(s, "instruments"), "v": labels(s, "verbs"),
             "t": labels(s, "targets")} for s in samples]
    df = pd.DataFrame(rows).sort_values(["video", "frame"]).reset_index(drop=True)
    # every video must be frames 0..n-1 at 1 fps (official clips step back through frame ids)
    for v, g in df.groupby("video"):
        assert (g.frame.values == np.arange(len(g))).all(), f"video {v}: frames not contiguous from 0"

    val, test = set(p2["val_videos"]), set(p2["test_videos"])
    df["split"] = np.where(df.video.isin(test), "test", np.where(df.video.isin(val), "val", "train"))
    global N_TOTAL
    N_TOTAL = len(df)

    Y = {k: np.zeros((len(df), len(maps[m])), np.uint8)
         for k, m in [("ivt", "triplet"), ("i", "instrument"), ("v", "verb"), ("t", "target")]}
    unknown = set()
    for r, row in enumerate(df.itertuples()):
        for k, m in [("ivt", "triplet"), ("i", "instrument"), ("v", "verb"), ("t", "target")]:
            for lab in getattr(row, k):
                if lab in idx[m]:
                    Y[k][r, idx[m][lab]] = 1
                else:
                    unknown.add((k, lab))
    assert not unknown, f"labels not in the official class lists: {sorted(unknown)[:10]}"
    np.savez_compressed(CACHE / "labels_full.npz", **Y)

    # frame cache (incremental)
    if not MM.exists():
        np.memmap(MM, np.uint8, "w+", shape=(N_TOTAL, H, W, 3)).flush()
    filled = np.load(FILLED) if FILLED.exists() else np.zeros(N_TOTAL, bool)
    todo = [r for r in np.where(~filled)[0] if complete_png(RAW / df.rel[r])]
    print(f"{N_TOTAL} frames; {filled.sum()} cached; {len(todo)} new complete PNGs to resize to {W}x{H}")
    chunks = [(todo[i:i + 500], [str(RAW / df.rel[r]) for r in todo[i:i + 500]]) for i in range(0, len(todo), 500)]
    with ProcessPoolExecutor(a.workers, initializer=_init_worker, initargs=(N_TOTAL,)) as ex:
        for k, done in enumerate(ex.map(_fill, chunks)):
            filled[done] = True
            if k % 10 == 0:
                np.save(FILLED, filled)
                print(f"  cached {filled.sum()}/{N_TOTAL}", flush=True)
    np.save(FILLED, filled)
    df["cached"] = filled
    df.drop(columns=["ivt", "i", "v", "t"]).to_csv(out / "index_full.csv", index=False)

    lines = ["# CholecT50 full — data summary (phase 2)", "",
             f"Split `{p2['split']}`: val {sorted(val)}, test {sorted(test)}. Frames {W}x{H} (PIL bilinear).", "",
             "| split | videos | frames | cached | IVT classes present |", "|---|---|---|---|---|"]
    for sp in ["train", "val", "test"]:
        m = (df.split == sp).values
        lines.append(f"| {sp} | {df[m].video.nunique()} | {m.sum()} | {filled[m].sum()} | {(Y['ivt'][m].sum(0) > 0).sum()} / 100 |")
    lines += ["", f"Class counts (official): I {Y['i'].shape[1]}, V {Y['v'].shape[1]}, T {Y['t'].shape[1]}, IVT {Y['ivt'].shape[1]}.",
              f"Frames with no triplet label: {(Y['ivt'].sum(1) == 0).sum()}."]
    (out / "data_summary_full.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("\n".join(lines))


def _init_worker(n):
    global N_TOTAL
    N_TOTAL = n


N_TOTAL = 0
if __name__ == "__main__":
    main()
