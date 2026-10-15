"""Lay our CholecT50 copy out the way the OFFICIAL SurgVISTA fine-tuning code reads it.

The official code (<work>/ref/SurgVISTA) then runs unchanged on it: its own label builder
(datasets/data_processing/finetune_preprosses/generate_pkl_cholect50.py) writes the sample pickles, its own
loader / training / test code does the rest. Only this layout step is ours.

Writes under <work_root>/official_ft/CholecT50/ :
  videos/VIDxx/000000.png   link to our PNG (symlink on Linux, hard link on Windows; copy if neither works).
                            Frames the HF copy stores at another size than 854x480 (train videos 78/79/80) are
                            written resized (PIL bilinear, as finetune_official.py --frames png): the official
                            loader stacks a clip's frames directly and needs one size.
  triplet/VIDxx.txt         header line, then "<frame> <id,id,...>" or "<frame> -1" (no triplet), official ids
  labels_pkl/{train,val,test}/1fps{split}.pickle   written by the official script itself (ROOT_DIR set to the
                            folder above; that string is the only change, applied in memory, the clone is untouched)
Then checks the official sample lists and labels against ours (TripletClips, empty frames dropped) and writes
04_finetune/output/official_ft_data_check.json (subset/--root runs: next to the root instead). Re-runnable: existing links/files are kept.

    python 04_finetune/official_ft_prepare.py                 # all 50 videos
    python 04_finetune/official_ft_prepare.py --videos 78 5 111 --root <scratch>/CholecT50 --thin 20   # smoke test
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
from surgvista_mini import load_cfg  # noqa: E402

CFG = load_cfg()
WORK = Path(CFG["work_root"])
REF = WORK / "ref" / "SurgVISTA"
RAW = Path(CFG["data"]["raw_dir"])
NATIVE_WH = (854, 480)
GEN = REF / "datasets" / "data_processing" / "finetune_preprosses" / "generate_pkl_cholect50.py"


def link(src, dst):
    if dst.exists():
        return "kept"
    try:
        if os.name == "nt":
            os.link(src, dst)
        else:
            os.symlink(src, dst)
        return "linked"
    except OSError:
        dst.write_bytes(Path(src).read_bytes())
        return "copied"


def layout(root, df, ivt, videos):
    counts = {"linked": 0, "kept": 0, "copied": 0, "resized": 0}
    (root / "triplet").mkdir(parents=True, exist_ok=True)
    for v in videos:
        g = df[df.video == v]
        name = f"VID{v:02d}"
        vd = root / "videos" / name
        vd.mkdir(parents=True, exist_ok=True)
        for r in g.itertuples():
            src, dst = RAW / r.rel, vd / f"{r.frame:06d}.png"
            if dst.exists():
                counts["kept"] += 1
                continue
            with Image.open(src) as im:
                if im.size == NATIVE_WH:
                    counts[link(src, dst)] += 1
                else:
                    im.convert("RGB").resize(NATIVE_WH, Image.BILINEAR).save(dst)
                    counts["resized"] += 1
        lines = ["frame triplet"]
        for r in g.itertuples():
            ids = np.flatnonzero(ivt[r.Index])
            lines.append(f"{r.frame} {','.join(map(str, ids)) if len(ids) else '-1'}")
        (root / "triplet" / f"{name}.txt").write_text("\n".join(lines) + "\n")
        print(f"  {name}: {len(g)} frames", flush=True)
    return counts


def run_official_builder(root):
    src = GEN.read_text(encoding="utf8")
    old = 'ROOT_DIR = "path/to/your/dataset"'
    assert src.count(old) == 1, "official builder changed: ROOT_DIR line not found exactly once"
    ns = {"__name__": "official_builder"}
    exec(compile(src.replace(old, f"ROOT_DIR = {str(root)!r}"), str(GEN), "exec"), ns)
    ns["main"]()


def check_against_ours(root, df, ivt, videos):
    import pickle
    sys.path.insert(0, str(HERE))
    from finetune_official import TripletClips
    out = {}
    for split in ["train", "val", "test"]:
        pk = pickle.load(open(root / "labels_pkl" / split / f"1fps{split}.pickle", "rb"))
        off = {(int(k[3:]), f["frame_id"]): tuple(f["triplet_gt"]) for k, fs in pk.items() for f in fs}
        ds = TripletClips(split, False, 1, "png", keep_empty=False)
        ours = {}
        for r in ds.rows:
            if ds.videos[r] in videos:
                ours[(int(ds.videos[r]), int(ds.frames[r]))] = tuple(np.flatnonzero(ds.labels[r]).tolist())
        only_off, only_ours = sorted(set(off) - set(ours)), sorted(set(ours) - set(off))
        diff = [k for k in set(off) & set(ours) if off[k] != ours[k]]
        out[split] = {"official_samples": len(off), "our_samples": len(ours), "only_official": len(only_off),
                      "only_ours": len(only_ours), "label_mismatch": len(diff),
                      "examples": {"only_official": only_off[:5], "only_ours": only_ours[:5], "label_mismatch": diff[:5]}}
        print(f"  {split}: official {len(off)} / ours {len(ours)} samples; only-official {len(only_off)}, "
              f"only-ours {len(only_ours)}, label mismatches {len(diff)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, nargs="*", help="subset of video ids (default: all)")
    ap.add_argument("--root", default=str(WORK / "official_ft" / "CholecT50"), help="output folder")
    ap.add_argument("--thin", type=int, default=1, help="smoke tests: keep every Nth sample in the pickles (after the check)")
    a = ap.parse_args()
    root = Path(a.root)
    df = pd.read_csv(HERE.parent / "02_data" / "output" / "index_full.csv")
    ivt = np.load(Path(CFG["data"]["cache"]) / "labels_full.npz")["ivt"]
    videos = sorted(a.videos or df.video.unique().tolist())
    if a.videos and (root / "videos").exists():
        extra = {int(p.name[3:]) for p in (root / "videos").iterdir()} - set(videos)
        assert not extra, f"{root} already holds other videos {sorted(extra)}; use a fresh work root for subsets"
    print(f"layout: {len(videos)} videos -> {root}", flush=True)
    counts = layout(root, df, ivt, videos)
    print(f"frames: {counts}", flush=True)
    print("official label builder (generate_pkl_cholect50.py):", flush=True)
    run_official_builder(root)
    print("check against our sample lists (empty frames dropped):", flush=True)
    chk = check_against_ours(root, df, ivt, set(videos))
    ok = all(c["only_official"] == c["only_ours"] == c["label_mismatch"] == 0 for c in chk.values())
    res = {"root": str(root), "videos": videos, "frames": counts, "splits": chk, "identical": ok}
    full = root == WORK / "official_ft" / "CholecT50" and not a.videos  # smoke tests keep their check next to root
    dst = HERE / "output" / "official_ft_data_check.json" if full else root.parent / "official_ft_data_check.json"
    dst.write_text(json.dumps(res, indent=2))
    print("IDENTICAL" if ok else f"DIFFERENCES FOUND (see {dst})", flush=True)
    if a.thin > 1:
        assert not full, "--thin is for smoke-test roots only"
        thin(root, a.thin)


def thin(root, n):
    """Smoke tests only: keep every n-th sample per video and renumber the unique ids (the official merge sorts by them)."""
    import pickle
    for split in ["train", "val", "test"]:
        p = root / "labels_pkl" / split / f"1fps{split}.pickle"
        d, u = pickle.load(open(p, "rb")), 0
        for k in d:
            d[k] = d[k][::n]
            for f in d[k]:
                f["unique_id"] = u
                u += 1
        pickle.dump(d, open(p, "wb"))
        print(f"  thinned {split}: {u} samples", flush=True)


if __name__ == "__main__":
    main()
