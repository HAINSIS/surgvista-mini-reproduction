"""Re-score saved test predictions on the frames the official protocol evaluates.

The official label builder (datasets/data_processing/finetune_preprosses/generate_pkl_cholect50.py) skips every
frame whose triplet id is "-1" (no triplet at all) when it writes the train/val/test pickles, so those frames
are never trained on or scored. Fine-tunes before 2026-10-13 kept them as all-negative rows (24.7 % of the
challenge test frames). This script needs no GPU: it reads ft_<name>_test_preds.npz and reports the official
metric on all frames vs labelled frames only.

    python 04_finetune/rescore_official_frames.py               # every ft_*_test_preds.npz in output/
    python 04_finetune/rescore_official_frames.py mvd mvd_png   # chosen runs
Writes output/rescore_official_frames.md / .json.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
import triplet_metrics as tm  # noqa: E402

OUT = HERE / "output"
COMPS = ["i", "v", "t", "iv", "it", "ivt"]


def rescore(npz):
    d = np.load(npz)
    p, y, v = d["predictions"], d["labels"], d["video"]
    keep = y.sum(1) > 0
    allf = tm.official_map(p, y, v)
    lab = tm.official_map(p[keep], y[keep], v[keep])
    ci = tm.bootstrap_ci(p[keep], y[keep], v[keep], frames=d["frame"][keep])
    return {"frames": int(len(y)), "labelled_frames": int(keep.sum()),
            "all_frames": {c: allf[f"mAP_{c}"] for c in COMPS},
            "labelled_only": {c: lab[f"mAP_{c}"] for c in COMPS}, "labelled_only_ci95": ci}


def main():
    names = sys.argv[1:] or sorted(p.name[3:-len("_test_preds.npz")] for p in OUT.glob("ft_*_test_preds.npz"))
    if not names:
        sys.exit("no ft_*_test_preds.npz in 04_finetune/output/ (copy them from the machine that ran the fine-tunes)")
    res = {n: rescore(OUT / f"ft_{n}_test_preds.npz") for n in names}
    paper = {"mvd": 35.3}
    lines = ["# Test AP on the official frame set (labelled frames only)", "",
             "Official label builder skips frames with triplet id -1; our fine-tunes kept them. Same predictions, "
             "two frame sets. Paper MVD AP_IVT 35.3, SurgVISTA 41.1.", "",
             "| run | frames | AP_IVT all | AP_IVT labelled only | 95% CI | I | V | T | IV | IT |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in res.items():
        a, b, ci = r["all_frames"], r["labelled_only"], r["labelled_only_ci95"]["ivt"]
        lines.append(f"| {n} | {r['labelled_frames']}/{r['frames']} | {a['ivt']:.1f} | **{b['ivt']:.1f}** | "
                     f"{ci[0]:.1f}-{ci[1]:.1f} | " + " | ".join(f"{b[c]:.1f}" for c in ["i", "v", "t", "iv", "it"]) + " |")
        print(f"{n:24s} all {a['ivt']:.2f}  labelled-only {b['ivt']:.2f}  (CI {ci[0]:.1f}-{ci[1]:.1f})", flush=True)
    lines += ["", "Training still included the empty frames in these runs; runs from 2026-10-13 on drop them in "
              "training, validation and test, like the official code (`--keep-empty` restores the old behaviour)."]
    (OUT / "rescore_official_frames.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    (OUT / "rescore_official_frames.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
