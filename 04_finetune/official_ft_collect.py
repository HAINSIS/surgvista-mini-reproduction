"""Collect an OFFICIAL-code fine-tune (official_ft_launch.py) into our output format.

Reads the run folder the official script made (<base>/<model>_<data>_<method>_..._Fixed_Stride_4/):
  0.txt    test predictions written by final_triplet_test (best-val checkpoint): "uid VIDxx frame [logits] [labels]"
  log.txt  one JSON line per epoch (train_*, val_loss, val_mAP = val AP_IVT) and a final "Final mAP-*" line
Re-scores the test predictions with our copy of the official metric (must equal the script's own Final numbers),
adds the bootstrap 95% CI, and writes 04_finetune/output/ft_<name>_{metrics.json, log.csv, test_preds.npz}.

    python 04_finetune/official_ft_collect.py --name off_mvd --base <work_root>/official_ft/out_mvd
Evaluation-only run of a screen arm (official --eval of its best-val checkpoint, run_screen.sh TEST_ARM=...): the run
has no epoch rows, so the training curve and best epoch come from the arm's screen summary (--screen-json).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
import triplet_metrics as tm  # noqa: E402

COMPS = ["i", "v", "t", "iv", "it", "ivt"]


def read_preds(path):
    uid, vid, frame, logits, labels = [], [], [], [], []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        head, data, lab = line.split("[")[0].split(), line.split("[")[1].split("]")[0], line.split("[")[2].split("]")[0]
        uid.append(int(head[0])); vid.append(int(head[1][3:])); frame.append(int(head[2]))
        logits.append(np.array(data.split(","), float)); labels.append(np.array(lab.split(","), float))
    o = np.argsort(uid)  # the official merge sorts by unique id (= per-split order: videos, then frames)
    return (np.stack(logits)[o], np.stack(labels)[o].astype(np.uint8), np.array(vid)[o], np.array(frame)[o])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", required=True, help="the --output_dir given to the official script")
    ap.add_argument("--screen-json", help="eval-only run of a screen arm: its screen_<arm>.json (curve, best epoch)")
    a = ap.parse_args()
    runs = [p.parent for p in Path(a.base).glob("*/log.txt")]
    assert len(runs) == 1, f"expected one run folder under {a.base}, found {runs}"
    run = runs[0]
    rows = [json.loads(l) for l in open(run / "log.txt") if l.strip()]
    final = next((r for r in rows if "Final mAP-ivt" in r), None)
    assert final is not None, f"{run}/log.txt has no Final line: the run has not finished its test"
    scr = json.load(open(a.screen_json)) if a.screen_json else None
    if scr:
        ep = pd.DataFrame({"epoch": range(scr["epochs"]), "train_loss": scr["train_loss_per_epoch"],
                           "val_loss": scr["val_loss_per_epoch"], "val_mAP": scr["val_ivt_per_epoch"]})
    else:
        ep = pd.DataFrame([r for r in rows if "epoch" in r])
    ep.to_csv(HERE / "output" / f"ft_{a.name}_log.csv", index=False)

    p, y, v, f = read_preds(run / "0.txt")
    res = tm.official_map(p, y, v)
    test = {f"mAP_{c}": res[f"mAP_{c}"] for c in COMPS}
    official = {f"mAP_{c}": float(final[f"Final mAP-{c}"]) for c in COMPS}
    gap = max(abs(test[k] - official[k]) for k in test)  # percentage points; float rounding only (numpy vs torch
    assert gap < 1e-3, f"our re-score differs from the official script's Final numbers by {gap}"  # sigmoid ~1e-5)
    ci = tm.bootstrap_ci(p, y, v, frames=f)
    best = ep.loc[ep["val_mAP"].idxmax()]
    meta = {"code": "official run_triplet_training.py via 04_finetune/official_ft_launch.py",
            "run_dir": str(run), "test_frames": int(len(y)),
            "hyperparameters": (run / "hyerparamter.txt").read_text().splitlines()[0][:2000]
            if (run / "hyerparamter.txt").exists() else None}
    if scr:
        d = scr.get("data") or {}
        meta["deviations"] = {"recipe": "regularisation screen arm " + scr["name"], "epochs": scr["epochs"],
                              "warmup_epochs": 2, "train_clips": f"every {d.get('thin', '?')}th training sample",
                              "arm_args": scr["args"] or "none (official values)",
                              "checkpoint": "best-val epoch of the screen run, scored once with the official --eval"}
        meta["screen_run_dir"] = scr["run_dir"]
    metrics = {"name": a.name, "best_val_epoch": int(best["epoch"]), "val_at_best": {"mAP_ivt": float(best["val_mAP"])},
               "test": test, "test_official_script": official, "rescore_max_abs_diff": gap, "test_ci95": ci,
               "meta": meta}
    (HERE / "output" / f"ft_{a.name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(HERE / "output" / f"ft_{a.name}_test_preds.npz", predictions=p, labels=y, video=v, frame=f)
    print("TEST", json.dumps({k: round(x, 2) for k, x in test.items()}), "CI95 ivt", [round(x, 1) for x in ci["ivt"]],
          f"(best val epoch {int(best['epoch'])}, val AP_IVT {best['val_mAP']:.2f})", flush=True)


if __name__ == "__main__":
    main()
