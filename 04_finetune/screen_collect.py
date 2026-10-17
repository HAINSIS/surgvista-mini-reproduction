"""Regularisation screen: checkpoint averaging and validation summary for one screening run.

The screening runs use the official script on the screen root (screen_prepare.py: test pickle = val), with a
checkpoint every epoch (--save_ckpt_freq 1). Everything here is VALIDATION ONLY.

  average   <out>: uniform weight average of the last k epochs (k from --last), written as <out>/avg_last<k>.pth
            ({"model": state_dict}, what the official --eval --resume loads). Fixed rule, chosen before any result.
  summarize <out>: per-epoch val AP_IVT from log.txt; the official "final test" lines of the best-val checkpoint and
            of each average (all scored on val); val predictions saved for paired comparisons ->
            04_finetune/output/screen_<name>.json, screen_<name>_<model>_val_preds.npz; rebuilds screen_results.md.
  clean     <out>: deletes the per-epoch checkpoints (~1 GB each, model + optimiser) once the summary exists.

    python 04_finetune/screen_collect.py average   --out $OFF/screen_out/s_base --last 3 5
    python 04_finetune/screen_collect.py summarize --out $OFF/screen_out/s_base --name s_base --args "--lr 5e-4"
    python 04_finetune/screen_collect.py clean     --out $OFF/screen_out/s_base --name s_base
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("NPJ_SCREEN_OUT", HERE / "output"))  # smoke tests write elsewhere
sys.path.insert(0, str(HERE))
from official_ft_collect import read_preds  # noqa: E402

COMPS = ["i", "v", "t", "iv", "it", "ivt"]


def run_dir(base):
    runs = [p.parent for p in Path(base).glob("*/log.txt")]
    assert len(runs) == 1, f"expected one run folder under {base}, found {runs}"
    return runs[0]


def log_rows(run):
    return [json.loads(l) for l in open(run / "log.txt") if l.strip()]


def final_line(run):
    return next((r for r in log_rows(run) if "Final mAP-ivt" in r), None)


def epochs_of(run):
    return [r for r in log_rows(run) if "epoch" in r]


def average(a):
    import torch
    run = run_dir(a.out)
    n = len(epochs_of(run))
    for k in a.last:
        dst = Path(a.out) / f"avg_last{k}.pth"
        if k > n:
            print(f"skip: only {n} epochs, cannot average the last {k}", flush=True)
            continue
        if dst.exists():
            print(f"skip (done): {dst}", flush=True)
            continue
        eps = list(range(n - k, n))
        acc, ref = None, None
        for e in eps:
            sd = torch.load(run / f"checkpoint-{e}.pth", map_location="cpu", weights_only=False)["model"]
            if acc is None:
                ref = sd
                acc = {n_: v.double().clone() for n_, v in sd.items() if v.is_floating_point()}
            else:
                for n_ in acc:
                    acc[n_] += sd[n_].double()
        avg = {n_: (acc[n_] / len(eps)).to(v.dtype) if n_ in acc else v for n_, v in ref.items()}
        torch.save({"model": avg, "averaged_epochs": eps}, dst)
        print(f"averaged epochs {eps} -> {dst}", flush=True)


def summarize(a):
    run = run_dir(a.out)
    ep = epochs_of(run)
    val = [float(r["val_mAP"]) for r in ep]
    best = int(np.argmax(val))
    models = {"best_epoch": run}
    for p in sorted(Path(a.out).glob("eval_avg_last*")):
        models[p.name[5:]] = run_dir(p)
    scored = {}
    for m, r in models.items():
        f = final_line(r)
        assert f is not None, f"{r}/log.txt has no Final line (evaluation not finished)"
        scored[m] = {f"mAP_{c}": float(f[f"Final mAP-{c}"]) for c in COMPS}
        p, y, v, fr = read_preds(r / "0.txt")
        np.savez_compressed(OUT / f"screen_{a.name}_{m}_val_preds.npz", predictions=p, labels=y, video=v, frame=fr)
    meta_p = Path(a.data) / "screen_meta.json" if a.data else None
    res = {"name": a.name, "args": a.args, "split_scored": "VALIDATION (screen root: test pickle = val)",
           "data": json.loads(meta_p.read_text()) if meta_p and meta_p.exists() else None,
           "epochs": len(ep), "val_ivt_per_epoch": val, "val_loss_per_epoch": [float(r["val_loss"]) for r in ep],
           "train_loss_per_epoch": [float(r["train_loss"]) for r in ep],
           "best_epoch": best, "best_val_ivt": val[best],
           "mean_val_ivt_last3": float(np.mean(val[-3:])), "mean_val_ivt_last5": float(np.mean(val[-5:])),
           "val_full_eval": scored, "run_dir": str(run)}
    (OUT / f"screen_{a.name}.json").write_text(json.dumps(res, indent=2))
    print(f"{a.name}: best val AP_IVT {val[best]:.2f} (epoch {best}), mean last 5 {res['mean_val_ivt_last5']:.2f}; "
          + ", ".join(f"{m} {s['mAP_ivt']:.2f}" for m, s in scored.items()), flush=True)
    table()


def table():
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("screen_*.json"))]
    L = ["# Fine-tuning screen (validation only)", "",
         "MVD-K400 start, official fine-tuning script, official hyperparameters except the args column (seed 0 unless "
         "`--seed` is given). Epochs (warm-up 2, or 1 for 5-epoch arms) and training clips per arm below. All numbers "
         "are on the 5 validation videos; the test videos are never loaded. Seed spread on validation was ±0.02 "
         "(s_stride2, 3 seeds).", "",
         "| Arm | Training clips | Epochs | Args | Best epoch val AP_IVT (epoch) | Mean val, last 3 | "
         "Weight averages of the last k epochs | Best epoch: I / V / T / IT |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s = r["val_full_eval"]
        g = lambda m, c="ivt": f"{s[m][f'mAP_{c}']:.2f}" if m in s else "-"  # noqa: E731
        thin = (r.get("data") or {}).get("thin", "?")
        nth = {1: "every", 2: "every 2nd", 3: "every 3rd"}.get(thin, f"every {thin}th")
        avgs = ", ".join(f"last {m[8:]}: {g(m)}" for m in sorted(s, key=lambda m: int(m[8:]) if m[8:].isdigit() else 0)
                         if m.startswith("avg_last"))
        L.append(f"| {r['name']} | {nth} | {r['epochs']} | `{r['args'] or 'official'}` | {r['best_val_ivt']:.2f} "
                 f"({r['best_epoch']}) | {r['mean_val_ivt_last3']:.2f} | {avgs or '-'} | "
                 + " / ".join(g("best_epoch", c) for c in ("i", "v", "t", "it")) + " |")
    (OUT / "screen_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")  # Windows default: cp1252
    print(f"table: {OUT / 'screen_results.md'} ({len(rows)} arms)", flush=True)


def clean(a):
    assert (OUT / f"screen_{a.name}.json").exists(), "summarize first"
    run = run_dir(a.out)
    gone = [p for p in run.glob("checkpoint-*.pth") if p.stem.split("-")[1].isdigit() or p.stem == "checkpoint-best"]
    for p in gone:
        p.unlink()
    print(f"deleted {len(gone)} checkpoints with optimiser state (kept checkpoint-best/ and avg_last*.pth)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["average", "summarize", "clean", "table"])
    ap.add_argument("--out", help="the screening run's --output_dir")
    ap.add_argument("--name")
    ap.add_argument("--args", default="", help="the arm's deviations from the official recipe (for the table)")
    ap.add_argument("--data", help="screen root (for screen_meta.json)")
    ap.add_argument("--last", type=int, nargs="+", default=[3, 5])
    a = ap.parse_args()
    {"average": average, "summarize": summarize, "clean": clean, "table": lambda _: table()}[a.cmd](a)
