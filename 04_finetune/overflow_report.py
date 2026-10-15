"""Mixed-precision overflow record of the official fine-tunes, from the training log.

The official script (USE_DS=0 path) uses torch.cuda.amp.GradScaler with its defaults: initial scale 65,536, halve on
an overflow (that optimizer step is skipped), double after 2,000 consecutive overflow-free steps. The engine logs
loss_scale every iteration and grad_norm only on optimizer steps; on a skipped step grad_norm is inf, so that epoch's
average grad_norm prints as inf. The scale is a step function, so every halving is one skipped step and every
doubling marks 2,000 clean steps: following the logged scale counts the skipped steps exactly.

    python 04_finetune/overflow_report.py [log]      # default: logs/official_ft.log (repo) or output/rtx4090/ (project)
Writes output/overflow_report.md / .json.
"""
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else next(
    (p for p in (HERE / "output" / "rtx4090" / "official_ft_2026-10-13.log", HERE.parent / "logs" / "official_ft.log")
     if p.exists()), HERE.parent / "logs" / "official_ft.log")
STEP = re.compile(r"^Epoch: \[(\d+)\]\s+\[\s*(\d+)/(\d+)\].*?loss: ([\d.na-z]+) .*?loss_scale: ([\d.]+) \(")
NORM = re.compile(r"grad_norm: ([\d.na-z]+) \(")
RUNS = ("off_mvd_amp", "off_svc_mvd_long_amp")


def epoch_avg_norm(run):
    """Per-epoch average grad_norm as written by the official script (inf if any step of the epoch overflowed)."""
    import pandas as pd
    f = HERE / "output" / f"ft_{run}_log.csv"
    if not f.exists():
        return {}
    t = pd.read_csv(f)
    return {int(e): ("inf" if not math.isfinite(g) else f"{g:.3f}") for e, g in zip(t.epoch, t.train_grad_norm)}


def main():
    run, last = None, {}
    stats = {r: defaultdict(lambda: {"halvings": 0, "doublings": 0, "scale_min": math.inf, "scale_max": 0,
                                     "lines": 0, "inf_norm_lines": 0, "nonfinite_loss_lines": 0}) for r in RUNS}
    first_scale = {}
    for line in open(LOG, encoding="utf8", errors="ignore"):
        m = re.search(r"start: (off_\w+) \(", line)
        if m:
            run = m.group(1) if m.group(1) in RUNS else None
            last.pop(run, None)
            continue
        if run is None:
            continue
        m = STEP.match(line)
        if not m:
            continue
        ep, loss, scale = int(m.group(1)), m.group(4), float(m.group(5))
        s = stats[run][ep]
        s["lines"] += 1
        s["scale_min"], s["scale_max"] = min(s["scale_min"], scale), max(s["scale_max"], scale)
        n = NORM.search(line)
        if n and not math.isfinite(float(n.group(1))):
            s["inf_norm_lines"] += 1
        if not math.isfinite(float(loss)):
            s["nonfinite_loss_lines"] += 1
        first_scale.setdefault(run, scale)
        if run in last and scale != last[run]:
            k = round(math.log2(scale / last[run]))
            s["doublings" if k > 0 else "halvings"] += abs(k)
        last[run] = scale

    out, md = {}, ["# Mixed-precision overflow record (official fine-tunes, torch GradScaler)", "",
                   "Each halving of the loss scale = one optimizer step skipped because of an overflow (inf/NaN "
                   "gradients); the weights are not changed on that step. 1,140 optimizer steps per epoch.", ""]
    for r in RUNS:
        eps = stats[r]
        if not eps:
            continue
        tot_skip = sum(e["halvings"] for e in eps.values())
        tot_steps = 1140 * len(eps)
        rows = {ep: {k: v for k, v in e.items()} for ep, e in sorted(eps.items())}
        out[r] = {"epochs": len(eps), "optimizer_steps": tot_steps, "skipped_steps": tot_skip,
                  "skipped_share": tot_skip / tot_steps, "initial_scale": first_scale[r], "final_scale": last[r],
                  "nonfinite_loss_lines": sum(e["nonfinite_loss_lines"] for e in eps.values()), "per_epoch": rows}
        title = {"off_mvd_amp": "MVD-K400 fine-tune", "off_svc_mvd_long_amp": "SurgVISTA 30-epoch fine-tune"}[r]
        md += [f"## {title} ({r})", "",
               f"Skipped optimizer steps: **{tot_skip} of {tot_steps:,}** ({100 * tot_skip / tot_steps:.2f}%). "
               f"Loss scale {first_scale[r]:,.0f} → {last[r]:,.0f}. Non-finite loss values logged: "
               f"{out[r]['nonfinite_loss_lines']} (the official script stops on a non-finite loss).", "",
               "| Epoch | Skipped steps (halvings) | Scale doublings | Loss scale min to max | Epoch-average grad_norm (training log) |",
               "|---|---|---|---|---|"]
        avg = epoch_avg_norm(r)
        md += [f"| {ep + 1} | {e['halvings']} | {e['doublings']} | {e['scale_min']:,.0f} to {e['scale_max']:,.0f} | "
               f"{avg.get(ep, 'n/a')} |" for ep, e in sorted(eps.items())]
        md.append("")
    (HERE / "output" / "overflow_report.json").write_text(json.dumps(out, indent=1, default=float))
    (HERE / "output" / "overflow_report.md").write_text("\n".join(md) + "\n", encoding="utf8")
    for r, o in out.items():
        print(f"{r}: skipped {o['skipped_steps']} of {o['optimizer_steps']} steps ({100 * o['skipped_share']:.2f}%), "
              f"scale {o['initial_scale']:,.0f} -> {o['final_scale']:,.0f}, non-finite loss lines {o['nonfinite_loss_lines']}")


if __name__ == "__main__":
    main()
