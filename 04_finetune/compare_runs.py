"""Paired comparison of two fine-tuned runs on the same test frames.

Both runs must have been scored on the same frames (official frame set, same order): ft_<a>_test_preds.npz and
ft_<b>_test_preds.npz in output/. The difference b - a is bootstrapped PAIRED: in every replicate both runs get the
same random block weights (Bayesian block bootstrap, Exp(1) weight per one-minute block of 60 frames within each test
video, as triplet_metrics.bootstrap_ci), so frame-level noise shared by both runs cancels. Reported: point difference,
95% bias-corrected percentile CI of the difference, share of replicates with b > a, and the triplet classes that
changed most. This captures test-frame uncertainty only, not seed variance or variation across test videos.

    python 04_finetune/compare_runs.py --a off_mvd_amp --b off_svc_mvd_long_amp
Writes output/compare_<a>_vs_<b>.md / .json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
import triplet_metrics as tm  # noqa: E402

OUT = HERE / "output"
COMPS = ["ivt", "i", "v", "t", "iv", "it"]


def load(name):
    d = np.load(OUT / f"ft_{name}_test_preds.npz")
    return d["predictions"], d["labels"], d["video"], d["frame"]


def blocks(videos, block=60):
    pos = np.zeros(len(videos), int)
    for v in dict.fromkeys(videos.tolist()):
        m = videos == v
        pos[m] = np.arange(m.sum())
    _, bid = np.unique(np.stack([videos.astype(str), (pos // block).astype(str)], 1), axis=0, return_inverse=True)
    return bid.ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference run, e.g. off_mvd_amp")
    ap.add_argument("--b", required=True, help="compared run, e.g. off_svc_mvd_long_amp")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    pa, ya, va, fa = load(a.a)
    pb, yb, vb, fb = load(a.b)
    assert (va == vb).all() and (fa == fb).all(), "the two runs were not scored on the same frames in the same order"
    assert (ya == yb).all(), "labels differ between the two runs"
    names = json.load(open(HERE.parent / "02_data" / "output" / "classes_official.json"))["triplet"]

    ra, rb = tm.official_map(pa, ya, va), tm.official_map(pb, yb, vb)
    point = {c: rb[f"mAP_{c}"] - ra[f"mAP_{c}"] for c in COMPS}
    bid = blocks(va)
    rng = np.random.default_rng(0)
    diffs = {c: [] for c in COMPS}
    for _ in range(a.n):
        w = rng.exponential(1.0, bid.max() + 1)[bid]
        wa, wb = tm.weighted_map(pa, ya, va, w, COMPS), tm.weighted_map(pb, yb, vb, w, COMPS)
        for c in COMPS:
            diffs[c].append(wb[f"mAP_{c}"] - wa[f"mAP_{c}"])
    from scipy.stats import norm
    res = {}
    for c in COMPS:
        d = np.asarray(diffs[c])
        frac = np.clip((d < point[c]).mean(), 0.5 / a.n, 1 - 0.5 / a.n)
        z0 = norm.ppf(frac)
        lo, hi = norm.cdf(2 * z0 + norm.ppf(0.025)), norm.cdf(2 * z0 + norm.ppf(0.975))
        res[c] = {"a": ra[f"mAP_{c}"], "b": rb[f"mAP_{c}"], "diff": point[c],
                  "diff_ci95": [float(np.percentile(d, 100 * lo)), float(np.percentile(d, 100 * hi))],
                  "share_b_better": float((d > 0).mean())}

    apa, apb = np.array(ra["AP_ivt_per_class"], float), np.array(rb["AP_ivt_per_class"], float)
    ok = ~np.isnan(apa) & ~np.isnan(apb)
    delta = np.where(ok, apb - apa, np.nan) * 100
    order = [i for i in np.argsort(-np.nan_to_num(delta, nan=-1e9)) if ok[i]]
    pos_counts = ya.sum(0)
    top = lambda idx: [{"triplet": names[i], "test_frames": int(pos_counts[i]), "a": float(apa[i] * 100),
                        "b": float(apb[i] * 100), "diff": float(delta[i])} for i in idx]
    out = {"a": a.a, "b": a.b, "test_frames": int(len(ya)), "replicates": a.n, "components": res,
           "classes_b_better": int((delta[ok] > 0).sum()), "classes_compared": int(ok.sum()),
           "largest_gains": top(order[:8]), "largest_losses": top(order[::-1][:8]),
           "note": "paired block bootstrap over test frames; test-frame uncertainty only (no seed or video variance)"}
    (OUT / f"compare_{a.a}_vs_{a.b}.json").write_text(json.dumps(out, indent=2))

    L = [f"# {a.b} vs {a.a} (paired, same {len(ya):,} test frames)", "",
         "Paired Bayesian block bootstrap (200 replicates, one-minute blocks, same weights for both runs), 95% "
         "bias-corrected percentile CI of the difference. Test-frame uncertainty only: no seed or video variance.", "",
         f"| Metric | {a.a} | {a.b} | Difference (b - a) | 95% CI | Replicates with b > a |", "|---|---|---|---|---|---|"]
    for c in COMPS:
        r = res[c]
        L.append(f"| AP_{c.upper()} | {r['a']:.2f} | {r['b']:.2f} | **{r['diff']:+.2f}** | "
                 f"{r['diff_ci95'][0]:+.2f} to {r['diff_ci95'][1]:+.2f} | {100 * r['share_b_better']:.0f}% |")
    L += ["", f"Triplet classes better in {a.b}: {out['classes_b_better']} of {out['classes_compared']}.", "",
          "| Largest gains | test frames | a | b | diff |", "|---|---|---|---|---|"]
    L += [f"| {g['triplet']} | {g['test_frames']} | {g['a']:.1f} | {g['b']:.1f} | {g['diff']:+.1f} |" for g in out["largest_gains"]]
    L += ["", "| Largest losses | test frames | a | b | diff |", "|---|---|---|---|---|"]
    L += [f"| {g['triplet']} | {g['test_frames']} | {g['a']:.1f} | {g['b']:.1f} | {g['diff']:+.1f} |" for g in out["largest_losses"]]
    (OUT / f"compare_{a.a}_vs_{a.b}.md").write_text("\n".join(L) + "\n", encoding="utf8")
    print("\n".join(L[4:4 + 2 + len(COMPS)]), flush=True)


if __name__ == "__main__":
    main()
