"""Official CholecT50 triplet metric, exactly as SurgVISTA reports it.

Uses the ivtmetrics copy vendored in the official repo (downstream/downstream_triplet/ivtmetrics, v0.1.2) and
mirrors engine_for_triplet.compute_results: sigmoid of the 100 triplet logits, frames grouped per video,
Recognition.update per frame, video_end per video, compute_video_AP(component, ignore_null=True) for
i / v / t / iv / it / ivt. I, V, T scores are disentangled from the triplet scores by ivtmetrics itself.
Inputs must use the official 100-class order (02_data/output/classes_official.json).
"""
import sys
from pathlib import Path

import numpy as np

from surgvista_mini import load_cfg

sys.path.insert(0, str(Path(load_cfg()["work_root"]) / "ref" / "SurgVISTA" / "downstream" / "downstream_triplet"))
from ivtmetrics import Recognition  # noqa: E402
from ivtmetrics.disentangle import Disentangle  # noqa: E402

# Speed-up (2026-10-10): ivtmetrics' Disentangle.extract loops over every frame in Python (43.7 s for one full
# test-set evaluation). Same computation vectorised: for each component class, the max over the triplet columns
# that map to it (official bank). Everything else (AP, NaN handling, ignore_null) stays the official code.
# Equality with the original is checked by check_metric_speedup() (max |diff| must be 0).
_ORIG_EXTRACT = Disentangle.extract
_KEY = {"i": 1, "v": 2, "t": 3, "iv": 4, "it": 5, "vt": 6}
_GROUPS = {}


def _fast_extract(self, inputs, component="i"):
    if component == "ivt":
        return inputs
    if component not in _GROUPS:
        col = self.bank[:, _KEY[component]]
        _GROUPS[component] = [np.where(col == idx)[0] for idx in sorted(np.unique(col))]
    x = np.asarray(inputs)
    return np.stack([x[:, g].max(axis=1) for g in _GROUPS[component]], axis=1)


Disentangle.extract = _fast_extract

COMPONENTS = ["i", "v", "t", "iv", "it", "ivt"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def official_map(logits, labels, videos):
    """logits, labels: N x 100 (official order); videos: N video ids, frames in temporal order within a video.
    Returns {"mAP_i": .., ..., "mAP_ivt": ..} in percent, plus per-class AP for ivt."""
    probs = _sigmoid(np.asarray(logits, np.float64))
    labels = np.asarray(labels)
    videos = np.asarray(videos)
    rec = Recognition(100)
    rec.reset_global()
    for v in dict.fromkeys(videos.tolist()):  # keep first-seen order
        m = videos == v
        rec.reset()
        rec.update(labels[m], probs[m])
        rec.video_end()
    out = {}
    for c in COMPONENTS:
        r = rec.compute_video_AP(c, ignore_null=True)
        out[f"mAP_{c}"] = float(r["mAP"]) * 100
        if c == "ivt":
            out["AP_ivt_per_class"] = [None if np.isnan(a) else float(a) for a in np.asarray(r["AP"], float)]
    return out


def weighted_map(logits, labels, videos, weights, comps=("ivt", "i", "v", "t")):
    """official_map with per-frame weights: identical logic to ivtmetrics compute_video_AP (per video
    average_precision_score, 0/-0 -> NaN, nanmean over videos, ivt drops the 6 null classes, nanmean), with
    sklearn's sample_weight. With all weights = 1 it equals official_map exactly (check_weighted_map)."""
    import warnings
    from sklearn.metrics import average_precision_score
    probs, labels, videos, weights = _sigmoid(np.asarray(logits, np.float64)), np.asarray(labels), np.asarray(videos), np.asarray(weights)
    dis = Disentangle()
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c in comps:
            log = []
            for v in dict.fromkeys(videos.tolist()):
                m = videos == v
                t, p = dis.extract(labels[m], c), dis.extract(probs[m], c)
                ap = np.asarray(average_precision_score(t, p, average=None, sample_weight=weights[m]), float)
                ap[ap == -0.0] = np.nan
                log.append(ap)
            vw = np.nanmean(np.stack(log), axis=0)
            if c == "ivt":
                vw = vw[:-6]
            out[f"mAP_{c}"] = float(np.nanmean(vw)) * 100
    return out


def bootstrap_ci(logits, labels, videos, n=200, seed=0, comps=("ivt", "i", "v", "t"), block=60):
    """95% CI: Bayesian bootstrap (Rubin 1981; every frame gets a random Exp(1) weight, so no frame or rare class
    is ever dropped) + Efron's bias-corrected (BC) percentile interval.
    Why BC: AP is a ratio in which a rare positive's own weight enters both numerator and denominator, so
    resampled APs are biased upward when positives are few (a plain resample-with-replacement bootstrap gave
    11.7-13.4 around a point estimate of 11.4 on the MVD probe). BC shifts the percentiles by
    z0 = Phi^-1(share of replicates below the point estimate). Weights are drawn per BLOCK of `block`
    consecutive frames (60 = one minute = one clip's span) because neighbouring 1-fps frames are nearly
    duplicates; per-frame weights gave intervals ~3x too narrow. Frames must be in temporal order within each
    video. The video set is fixed (5 test videos).
    The paper does not state its CI method."""
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    videos = np.asarray(videos)
    pos = np.zeros(len(videos), int)
    for v in dict.fromkeys(videos.tolist()):
        m = videos == v
        pos[m] = np.arange(m.sum())
    _, block_id = np.unique(np.stack([videos.astype(str), (pos // block).astype(str)], 1), axis=0, return_inverse=True)
    block_id = block_id.ravel()
    point = weighted_map(logits, labels, videos, np.ones(len(videos)), comps)
    samples = {c: [] for c in comps}
    for _ in range(n):
        r = weighted_map(logits, labels, videos, rng.exponential(1.0, block_id.max() + 1)[block_id], comps)
        for c in comps:
            samples[c].append(r[f"mAP_{c}"])
    out = {}
    for c, smp in samples.items():
        smp = np.asarray(smp)
        frac = np.clip((smp < point[f"mAP_{c}"]).mean(), 0.5 / n, 1 - 0.5 / n)
        z0 = norm.ppf(frac)
        lo, hi = norm.cdf(2 * z0 + norm.ppf(0.025)), norm.cdf(2 * z0 + norm.ppf(0.975))
        out[c] = [float(np.percentile(smp, 100 * lo)), float(np.percentile(smp, 100 * hi))]
    return out


def check_weighted_map(n=3000, seed=0):
    """max |weighted_map(weights=1) - official_map| over i/v/t/iv/it/ivt (must be ~0)."""
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 100)) < 0.05).astype(np.int32)
    z = rng.normal(size=(n, 100))
    v = np.repeat(np.arange(5), n // 5 + 1)[:n]
    a, b = official_map(z, y, v), weighted_map(z, y, v, np.ones(n), COMPONENTS)
    return max(abs(a[f"mAP_{c}"] - b[f"mAP_{c}"]) for c in COMPONENTS)


def check_metric_speedup(n=3000, seed=0):
    """Compare the vectorised Disentangle with the original on random scores/labels: returns max |diff| over all
    six components (must be 0.0)."""
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 100)) < 0.05).astype(np.int32)
    p = rng.random((n, 100)).astype(np.float32)
    v = np.repeat(np.arange(5), n // 5 + 1)[:n]
    fast = official_map(np.log(p / (1 - p)), y, v)
    Disentangle.extract = _ORIG_EXTRACT
    try:
        slow = official_map(np.log(p / (1 - p)), y, v)
    finally:
        Disentangle.extract = _fast_extract
    return max(abs(fast[f"mAP_{c}"] - slow[f"mAP_{c}"]) for c in COMPONENTS)
