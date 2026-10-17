"""05_demo (phase 2): second-by-second online triplet recognition with a fine-tuned official-recipe model.

Loads a fine-tuned checkpoint of the official fine-tuning script (unified_base_st = our VideoEncoder, verified
identical) and runs it over a video at 1 fps. Each prediction at second t uses only the past: frames t, t-4, ..., t-60
(16 frames, stride 4, clamped at the start), resized to 224x224 and ImageNet-normalised exactly as the official test
loader. Nothing from the future is used, as in the operating room.

    # held-out CholecT50 video (never used for training or model selection), with ground truth
    python 05_demo/demo_official.py --ckpt <run>/checkpoint-best/mp_rank_00_model_states.pt --video 111 --name svc30
    # any laparoscopic video file (no ground truth)
    python 05_demo/demo_official.py --ckpt ... --mp4 path/to/video.mp4 --name svc30_external

Writes 05_demo/output/demo_<name>_<source>{_timeline.csv, _timeline.png, .mp4, _summary.json}:
  timeline.csv  one row per second: top-k triplets with probabilities, instrument/verb/target scores, ground truth
  timeline.png  probability over time of the most frequent triplets, ground truth marked
  .mp4          the video (played at --fps, default 4x; input is 1 fps) with the top-k predictions overlaid (optionally
                only those >= --show-thr); ground truth in green
    re-draw only:  python 05_demo/demo_official.py --video 111 --name <run> --render-only [--fps 1 --show-thr 0.5]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
import encoders as E  # noqa: E402
import surgvista as sv  # noqa: E402
from surgvista_mini import load_cfg  # noqa: E402

CFG = load_cfg()
CLASSES = json.load(open(HERE.parent / "02_data" / "output" / "classes_official.json"))
TRIP = CLASSES["triplet"]
T, RATE, SIZE = 16, 4, 224


def load_model(path, device):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    sd = next((raw[k] for k in ("model", "module") if isinstance(raw, dict) and k in raw), raw)
    sd = {k[7:] if k.startswith("module.") else k: v.float() for k, v in sd.items()}
    model = E.VideoEncoder(num_classes=100, num_frames=T, drop_path_rate=0.0, fc_drop_rate=0.0, pos="3d")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k != "pos_embed"]
    assert not missing, f"checkpoint lacks {missing[:5]}"
    print(f"loaded {path} ({len(sd)} tensors; ignored {len(unexpected)})", flush=True)
    sv.FAST_EVAL_ATTENTION = True  # fused attention for eval: identical math
    return model.to(device).eval()


def frames_cholect50(video):
    df = pd.read_csv(HERE.parent / "02_data" / "output" / "index_full.csv")
    rows = df.index[df.video == video].values
    assert len(rows), f"video {video} not in the index"
    split = df.split.values[rows[0]]
    labels = np.load(Path(CFG["data"]["cache"]) / "labels_full.npz")["ivt"][rows]
    raw = Path(CFG["data"]["raw_dir"])
    paths = [raw / r for r in df.rel.values[rows]]
    return (lambda i: np.asarray(Image.open(paths[i]).convert("RGB").resize((854, 480), Image.BILINEAR))), len(rows), labels, split


def frames_mp4(path):
    """1 frame per second, as CholecT50. Frames are decoded when asked for (the demo reads them in order), not held
    in memory, so a full-length surgery needs no more RAM than a short clip."""
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"cannot open {path}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
    last = {}

    def get(s):
        if s not in last:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(s * fps)))
            ok, fr = cap.read()
            if ok:
                fr = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (854, 480), interpolation=cv2.INTER_LINEAR)
            elif last:  # unreadable frame (damaged file end): repeat the previous second
                print(f"  warning: second {s} unreadable, previous frame repeated", flush=True)
                fr = next(iter(last.values()))
            else:
                raise RuntimeError(f"cannot read second {s} of {path}")
            last.clear(); last[s] = fr
        return last[s]
    return get, n, None, "external"


def clip_ids(t):
    ids, i = [], t
    for _ in range(T):
        ids.append(i)
        if i - RATE >= 0:
            i -= RATE
    return sorted(ids)


# Component of each triplet, from the official class names "instrument,verb,target". Same rule as ivtmetrics'
# Disentangle (a component's score = max over the triplets that contain it), without needing the official clone,
# so the demo runs on any video from a plain checkout. Checked equal to Disentangle: 05_demo/check_disentangle.py.
_COMP = {c: np.array([CLASSES[k].index(name.split(",")[j]) for name in TRIP])
         for j, (c, k) in enumerate((("i", "instrument"), ("v", "verb"), ("t", "target")))}


def disentangle(probs, comp):  # instrument / verb / target scores from the 100 triplet scores (official mapping)
    ids = _COMP[comp]
    return np.array([probs[ids == k].max() for k in range(ids.max() + 1)])


def render_video(stem, rows, get, n, has_gt, topk, fps, thr):
    """Annotated video: the frame, then the triplets with probability >= thr (up to topk), or "no triplet detected".
    Played at `fps` frames per second; the input is 1 fps, so fps=1 is real time."""
    vw = cv2.VideoWriter(f"{stem}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (854, 480 + 150))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for t in range(n):
        canvas = np.full((480 + 150, 854, 3), 20, np.uint8)
        canvas[:480] = cv2.cvtColor(get(t), cv2.COLOR_RGB2BGR)
        r = rows[t]
        head = f"{r['time']}  (online: past 60 s only, 16 frames 4 s apart)" + (f"   shown if p >= {thr:.2f}" if thr > 0 else "")
        cv2.putText(canvas, head, (10, 25), font, 0.6, (255, 255, 255), 2)
        shown = [(float(r[f"p{j}"]), r[f"top{j}"]) for j in range(1, topk + 1) if float(r[f"p{j}"]) >= thr]
        for j, (pj, name) in enumerate(shown, 1):
            y = 480 + 25 * j
            cv2.rectangle(canvas, (10, y - 14), (10 + int(300 * pj), y + 2), (102, 116, 13), -1)
            cv2.putText(canvas, f"{pj:.2f}  {name}", (320, y), font, 0.5, (255, 255, 255), 1)
        if not shown:
            cv2.putText(canvas, f"no triplet detected (all below {thr:.2f})", (10, 480 + 25), font, 0.5, (160, 160, 160), 1)
        if has_gt:
            truth = r["ground_truth"]
            truth = "no instrument in view" if (truth == "(none)" and thr > 0) else truth
            cv2.putText(canvas, "truth: " + str(truth)[:90], (10, 480 + 25 * (topk + 1) + 10), font, 0.5, (80, 220, 80), 1)
        if thr > 0:
            cv2.putText(canvas, "model trained with the official protocol: only frames with >= 1 triplet",
                        (10, 480 + 142), font, 0.4, (130, 130, 130), 1)
        vw.write(canvas)
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="fine-tuned checkpoint (not needed with --render-only)")
    ap.add_argument("--video", type=int, help="CholecT50 video id (held-out test videos: 92 96 103 110 111)")
    ap.add_argument("--mp4")
    ap.add_argument("--name", default="model")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fps", type=float, default=4.0, help="playback rate of the output video; 1 = real time (input is 1 fps)")
    ap.add_argument("--show-thr", type=float, default=0.0, help="video shows only top-k triplets with probability >= this (0 = all)")
    ap.add_argument("--render-only", action="store_true",
                    help="re-draw the video from an existing _timeline.csv (no model, no GPU)")
    ap.add_argument("--max-seconds", type=int, default=0, help="only the first N seconds (quick tests)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    assert bool(a.video) != bool(a.mp4), "give exactly one of --video / --mp4"
    get, n, gt, split = frames_cholect50(a.video) if a.video else frames_mp4(a.mp4)
    if a.max_seconds:
        n = min(n, a.max_seconds)
        gt = None if gt is None else gt[:n]
    src = f"v{a.video}" if a.video else Path(a.mp4).stem
    out = HERE / "output"; out.mkdir(exist_ok=True)
    stem = out / f"demo_{a.name}_{src}"
    if a.render_only:
        rows = pd.read_csv(f"{stem}_timeline.csv").to_dict("records")[:n]
        render_video(stem, rows, get, len(rows), gt is not None, a.topk, a.fps, a.show_thr)
        print(f"re-rendered {stem}.mp4 from {stem}_timeline.csv", flush=True)
        return
    assert a.ckpt, "--ckpt is required unless --render-only"
    model = load_model(a.ckpt, a.device)
    mean = torch.tensor(sv.IMAGENET_MEAN).view(3, 1, 1, 1)
    std = torch.tensor(sv.IMAGENET_STD).view(3, 1, 1, 1)
    small = {}

    def model_frame(i):
        if i not in small:
            small[i] = torch.from_numpy(cv2.resize(get(i), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR))
            for k in [k for k in small if k < i - RATE * T]:
                del small[k]
        return small[i]

    probs = np.zeros((n, 100), np.float32)
    with torch.no_grad():
        for s in range(0, n, a.batch):
            ts = list(range(s, min(n, s + a.batch)))
            x = torch.stack([torch.stack([model_frame(i) for i in clip_ids(t)]).permute(3, 0, 1, 2).float() / 255
                             for t in ts])
            x = ((x - mean) / std).to(a.device)
            probs[ts] = torch.sigmoid(model(x)).float().cpu().numpy()
            if s % (a.batch * 50) == 0:
                print(f"  second {s}/{n}", flush=True)

    inst, verb, targ = (CLASSES[k] for k in ("instrument", "verb", "target"))
    rows = []
    for t in range(n):
        p = probs[t]; top = np.argsort(-p)[:a.topk]
        r = {"second": t, "time": f"{t // 60:02d}:{t % 60:02d}"}
        for j, c in enumerate(top, 1):
            r[f"top{j}"], r[f"p{j}"] = TRIP[c], round(float(p[c]), 3)
        for comp, nm in (("i", inst), ("v", verb), ("t", targ)):
            q = disentangle(p, comp); r[f"best_{comp}"] = nm[int(q.argmax())]; r[f"p_{comp}"] = round(float(q.max()), 3)
        if gt is not None:
            g = np.flatnonzero(gt[t]); r["ground_truth"] = "; ".join(TRIP[c] for c in g) or "(none)"
            r["gt_in_topk"] = bool(set(g) & set(top)) if len(g) else None
        rows.append(r)
    tl = pd.DataFrame(rows)
    tl.to_csv(f"{stem}_timeline.csv", index=False)

    summary = {"checkpoint": str(a.ckpt), "source": src, "split": split, "seconds": n, "topk": a.topk}
    if gt is not None and gt.sum() > 0:
        import triplet_metrics as tm
        keep = gt.sum(1) > 0  # official frame set: frames with a triplet
        logit = np.log(np.clip(probs, 1e-7, 1 - 1e-7) / np.clip(1 - probs, 1e-7, 1))
        m = tm.official_map(logit[keep], gt[keep], np.full(keep.sum(), a.video))
        summary["video_AP"] = {k: round(v, 2) for k, v in m.items() if k.startswith("mAP")}
        summary["gt_in_topk_rate"] = float(tl.gt_in_topk.dropna().mean())
    (Path(f"{stem}_summary.json")).write_text(json.dumps(summary, indent=2))

    # timeline figure: the most frequent true triplets (up to 5), then the most frequently predicted ones (8 rows in
    # all); probability over time, ground truth as green ticks
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pred = pd.Series(np.concatenate([np.argsort(-probs[t])[:a.topk] for t in range(n)])).value_counts().index
    true = [] if gt is None else [int(c) for c in np.argsort(-gt.astype(np.int64).sum(0))[:5] if gt[:, c].sum() > 0]
    freq = (true + [int(c) for c in pred if c not in true])[:8]
    fig, axes = plt.subplots(len(freq), 1, figsize=(12, 1.1 * len(freq) + 0.8), sharex=True)
    for ax, c in zip(np.atleast_1d(axes), freq):
        ax.plot(np.arange(n) / 60, probs[:, c], lw=1, color="#0d7466")
        if gt is not None:
            on = np.flatnonzero(gt[:, c])
            ax.scatter(on / 60, np.full(len(on), 1.02), s=2, color="#2a9d3a", marker="|")
        ax.set_ylim(0, 1.08); ax.set_yticks([0, 1])
        ax.text(0.003, 0.82, TRIP[c], transform=ax.transAxes, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[-1].set_xlabel("minutes (each prediction uses only the past 60 s: 16 frames, 4 s apart)")
    title = f"{a.name} on {src}" + (" (green ticks: ground truth)" if gt is not None else "")
    fig.suptitle(title, fontsize=10); fig.tight_layout()
    fig.savefig(f"{stem}_timeline.png", dpi=130); plt.close(fig)

    render_video(stem, rows, get, n, gt is not None, a.topk, a.fps, a.show_thr)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {stem}_timeline.csv / _timeline.png / .mp4 / _summary.json", flush=True)


if __name__ == "__main__":
    main()
