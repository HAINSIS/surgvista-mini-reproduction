"""05_demo: STREAMING triplet recognition with latency measurement (real-time capability).

Unlike demo_official.py (predict the whole video, then draw it), this processes the input as a live feed:
frames are read sequentially, never ahead; every new second (1 fps, as CholecT50) is resized and pushed into a
ring buffer of the last 61 seconds, a clip of 16 frames 4 s apart (t, t-4, ..., t-60, earliest frame repeated at the
start, as the official loader) is built from the buffer, and the model predicts immediately (batch 1). Each step is
timed with CUDA synchronisation. A prediction meets its deadline when it is ready before the next frame arrives
(1,000 ms at 1 fps).

    # a video file, paced in real time (one new frame per second of wall-clock time), live window
    python 05_demo/stream_demo.py --ckpt <model>.pt --mp4 path/to/video.mp4 --show
    # a held-out CholecT50 test video, as fast as possible (throughput), compared with the offline demo
    python 05_demo/stream_demo.py --ckpt <model>.pt --video 111 --speed 0 --compare 05_demo/output/demo_<run>_v111_timeline.csv
    # a camera or stream (device index or URL), live
    python 05_demo/stream_demo.py --ckpt <model>.pt --camera 0 --show

Writes 05_demo/output/stream_<name>_<source>.csv (per second: wall time, latencies, top-3) and _summary.json
(mean / p95 / max latency, deadline misses, effective rate, agreement with the offline demo if --compare).
"""
import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
sys.path.insert(0, str(HERE))
import demo_official as D  # noqa: E402  (model loading, class names, clip rule)
import surgvista as sv  # noqa: E402

SIZE, T, RATE = 224, 16, 4


def source_frames(a):
    """Yield (second, 854x480 RGB frame) in order, one per second of video. Files are decoded sequentially
    (no seeking ahead), like a live feed; cameras/streams give one frame per wall-clock second."""
    if a.video:
        get, n, _, _ = D.frames_cholect50(a.video)
        for s in range(n if not a.max_seconds else min(n, a.max_seconds)):
            yield s, get(s)
        return
    cap = cv2.VideoCapture(a.camera if a.camera is not None else str(a.mp4))
    if a.camera is not None:
        s, t0 = 0, time.perf_counter()
        while not a.max_seconds or s < a.max_seconds:
            ok, fr = cap.read()
            if not ok:
                break
            if time.perf_counter() - t0 >= s:  # newest frame once per second
                yield s, cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (854, 480), interpolation=cv2.INTER_LINEAR)
                s += 1
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    i, s = 0, 0
    while not a.max_seconds or s < a.max_seconds:
        ok, fr = cap.read()
        if not ok:
            break
        if i >= round(s * fps):  # first decoded frame at or after each whole second
            yield s, cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (854, 480), interpolation=cv2.INTER_LINEAR)
            s += 1
        i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=int, help="CholecT50 video id (held-out test: 92 96 103 110 111)")
    src.add_argument("--mp4", help="video file")
    src.add_argument("--camera", help="camera index (e.g. 0) or stream URL")
    ap.add_argument("--name", default="model")
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time (one frame per second); 0 = as fast as possible")
    ap.add_argument("--amp", action="store_true", help="fp16 autocast, as the official evaluation (CUDA only)")
    ap.add_argument("--show", action="store_true", help="live window with the predictions")
    ap.add_argument("--max-seconds", type=int, default=0)
    ap.add_argument("--compare", help="offline demo timeline CSV of the same video: report top-1 agreement")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.camera is not None and a.camera.isdigit():
        a.camera = int(a.camera)
    dev = torch.device(a.device)
    model = D.load_model(a.ckpt, dev)
    mean = torch.tensor(sv.IMAGENET_MEAN, device=dev).view(1, 3, 1, 1, 1)
    std = torch.tensor(sv.IMAGENET_STD, device=dev).view(1, 3, 1, 1, 1)
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)

    with torch.no_grad():  # warm-up so the first real frame is not charged for CUDA initialisation
        for _ in range(3):
            model(torch.zeros(1, 3, T, SIZE, SIZE, device=dev))
        sync()

    buf = deque(maxlen=RATE * (T - 1) + 1)  # the last 61 seconds of 224x224 frames
    rows, t_start = [], time.perf_counter()
    src_name = f"v{a.video}" if a.video else ("camera" if a.camera is not None else Path(a.mp4).stem)
    print(f"streaming {src_name} on {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'}), "
          f"speed {'real time' if a.speed else 'max'}", flush=True)
    for s, frame in source_frames(a):
        if a.speed and a.camera is None:  # pace a file like a live feed: frame s arrives at s / speed seconds
            wait = t_start + s / a.speed - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        t_arrive = time.perf_counter()
        small = torch.from_numpy(cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR))
        buf.append(small)
        n = len(buf)  # buffer index n-1 is "now"; earlier history is clamped to the earliest frame in the buffer
        ids = D.clip_ids(n - 1)
        clip = torch.stack([buf[i] for i in ids]).permute(3, 0, 1, 2)[None].to(dev).float() / 255
        clip = (clip - mean) / std
        sync(); t_pre = time.perf_counter()
        with torch.no_grad(), torch.autocast(dev.type, enabled=a.amp and dev.type == "cuda"):
            p = torch.sigmoid(model(clip).float())[0].cpu().numpy()
        sync(); t_done = time.perf_counter()
        top = np.argsort(-p)[:3]
        r = {"second": s, "time": f"{s // 60:02d}:{s % 60:02d}", "wall_s": round(t_done - t_start, 3),
             "preprocess_ms": round(1e3 * (t_pre - t_arrive), 2), "model_ms": round(1e3 * (t_done - t_pre), 2),
             "latency_ms": round(1e3 * (t_done - t_arrive), 2)}
        for j, c in enumerate(top, 1):
            r[f"top{j}"], r[f"p{j}"] = D.TRIP[c], round(float(p[c]), 3)
        rows.append(r)
        if s % 60 == 0:
            print(f"  {r['time']}  latency {r['latency_ms']:.1f} ms  top-1 {r['top1']} ({r['p1']:.2f})", flush=True)
        if a.show:
            canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
            cv2.putText(canvas, f"{r['time']}  live  latency {r['latency_ms']:.0f} ms", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            for j in range(1, 4):
                cv2.putText(canvas, f"{r[f'p{j}']:.2f}  {r[f'top{j}']}", (10, 25 + 25 * j), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 1)
            cv2.imshow("stream", canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    if a.show:
        cv2.destroyAllWindows()
    out = HERE / "output"; out.mkdir(exist_ok=True)
    stem = out / f"stream_{a.name}_{src_name}"
    df = pd.DataFrame(rows); df.to_csv(f"{stem}.csv", index=False)
    lat = df.latency_ms.values
    summary = {"checkpoint": str(a.ckpt), "source": src_name, "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU",
               "amp": bool(a.amp and dev.type == "cuda"), "speed": a.speed, "seconds": len(df),
               "latency_ms": {"mean": round(float(lat.mean()), 2), "p95": round(float(np.percentile(lat, 95)), 2),
                              "max": round(float(lat.max()), 2)},
               "model_ms_mean": round(float(df.model_ms.mean()), 2), "preprocess_ms_mean": round(float(df.preprocess_ms.mean()), 2),
               "deadline_ms": 1000, "deadline_misses": int((lat > 1000).sum()),
               "effective_rate_fps": round(len(df) / max(df.wall_s.iloc[-1], 1e-9), 2)}
    if a.compare:
        off = pd.read_csv(a.compare).set_index("second")
        both = df.set_index("second").join(off[["top1"]], rsuffix="_offline", how="inner")
        summary["top1_agreement_with_offline"] = round(float((both.top1 == both.top1_offline).mean()), 4)
    Path(f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
