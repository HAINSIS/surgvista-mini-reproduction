"""05_demo: STREAMING triplet recognition with end-to-end latency (capture to display) and backlog accounting.

Timing covers the whole path. Each frame gets a capture time: for a camera, the moment the capture thread receives
it; for a file or a CholecT50 video, its scheduled arrival time in a real-time replay (second s arrives at
t0 + s / speed). From that moment the clock runs through reading and decoding, resizing, the 60-second ring
buffer, the model, drawing the overlay and, with --show, putting it on screen (imshow + waitKey). A producer thread
delivers frames on schedule into a queue; the predictor takes them in order, so when it falls behind, the waiting
time is part of each frame's latency and the queue length shows the backlog. Cameras: the capture thread reads
continuously with a 1-frame driver buffer and keeps only the newest frame, so stale buffered frames cannot pile up.

    # real-time replay of a file with the live window (full path: capture, decode, model, drawing, display)
    python 05_demo/stream_demo.py --ckpt <model>.pt --mp4 path/to/video.mp4 --show
    # held-out CholecT50 video in real time; then an overload test at 8 frames per second
    python 05_demo/stream_demo.py --ckpt <model>.pt --video 111 --max-seconds 300
    python 05_demo/stream_demo.py --ckpt <model>.pt --video 111 --speed 8 --max-seconds 120
    # camera or stream URL
    python 05_demo/stream_demo.py --ckpt <model>.pt --camera 0 --show

Writes 05_demo/output/stream_<name>_<source>_x<speed>.csv (per frame: capture time, every stage, queue length) and
_summary.json (end-to-end latency, stages, backlog, real-time verdict; agreement with the offline demo if --compare).
"""
import argparse
import json
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
sys.path.insert(0, str(HERE))
import demo_official as D  # noqa: E402  (model loading, class names, clip rule)
import surgvista as sv  # noqa: E402

SIZE, T, RATE = 224, 16, 4
STOP = object()


def to_frame(bgr):
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (854, 480), interpolation=cv2.INTER_LINEAR)


def producer(a, q, t0):
    """Puts (second, frame, capture_time, decode_ms) on the queue on schedule. Decoding happens after the capture
    time, so it counts in the latency."""
    try:
        if a.camera is not None:
            cap = cv2.VideoCapture(a.camera)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # no queue of stale frames inside the driver
            latest, lock, alive = [None], threading.Lock(), [True]

            def reader():  # read continuously so the newest frame is always the one we take
                while alive[0]:
                    ok, fr = cap.read()
                    if not ok:
                        alive[0] = False
                        break
                    with lock:
                        latest[0] = (fr, time.perf_counter())
            threading.Thread(target=reader, daemon=True).start()
            s = 0
            while alive[0] and (not a.max_seconds or s < a.max_seconds):
                wait = t0 + s - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                with lock:
                    item = latest[0]
                if item is None:
                    continue
                fr, t_cap = item
                d0 = time.perf_counter()
                frame = to_frame(fr)
                q.put((s, frame, t_cap, 1e3 * (time.perf_counter() - d0)))
                s += 1
            alive[0] = False
            return
        if a.video:
            get, n, _, _ = D.frames_cholect50(a.video)
            n = n if not a.max_seconds else min(n, a.max_seconds)
            src = ("png", get, n)
        else:
            cap = cv2.VideoCapture(str(a.mp4))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
            n = n if not a.max_seconds else min(n, a.max_seconds)
            src = ("mp4", cap, fps)
        i = 0
        for s in range(n):
            t_cap = t0 + s / a.speed if a.speed else time.perf_counter()
            wait = t_cap - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            d0 = time.perf_counter()
            if src[0] == "png":
                frame = src[1](s)                      # read + decode PNG + resize to 854x480
            else:
                cap, fps = src[1], src[2]
                while i < round(s * fps):              # skip without decoding, as a camera would
                    cap.grab(); i += 1
                ok, fr = cap.read(); i += 1
                if not ok:
                    break
                frame = to_frame(fr)
            q.put((s, frame, t_cap, 1e3 * (time.perf_counter() - d0)))
    finally:
        q.put(STOP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=int, help="CholecT50 video id (held-out test: 92 96 103 110 111)")
    src.add_argument("--mp4", help="video file")
    src.add_argument("--camera", help="camera index (e.g. 0) or stream URL")
    ap.add_argument("--name", default="model")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="frames per second fed in (1 = real time for 1-fps input; >1 = overload test; 0 = no pacing)")
    ap.add_argument("--amp", action="store_true", help="fp16 autocast, as the official evaluation (CUDA only)")
    ap.add_argument("--show", action="store_true", help="live window; its drawing and display are timed")
    ap.add_argument("--max-seconds", type=int, default=0)
    ap.add_argument("--compare", help="offline demo timeline CSV of the same video: report top-1 agreement")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.camera is not None and a.camera.isdigit():
        a.camera = int(a.camera)
    if a.camera is not None:
        a.speed = 1.0
    dev = torch.device(a.device)
    model = D.load_model(a.ckpt, dev)
    mean = torch.tensor(sv.IMAGENET_MEAN, device=dev).view(1, 3, 1, 1, 1)
    std = torch.tensor(sv.IMAGENET_STD, device=dev).view(1, 3, 1, 1, 1)
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    with torch.no_grad():  # warm-up so the first frame is not charged for CUDA initialisation
        for _ in range(3):
            model(torch.zeros(1, 3, T, SIZE, SIZE, device=dev))
        sync()

    q = queue.Queue()
    t0 = time.perf_counter() + 0.5
    threading.Thread(target=producer, args=(a, q, t0), daemon=True).start()
    buf = deque(maxlen=RATE * (T - 1) + 1)
    rows = []
    src_name = f"v{a.video}" if a.video else ("camera" if a.camera is not None else Path(a.mp4).stem)
    period = 1.0 / a.speed if a.speed else None
    print(f"streaming {src_name} on {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'}), "
          f"{'%.3g frames/s fed in' % a.speed if a.speed else 'no pacing'}, display {'on' if a.show else 'off'}", flush=True)
    while True:
        item = q.get()
        if item is STOP:
            break
        s, frame, t_cap, decode_ms = item
        t_start = time.perf_counter()
        backlog = sum(1 for x in list(q.queue) if x is not STOP)  # frames already waiting (not the end marker)
        small = torch.from_numpy(cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR))
        buf.append(small)
        ids = D.clip_ids(len(buf) - 1)
        clip = torch.stack([buf[i] for i in ids]).permute(3, 0, 1, 2)[None].to(dev).float() / 255
        clip = (clip - mean) / std
        sync(); t_pre = time.perf_counter()
        with torch.no_grad(), torch.autocast(dev.type, enabled=a.amp and dev.type == "cuda"):
            p = torch.sigmoid(model(clip).float())[0].cpu().numpy()
        sync(); t_model = time.perf_counter()
        top = np.argsort(-p)[:3]
        canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # draw the overlay (always timed)
        lat_so_far = 1e3 * (time.perf_counter() - t_cap)
        cv2.putText(canvas, f"{s // 60:02d}:{s % 60:02d}  live  latency {lat_so_far:.0f} ms", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for j, c in enumerate(top, 1):
            cv2.putText(canvas, f"{p[c]:.2f}  {D.TRIP[c]}", (10, 25 + 25 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1)
        if a.show:
            cv2.imshow("stream", canvas)
            quit_ = cv2.waitKey(1) & 0xFF == 27
        t_end = time.perf_counter()
        r = {"second": s, "capture_s": round(t_cap - t0, 3), "queue_wait_ms": round(1e3 * (t_start - t_cap) - decode_ms, 2),
             "decode_ms": round(decode_ms, 2), "preprocess_ms": round(1e3 * (t_pre - t_start), 2),
             "model_ms": round(1e3 * (t_model - t_pre), 2), "draw_display_ms": round(1e3 * (t_end - t_model), 2),
             "end_to_end_ms": round(1e3 * (t_end - t_cap), 2), "frames_waiting": backlog}
        for j, c in enumerate(top, 1):
            r[f"top{j}"], r[f"p{j}"] = D.TRIP[c], round(float(p[c]), 3)
        rows.append(r)
        if s % 60 == 0:
            print(f"  {s // 60:02d}:{s % 60:02d}  end-to-end {r['end_to_end_ms']:.0f} ms  waiting {backlog}  top-1 {r['top1']}", flush=True)
        if a.show and quit_:
            break
    if a.show:
        cv2.destroyAllWindows()

    df = pd.DataFrame(rows)
    out = HERE / "output"; out.mkdir(exist_ok=True)
    stem = out / f"stream_{a.name}_{src_name}_x{a.speed:g}"
    df.to_csv(f"{stem}.csv", index=False)
    e2e = df.end_to_end_ms.values
    stats = lambda x: {"mean": round(float(np.mean(x)), 2), "p95": round(float(np.percentile(x, 95)), 2), "max": round(float(np.max(x)), 2)}
    third = max(len(df) // 3, 1)
    summary = {"checkpoint": Path(a.ckpt).name, "source": src_name,
               "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU",
               "amp": bool(a.amp and dev.type == "cuda"), "frames_per_second_fed": a.speed, "display": a.show,
               "frames": len(df),
               "end_to_end_ms": stats(e2e),
               "stages_ms_mean": {k: round(float(df[k].mean()), 2) for k in
                                  ["queue_wait_ms", "decode_ms", "preprocess_ms", "model_ms", "draw_display_ms"]},
               "backlog": {"max_frames_waiting": int(df.frames_waiting.max()),
                           "end_to_end_first_third_mean_ms": round(float(e2e[:third].mean()), 2),
                           "end_to_end_last_third_mean_ms": round(float(e2e[-third:].mean()), 2)}}
    if period:
        late = e2e > 1e3 * period
        summary["frame_period_ms"] = round(1e3 * period, 1)
        summary["frames_later_than_next_arrival"] = int(late.sum())
        summary["keeps_up"] = bool(df.frames_waiting.max() == 0 and not late.any())
    if a.compare:
        off = pd.read_csv(a.compare).set_index("second")
        both = df.set_index("second").join(off[["top1"]], rsuffix="_offline", how="inner")
        summary["top1_agreement_with_offline"] = round(float((both.top1 == both.top1_offline).mean()), 4)
    Path(f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
