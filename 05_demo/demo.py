"""Online triplet recognition on a surgical video with a live overlay.

  python 05_demo/demo.py --init imagenet --video 111 --show    # held-out CholecT50 test video (with ground truth)
  python 05_demo/demo.py --init imagenet --mp4 path/to/clip.mp4 # any laparoscopic video, sampled at 1 fps

Needs a fine-tuned checkpoint <work_root>/ckpt/ft_<init>.pth (04_finetune/finetune.py).
Each prediction uses only the past 16 seconds (online, as in the operating room).
Writes output/demo_<init>_<source>.mp4
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
from clips import clip_index, gather  # noqa: E402
from surgvista_mini import MEAN, STD, VideoViT, load_cfg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--init", default="surgvista")
ap.add_argument("--video", type=int, default=111)
ap.add_argument("--mp4", default=None)
ap.add_argument("--show", action="store_true")
ap.add_argument("--thr", type=float, default=0.5)
args = ap.parse_args()

cfg = load_cfg(); mc = cfg["model"]; dev = "cuda"
cache = Path(cfg["data"]["cache"]); ck = Path(cfg["work_root"]) / "ckpt"
classes = json.load(open(HERE.parent / "02_data" / "output" / "classes.json"))
names = sum(classes.values(), [])
nI, nV, nT = len(classes["instruments"]), len(classes["verbs"]), len(classes["targets"])
tri0 = nI + nV + nT
S = cfg["data"]["img_size"]

model = VideoViT(S, mc["patch"], mc["tubelet"], mc["num_frames"], mc["embed_dim"], mc["depth"], mc["heads"],
                 num_classes=len(names))
model.load_state_dict(torch.load(ck / f"ft_{args.init}.pth")); model.eval().to(dev)

gt = None
if args.mp4:
    cap = cv2.VideoCapture(args.mp4); fps = cap.get(cv2.CAP_PROP_FPS) or 25; big, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i % round(fps) == 0: big.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        i += 1
    big = np.stack(big); small = np.stack([cv2.resize(f, (S, S), interpolation=cv2.INTER_AREA) for f in big])
    df = pd.DataFrame({"video": 0, "frame": np.arange(len(big))}); src = Path(args.mp4).stem
else:
    full = pd.read_csv(HERE.parent / "02_data" / "output" / "index.csv")
    rows = np.where(full.video.values == args.video)[0]
    df = full.iloc[rows].reset_index(drop=True)
    big = np.load(cache / "frames224.npy", mmap_mode="r")[rows]
    small = np.load(cache / "frames112.npy", mmap_mode="r")[rows]
    gt = np.load(cache / "labels.npy")[rows]; src = f"VID{args.video:02d}"

cidx = clip_index(df, mc["num_frames"], mc["sampling_rate"])
frames = torch.from_numpy(np.ascontiguousarray(small)).to(dev)
mean, std = MEAN.to(dev), STD.to(dev)
with torch.no_grad():
    P = torch.cat([torch.sigmoid(model((gather(frames, torch.from_numpy(cidx[i:i + 32]).to(dev)) - mean) / std)).cpu()
                   for i in range(0, len(df), 32)]).numpy()

out = HERE / "output"; out.mkdir(exist_ok=True)
W, H = 960, 540
vw = cv2.VideoWriter(str(out / f"demo_{args.init}_{src}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 4, (W, H))
for i in range(len(df)):
    canvas = np.full((H, W, 3), 18, np.uint8)
    canvas[30:30 + 480, 10:10 + 480] = cv2.cvtColor(cv2.resize(np.asarray(big[i]), (480, 480)), cv2.COLOR_RGB2BGR)
    put = lambda t, y, c=(235, 235, 235), s=0.55: cv2.putText(canvas, t, (510, y), cv2.FONT_HERSHEY_SIMPLEX, s, c, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{src}  t={int(df.frame[i])}s   model: {args.init}-mini (online, past 16 s)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    put("Instruments", 50, (120, 200, 255), 0.6); y = 75
    for k in range(nI):
        p = P[i, k]; on = gt is not None and gt[i, k]
        cv2.rectangle(canvas, (640, y - 11), (640 + int(250 * p), y + 2), (80, 180, 80) if p > args.thr else (90, 90, 90), -1)
        put(f"{names[k][:12]:<12}", y)
        if on: cv2.putText(canvas, "GT", (900, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 255, 120), 1, cv2.LINE_AA)
        y += 22
    put("Top-3 triplets <instrument, verb, target>", y + 15, (120, 200, 255), 0.6); y += 40
    for k in np.argsort(-P[i, tri0:])[:3]:
        hit = gt is not None and gt[i, tri0 + k]
        put(f"{P[i, tri0 + k]:.2f}  {names[tri0 + k]}" + ("  [GT]" if hit else ""), y, (120, 255, 120) if hit else (235, 235, 235)); y += 24
    if gt is not None:
        put("Ground truth: " + "; ".join(names[tri0 + k] for k in np.where(gt[i, tri0:])[0]) or "none", y + 12, (180, 180, 180), 0.45)
    vw.write(canvas)
    if args.show:
        cv2.imshow("SurgVISTA-mini demo (q to quit)", canvas)
        if cv2.waitKey(250) & 0xFF == ord("q"): break
vw.release(); cv2.destroyAllWindows()
print("wrote", out / f"demo_{args.init}_{src}.mp4")
