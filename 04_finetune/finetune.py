"""04_finetune: end-to-end fine-tuning on CholecT50 triplet recognition (paper protocol: encoder + 1 FC head).

  python finetune.py --init scratch|imagenet|videomae|surgvista [--minutes 6]

Multi-label BCE over instruments(I) + verbs(V) + targets(T) + triplets(IVT); the label of a clip is the
label of its last frame (online recognition). Metric: mAP over classes present in the test set, per group.
Writes output/<init>_metrics.json, output/<init>_log.csv, <work_root>/ckpt/ft_<init>.pth
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
from clips import clip_index, gather  # noqa: E402
from surgvista_mini import MEAN, STD, VideoViT, inflate_imagenet_vit_s, load_cfg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True, help="scratch | imagenet | <name> with ckpt/<name>_encoder.pth")
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--minutes", type=float, default=6)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

cfg = load_cfg(); fc, mc = cfg["finetune"], cfg["model"]
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = "cuda"
cache = Path(cfg["data"]["cache"]); ck = Path(cfg["work_root"]) / "ckpt"
out = HERE / "output"; out.mkdir(exist_ok=True)

df = pd.read_csv(HERE.parent / "02_data" / "output" / "index.csv")
classes = json.load(open(HERE.parent / "02_data" / "output" / "classes.json"))
groups, off = {}, 0
for k, short in zip(["instruments", "verbs", "targets", "triplets"], ["I", "V", "T", "IVT"]):
    groups[short] = (off, off + len(classes[k])); off += len(classes[k])
cidx = clip_index(df, mc["num_frames"], mc["sampling_rate"])
Y = torch.from_numpy(np.load(cache / "labels.npy")).float().to(dev)
frames = torch.from_numpy(np.load(cache / "frames112.npy")).to(dev)
tr = np.where(df.split.values == "train")[0]; te = np.where(df.split.values == "test")[0]

model = VideoViT(cfg["data"]["img_size"], mc["patch"], mc["tubelet"], mc["num_frames"], mc["embed_dim"],
                 mc["depth"], mc["heads"], drop_path=fc["drop_path"], num_classes=Y.shape[1])
if args.init == "scratch":
    pass
elif args.init == "imagenet":
    print("inflate ImageNet ViT-S:", inflate_imagenet_vit_s(model))
else:  # any self-supervised encoder saved by 03_pretrain
    print("load", args.init, model.load_state_dict(torch.load(ck / f"{args.init}_encoder.pth"), strict=False))
model.to(dev)

# layer-wise lr decay (BEiT/VideoMAE fine-tuning recipe)
depth = mc["depth"]
def layer_id(n):
    if n.startswith("patch_embed"): return 0
    if n.startswith("blocks."): return int(n.split(".")[1]) + 1
    return depth + 1
param_groups = {}
for n, p in model.named_parameters():
    lid = layer_id(n); wd = 0.0 if p.ndim == 1 else fc["weight_decay"]
    param_groups.setdefault((lid, wd), []).append(p)
opt = torch.optim.AdamW([{"params": ps, "weight_decay": wd, "scale": fc["layer_decay"] ** (depth + 1 - lid)}
                         for (lid, wd), ps in param_groups.items()], lr=fc["lr"])
bs = fc["batch_size"]; spe = len(tr) // bs
epochs = args.epochs or fc["epochs"]; total = epochs * spe; warm = spe
mean, std = MEAN.to(dev), STD.to(dev)
pos_w = ((len(tr) - Y[tr].sum(0)) / Y[tr].sum(0).clamp(min=1)).clamp(max=20)  # rare-class weighting


@torch.no_grad()
def evaluate():
    model.eval(); P = []
    for i in range(0, len(te), 32):
        idx = torch.from_numpy(cidx[te[i:i + 32]]).to(dev)
        P.append(torch.sigmoid(model((gather(frames, idx) - mean) / std)).cpu())
    P = torch.cat(P).numpy(); T = Y[te].cpu().numpy(); res = {}
    for g, (a, b) in groups.items():
        keep = T[:, a:b].sum(0) > 0
        res[f"mAP_{g}"] = float(average_precision_score(T[:, a:b][:, keep], P[:, a:b][:, keep], average="macro"))
        res[f"n_classes_{g}"] = int(keep.sum())
    model.train(); return res


print(f"{args.init}: {len(tr)} train / {len(te)} test clips, {spe} steps/epoch")
log, step, t0 = [], 0, time.time()
for ep in range(epochs):
    model.train(); perm = np.random.permutation(tr); tot = 0.0
    for b in range(spe):
        lr = fc["lr"] * (step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
        for g in opt.param_groups: g["lr"] = lr * g["scale"]
        ends = perm[b * bs:(b + 1) * bs]
        idx = torch.from_numpy(cidx[ends]).to(dev)
        flip = torch.rand(len(idx), device=dev) < 0.5
        logits = model((gather(frames, idx, flip) - mean) / std)
        loss = F.binary_cross_entropy_with_logits(logits, Y[ends], pos_weight=pos_w)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        step += 1; tot += float(loss) / spe
    mins = (time.time() - t0) / 60
    log.append({"epoch": ep + 1, "minutes": round(mins, 2), "train_loss": round(tot, 4), **evaluate()})
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in log[-1].items()}, flush=True)
    pd.DataFrame(log).to_csv(out / f"{args.init}_log.csv", index=False)
    if mins > args.minutes:
        print("time cap reached"); break

final = {"init": args.init, "seed": args.seed, "epochs_run": len(log), **log[-1],
         "best_mAP_IVT": max(r["mAP_IVT"] for r in log)}
json.dump(final, open(out / f"{args.init}_metrics.json", "w"), indent=1)
torch.save(model.state_dict(), ck / f"ft_{args.init}.pth")
print("FINAL", final)
