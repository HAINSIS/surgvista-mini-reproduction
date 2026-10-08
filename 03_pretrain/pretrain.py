"""Self-supervised video pretraining on TRAIN videos only (test videos are never seen).

  python 03_pretrain/pretrain.py --variant surgvista                       # reconstruction + teacher distillation (the paper)
  python 03_pretrain/pretrain.py --variant surgvista --init-from imagenet  # same, starting from ImageNet ViT-S
  python 03_pretrain/pretrain.py --variant videomae                        # reconstruction only (optional ablation, not in reported results)

Writes <work_root>/ckpt/<variant>_encoder.pth, output/<variant>_log.csv, output/<variant>_recon.png
(--init-from imagenet saves under the name <variant>_in)
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
from clips import clip_index, gather  # noqa: E402
from surgvista_mini import MEAN, STD, SurgVISTAMini, inflate_imagenet_vit_s, load_cfg, tube_mask  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["surgvista", "videomae"], default="surgvista")
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--minutes", type=float, default=25, help="wall-clock cap")
ap.add_argument("--init-from", choices=["scratch", "imagenet"], default="scratch",
                help="imagenet = domain-adaptive continued pretraining from natural-image weights")
args = ap.parse_args()

cfg = load_cfg(); pc, mc = cfg["pretrain"], cfg["model"]
torch.manual_seed(0); np.random.seed(0)
dev = "cuda"
cache = Path(cfg["data"]["cache"]); ck = Path(cfg["work_root"]) / "ckpt"; ck.mkdir(exist_ok=True)
out = HERE / "output"; out.mkdir(exist_ok=True)

df = pd.read_csv(HERE.parent / "02_data" / "output" / "index.csv")
cidx = clip_index(df, mc["num_frames"], mc["sampling_rate"])
train_ends = np.where(df.split.values == "train")[0]
frames = torch.from_numpy(np.load(cache / "frames112.npy")).to(dev)
use_distill = args.variant == "surgvista"
teach = torch.from_numpy(np.load(cache / "teacher_feats.npy")).to(dev) if use_distill else None

model = SurgVISTAMini(cfg, use_distill=use_distill)
if args.init_from == "imagenet":
    print("encoder init from ImageNet:", inflate_imagenet_vit_s(model.encoder))
model.to(dev)
name = args.variant + ("_in" if args.init_from == "imagenet" else "")
grid = model.encoder.patch_embed.grid
opt = torch.optim.AdamW(model.parameters(), lr=pc["lr"], betas=tuple(pc["betas"]), weight_decay=pc["weight_decay"])
bs = pc["batch_size"]
steps_per_epoch = len(train_ends) // bs
epochs = args.epochs or pc["epochs"]
total = epochs * steps_per_epoch
warm = pc["warmup_epochs"] * steps_per_epoch
mean, std = MEAN.to(dev), STD.to(dev)


def teacher_targets(idx, flip):
    """B x T frames -> B x (T/2*49) x 1024, averaged over each tubelet, layer-normed, flipped with input."""
    B, T = idx.shape
    t = teach[idx].float().view(B, T // 2, 2, 7, 7, -1).mean(2)
    t[flip] = t[flip].flip(3)  # flip the W axis of the 7x7 grid to match flipped frames
    t = F.layer_norm(t, t.shape[-1:])
    return t.reshape(B, -1, t.shape[-1])


print(f"{name}: {len(train_ends)} train clips, {steps_per_epoch} steps/epoch, up to {epochs} epochs, cap {args.minutes} min")
log, step, t0 = [], 0, time.time()
for ep in range(epochs):
    model.train()
    perm = torch.from_numpy(np.random.permutation(train_ends)).to(dev)
    acc = {"loss": 0.0, "loss_vid": 0.0, "loss_img": 0.0}
    for b in range(steps_per_epoch):
        lr = pc["lr"] * (step / max(1, warm) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
        for g in opt.param_groups:
            g["lr"] = lr
        ends = perm[b * bs:(b + 1) * bs]
        idx = torch.from_numpy(cidx[ends.cpu().numpy()]).to(dev)
        flip = torch.rand(len(idx), device=dev) < 0.5
        x = gather(frames, idx, flip)
        mask = tube_mask(len(idx), grid, pc["mask_ratio"], dev)
        res = model((x - mean) / std, x, mask, teacher_targets(idx, flip) if use_distill else None)
        opt.zero_grad(set_to_none=True)
        res["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        step += 1
        for k in acc:
            acc[k] += float(res.get(k, torch.tensor(0.0))) / steps_per_epoch
    mins = (time.time() - t0) / 60
    log.append({"epoch": ep + 1, "minutes": round(mins, 2), "lr": lr, **{k: round(v, 5) for k, v in acc.items()}})
    print(log[-1], flush=True)
    pd.DataFrame(log).to_csv(out / f"{name}_log.csv", index=False)
    if mins > args.minutes:
        print(f"time cap reached after {ep + 1} epochs")
        break

torch.save(model.encoder.state_dict(), ck / f"{name}_encoder.pth")

# reconstruction figure: masked input | reconstruction | original, 4 time slots
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

model.eval()
test_end = np.where(df.split.values == "test")[0][len(np.where(df.split.values == "test")[0]) // 3]
idx = torch.from_numpy(cidx[[test_end]]).to(dev)
x = gather(frames, idx)
torch.manual_seed(1)
mask = tube_mask(1, grid, pc["mask_ratio"], dev)
with torch.no_grad():
    vis = model.encoder.forward_features((x - mean) / std, mask)
    pred = model.video_dec(vis, mask)
p, t = mc["patch"], mc["tubelet"]
tgt = model.pixel_targets(x)  # normalized cubes; un-normalize prediction with the true cube stats
B, C, T, H, W = x.shape
cubes = x.reshape(B, C, T // t, t, H // p, p, W // p, p).permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, -1, t * p * p, C)
mu, sd = cubes.mean(-2, keepdim=True), cubes.var(-2, unbiased=True, keepdim=True).sqrt() + 1e-6
rec = cubes.clone()
rec[mask] = (pred.reshape(-1, t * p * p, C) * sd[mask] + mu[mask])
msk = cubes.clone(); msk[mask] = 0.5


def to_video(c):
    v = c.reshape(B, T // t, H // p, W // p, t, p, p, C).permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, C, T, H, W)
    return v[0].permute(1, 2, 3, 0).clamp(0, 1).cpu().numpy()


rows = [("masked input (85%)", to_video(msk)), ("reconstruction", to_video(rec)), ("original", to_video(cubes))]
fig, ax = plt.subplots(3, 4, figsize=(8, 6.2))
for r, (row_name, v) in enumerate(rows):
    for c, fi in enumerate([0, 5, 10, 15]):
        ax[r, c].imshow(v[fi]); ax[r, c].axis("off")
        if c == 0:
            ax[r, c].set_title(row_name, fontsize=9, loc="left")
fig.suptitle(f"{name} - held-out test clip, frames 1/6/11/16", fontsize=10)
fig.tight_layout(); fig.savefig(out / f"{name}_recon.png", dpi=130)
print("done", out / f"{name}_recon.png")
