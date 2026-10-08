"""Run the frozen surgical image expert once per frame and cache its features.

  python 03_pretrain/extract_teacher.py

SurgVISTA distills frame-level spatial features of EndoSSL ViT-L/16 (1024-d) into the video student.
The teacher is frozen, so features are computed once instead of at every training step.
Reads  <work_root>/ckpt/teacher_endossl_vitl.pth, <cache>/frames224.npy
Writes <cache>/teacher_feats.npy : N x 49 x 1024 float16 (14x14 patch grid avg-pooled to 7x7,
matching the 112 px / patch 16 student grid).
"""
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from surgvista_mini import load_cfg  # noqa: E402

cfg = load_cfg()
cache = Path(cfg["data"]["cache"])
ckpt = Path(cfg["work_root"]) / "ckpt" / "teacher_endossl_vitl.pth"
dev = "cuda"

# Authors' converted checkpoint (MVD format): pos_embed covers patches only (cls token gets no
# position) -> timm no_embed_class=True; attention stores q/k/v separately -> fuse into timm's qkv.
teacher = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0, no_embed_class=True)
sd = torch.load(ckpt, map_location="cpu", weights_only=False)
for key in ["model", "state_dict", "target_encoder", "encoder"]:
    if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
        sd = sd[key]
sd = {k.replace("module.", "").replace("backbone.", "").replace("encoder.", ""): v for k, v in sd.items()}
for i in range(len(teacher.blocks)):
    p = f"blocks.{i}.attn."
    if p + "q.weight" in sd:
        sd[p + "qkv.weight"] = torch.cat([sd.pop(p + c + ".weight") for c in "qkv"])
        sd[p + "qkv.bias"] = torch.cat([sd.pop(p + c + ".bias") for c in "qkv"])
missing, unexpected = teacher.load_state_dict(sd, strict=False)
print(f"teacher loaded: {len(missing)} missing, {len(unexpected)} unexpected")
print("  missing:", missing[:8]); print("  unexpected:", unexpected[:8])
assert len(missing) < 5, "teacher checkpoint did not map onto timm ViT-L/16 - check key names"
teacher.eval().to(dev)

frames = np.load(cache / "frames224.npy", mmap_mode="r")
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
out = np.zeros((len(frames), 49, 1024), np.float16)
bs = 32
with torch.no_grad():
    for i in tqdm(range(0, len(frames), bs), desc="teacher"):
        x = torch.from_numpy(np.ascontiguousarray(frames[i:i + bs])).to(dev).permute(0, 3, 1, 2).float() / 255
        tok = teacher.forward_features((x - mean) / std)[:, teacher.num_prefix_tokens:]  # B x 196 x 1024
        g = tok.transpose(1, 2).reshape(-1, 1024, 14, 14)
        g = F.avg_pool2d(g, 2).flatten(2).transpose(1, 2)  # B x 49 x 1024
        out[i:i + bs] = g.cpu().numpy().astype(np.float16)
np.save(cache / "teacher_feats.npy", out)
print("saved", out.shape, "feature std", float(out.astype(np.float32).std()))
