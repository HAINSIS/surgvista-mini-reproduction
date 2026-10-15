"""Mini SurgVISTA: video ViT encoder + two decoders (pixel reconstruction, image-teacher distillation).

Follows the official recipe (isyangshu/SurgVISTA, pretrain_SurgVISTA/pretrain.sh):
  * tube masking, ratio 0.85, encoder sees visible tokens only
  * video decoder (depth 4) reconstructs normalized pixels of masked cubes, MSE
  * image decoder (depth 2, dim 384) regresses frame-level teacher features, SmoothL1, weight 0.05
Scaled down: ViT-S instead of ViT-B, 112 px instead of 224 px.
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block

ROOT = Path(__file__).resolve().parent.parent


def load_cfg():
    """config.yaml, with every path under its `work_root` relocated to $NPJ_WORK_ROOT when that is set
    (e.g. NPJ_WORK_ROOT=/data/npj-dm on the Linux machine); `python` follows the platform's venv layout."""
    import os
    cfg = yaml.safe_load(open(ROOT / "_shared" / "config.yaml", encoding="utf8"))
    new = os.environ.get("NPJ_WORK_ROOT")
    if not new and not Path(cfg["work_root"]).is_absolute():  # relative work_root (e.g. "work"): next to the project
        new = str((ROOT / cfg["work_root"]).resolve())
    if not new and not Path(cfg["work_root"]).exists() and (ROOT.parent / "npj-dm").is_dir():
        new = str((ROOT.parent / "npj-dm").resolve())  # bundle layout: work folder next to the project
    if new:
        old = cfg["work_root"]

        def fix(o):
            if isinstance(o, dict):
                return {k: fix(v) for k, v in o.items()}
            if isinstance(o, list):
                return [fix(v) for v in o]
            return new + o[len(old):] if isinstance(o, str) and o.startswith(old) else o
        cfg = fix(cfg)
        if os.name != "nt":
            cfg["python"] = new + "/.venv/bin/python"
    return cfg


def sincos_table(n_pos, dim):
    pos = np.arange(n_pos)[:, None]
    i = np.arange(dim)[None, :]
    ang = pos / np.power(10000, 2 * (i // 2) / dim)
    ang[:, 0::2] = np.sin(ang[:, 0::2])
    ang[:, 1::2] = np.cos(ang[:, 1::2])
    return torch.tensor(ang, dtype=torch.float32)[None]  # 1 x N x D


class PatchEmbed3D(nn.Module):
    def __init__(self, img=112, patch=16, tubelet=2, frames=16, dim=384):
        super().__init__()
        self.grid = (frames // tubelet, img // patch, img // patch)
        self.num_patches = int(np.prod(self.grid))
        self.proj = nn.Conv3d(3, dim, kernel_size=(tubelet, patch, patch), stride=(tubelet, patch, patch))

    def forward(self, x):  # B C T H W
        return self.proj(x).flatten(2).transpose(1, 2)


class VideoViT(nn.Module):
    """Encoder. forward(x, mask) keeps only visible tokens (mask True = hidden)."""

    def __init__(self, img=112, patch=16, tubelet=2, frames=16, dim=384, depth=12, heads=6,
                 drop_path=0.0, num_classes=0):
        super().__init__()
        self.patch_embed = PatchEmbed3D(img, patch, tubelet, frames, dim)
        self.register_buffer("pos", sincos_table(self.patch_embed.num_patches, dim), persistent=False)
        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList([Block(dim, heads, 4.0, qkv_bias=True, drop_path=dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes) if num_classes else None
        self.fc_norm = nn.LayerNorm(dim) if num_classes else None
        self.apply(_init)

    def forward_features(self, x, mask=None):
        x = self.patch_embed(x) + self.pos
        B, N, C = x.shape
        if mask is not None:
            x = x[~mask].reshape(B, -1, C)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, x):
        feats = self.forward_features(x)
        return self.head(self.fc_norm(feats.mean(1)))


class Decoder(nn.Module):
    def __init__(self, in_dim, dim, depth, heads, out_dim, num_patches):
        super().__init__()
        self.embed = nn.Linear(in_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_buffer("pos", sincos_table(num_patches, dim), persistent=False)
        self.blocks = nn.ModuleList([Block(dim, heads, 4.0, qkv_bias=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, out_dim)
        trunc_normal_(self.mask_token, std=0.02)
        self.apply(_init)

    def forward(self, vis, mask):
        """vis: B x Nv x C encoder tokens; returns predictions for the masked positions, B x Nm x out."""
        B = vis.shape[0]
        pos = self.pos.expand(B, -1, -1)
        x_vis = self.embed(vis) + pos[~mask].reshape(B, -1, pos.shape[-1])
        x_msk = self.mask_token + pos[mask].reshape(B, -1, pos.shape[-1])
        x = torch.cat([x_vis, x_msk], 1)
        for blk in self.blocks:
            x = blk(x)
        n_m = x_msk.shape[1]
        return self.out(self.norm(x[:, -n_m:]))


class SurgVISTAMini(nn.Module):
    def __init__(self, cfg, teacher_dim=1024, use_distill=True):
        super().__init__()
        m, p, img = cfg["model"], cfg["pretrain"], cfg["data"]["img_size"]
        self.encoder = VideoViT(img, m["patch"], m["tubelet"], m["num_frames"], m["embed_dim"], m["depth"], m["heads"])
        N = self.encoder.patch_embed.num_patches
        self.patch, self.tubelet = m["patch"], m["tubelet"]
        pix = 3 * m["tubelet"] * m["patch"] ** 2
        self.video_dec = Decoder(m["embed_dim"], p["decoder_dim_video"], p["decoder_depth_video"], 3, pix, N)
        self.use_distill = use_distill
        if use_distill:
            self.image_dec = Decoder(m["embed_dim"], p["decoder_dim_image"], p["decoder_depth_image"], 6, teacher_dim, N)
        self.w_img = p["image_teacher_loss_weight"]
        self.w_vid = p["video_reconstruction_loss_weight"]

    def pixel_targets(self, x):
        """Per-cube normalized pixels (VideoMAE norm_pix_loss). x: B C T H W in [0,1] (un-normalized)."""
        B, C, T, H, W = x.shape
        t, p = self.tubelet, self.patch
        x = x.reshape(B, C, T // t, t, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, (T // t) * (H // p) * (W // p), t * p * p, C)
        x = (x - x.mean(-2, keepdim=True)) / (x.var(-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        return x.reshape(B, x.shape[1], -1)

    def forward(self, x_norm, x_raw, mask, teacher=None):
        vis = self.encoder.forward_features(x_norm, mask)
        B = vis.shape[0]
        tgt = self.pixel_targets(x_raw)[mask].reshape(B, -1, self.video_dec.out.out_features)
        loss_vid = F.mse_loss(self.video_dec(vis, mask), tgt)
        out = {"loss_vid": loss_vid, "loss": self.w_vid * loss_vid}
        if self.use_distill and teacher is not None:
            tt = teacher[mask].reshape(B, -1, teacher.shape[-1])
            loss_img = F.smooth_l1_loss(self.image_dec(vis, mask), tt)
            out["loss_img"] = loss_img
            out["loss"] = out["loss"] + self.w_img * loss_img
        return out


def tube_mask(batch, grid, ratio, device):
    """Same spatial mask for every temporal slot (tube masking). True = masked."""
    T, H, W = grid
    n_sp = H * W
    n_keep = int(round(n_sp * (1 - ratio)))
    noise = torch.rand(batch, n_sp, device=device)
    keep = noise.argsort(1)[:, :n_keep]
    m = torch.ones(batch, n_sp, dtype=torch.bool, device=device)
    m.scatter_(1, keep, False)
    return m[:, None, :].expand(batch, T, n_sp).reshape(batch, T * n_sp)


def _init(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def inflate_imagenet_vit_s(model: VideoViT):
    """Natural-domain baseline: copy timm ImageNet ViT-S/16 weights into the video encoder
    (2D patch kernel replicated over the tubelet and divided by its length, I3D-style)."""
    import timm
    src = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=True).state_dict()
    sd = {}
    w = src["patch_embed.proj.weight"]
    t = model.patch_embed.proj.weight.shape[2]
    sd["patch_embed.proj.weight"] = w[:, :, None].repeat(1, 1, t, 1, 1) / t
    sd["patch_embed.proj.bias"] = src["patch_embed.proj.bias"]
    for k, v in src.items():
        if k.startswith("blocks.") or k.startswith("norm."):
            sd[k] = v
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
