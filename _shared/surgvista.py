"""SurgVISTA, phase 2: a faithful re-implementation of the official pretraining model and loss.

Mirrors isyangshu/SurgVISTA @ 00ae4bc (clone: <work>/ref/SurgVISTA):
  pretrain_SurgVISTA/modeling_surgery_student.py  -> PretrainMaskedVideoStudent
  pretrain_SurgVISTA/modeling_surgery_teacher.py  -> SurgeryTeacher (EndoSSL ViT-L)
  pretrain_SurgVISTA/utils_.py                    -> Attention / Block / PatchEmbed / get_3d_sincos_pos_embed
  pretrain_SurgVISTA/engine_for_pretraining_surgery.py -> surgvista_loss
Parameter names are identical to the official ones, so official checkpoints load with strict=True and
03_pretrain/check_official.py can prove output equality. Written from scratch (the official code is
CC-BY-NC-ND), not copied.

Phase 1 code (surgvista_mini.py) is kept for reference; it also provides load_cfg().
"""
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath, trunc_normal_ as _timm_trunc_normal_

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def trunc_normal_(t, std=0.02):
    # official: truncate at +-std (not +-2)
    _timm_trunc_normal_(t, mean=0.0, std=std, a=-std, b=std)


# ---------------------------------------------------------------- position embeddings (utils_.py)
def _sincos_1d(dim, pos):
    omega = np.arange(dim // 2, dtype=np.float32) / (dim / 2.0)
    omega = 1.0 / 10000 ** omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_3d_sincos_pos_embed(embed_dim, grid_size, t_size):
    """1 x (T*H*W) x D. First D/4 channels encode time, the other 3D/4 encode (h, w)."""
    assert embed_dim % 4 == 0
    d_sp, d_t = embed_dim // 4 * 3, embed_dim // 4
    gw, gh = np.meshgrid(np.arange(grid_size, dtype=np.float32), np.arange(grid_size, dtype=np.float32))
    # official 2D: first half from grid[0] (= w coords, since meshgrid(w, h)), second half from grid[1]
    sp = np.concatenate([_sincos_1d(d_sp // 2, gw), _sincos_1d(d_sp // 2, gh)], axis=1)  # HW x d_sp
    tm = _sincos_1d(d_t, np.arange(t_size, dtype=np.float32))  # T x d_t
    pe = np.concatenate([np.repeat(tm[:, None], grid_size ** 2, 1), np.repeat(sp[None], t_size, 0)], -1)
    return torch.FloatTensor(pe.reshape(-1, embed_dim)).unsqueeze(0)


def sinusoid_table(n_position, d_hid):
    """Official get_sinusoid_encoding_table (VideoMAE 1-D table, used by official unified_base)."""
    pos = np.arange(n_position)[:, None]
    i = np.arange(d_hid)[None, :]
    t = pos / np.power(10000, 2 * (i // 2) / d_hid)
    t[:, 0::2] = np.sin(t[:, 0::2])
    t[:, 1::2] = np.cos(t[:, 1::2])
    return torch.tensor(t, dtype=torch.float32).unsqueeze(0)


def make_pos_embed(kind, dim, pe):
    """'3d' = SurgVISTA/MVD (official unified_base_st); '1d' = VideoMAE/VideoMAE V2 (official unified_base)."""
    return get_3d_sincos_pos_embed(dim, pe.num_patches_h, pe.num_patches_t) if kind == "3d" else sinusoid_table(pe.num_patches, dim)


# Evaluation-only speed-up: fused attention (SDPA) instead of the explicit softmax(QK^T)V. Same math, float
# rounding differs (~1e-6 relative). Off by default so check_official.py stays bit-exact; probe.py and
# finetune_official.predict() switch it on (measured +18% inference speed on the GTX 1070, 2026-10-10).
FAST_EVAL_ATTENTION = False
# Training too (opt-in, NPJ_SDPA=1): fused attention is mathematically identical (attn_drop is 0 in every config) and
# on fp16 GPUs (RTX 4090) much faster and lighter; check_official.py and the defaults keep the explicit path.
import os as _os
FAST_ATTENTION = _os.environ.get("NPJ_SDPA") == "1"

# ---------------------------------------------------------------- student blocks (utils_.py)
class Mlp(nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1, self.act, self.fc2, self.drop = nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim), nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Attention(nn.Module):
    """VideoMAE attention: fused qkv without bias, separate q_bias and v_bias (k has none)."""

    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim)) if qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(dim)) if qkv_bias else None
        self.attn_drop, self.proj, self.proj_drop = nn.Dropout(attn_drop), nn.Linear(dim, dim), nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        bias = None
        if self.q_bias is not None:
            bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        q, k, v = F.linear(x, self.qkv.weight, bias).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        if FAST_ATTENTION or (FAST_EVAL_ATTENTION and not self.training):
            return self.proj(F.scaled_dot_product_attention(q, k, v, scale=self.scale).transpose(1, 2).reshape(B, N, -1))
        attn = self.attn_drop(((q * self.scale) @ k.transpose(-2, -1)).softmax(-1))
        return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B, N, -1)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 init_values=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1 = self.gamma_2 = None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            return x + self.drop_path(self.mlp(self.norm2(x)))
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        return x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    """Cube embedding: Conv3d with kernel = stride = (tubelet, patch, patch)."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        self.tubelet_size = tubelet_size
        self.patch_size = (patch_size, patch_size)
        self.num_patches_h = self.num_patches_w = img_size // patch_size
        self.num_patches_t = num_frames // tubelet_size
        self.num_patches = self.num_patches_t * self.num_patches_h * self.num_patches_w
        self.proj = nn.Conv3d(in_chans, embed_dim, (tubelet_size, patch_size, patch_size),
                              (tubelet_size, patch_size, patch_size))

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


def _xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PretrainVisionTransformerDecoder(nn.Module):
    def __init__(self, num_classes, embed_dim, depth, num_heads, mlp_ratio=4.0, qkv_bias=True, drop_path_rate=0.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.0):
        super().__init__()
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_path=dpr[i],
                                           init_values=init_values, norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(_xavier_init)

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, -return_token_num:]))


# ---------------------------------------------------------------- student (modeling_surgery_student.py)
STUDENT_SIZES = {  # official registered models; decoder_embed_dim/heads differ only for large/huge
    "small": dict(encoder_embed_dim=384, encoder_depth=12, encoder_num_heads=6),
    "base": dict(encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12),
}


class PretrainMaskedVideoStudent(nn.Module):
    """forward(x, mask) -> (image-teacher predictions, pixel predictions) for the masked tokens.
    mask: B x N bool, True = masked. x: B x 3 x T x H x W, ImageNet-normalized."""

    def __init__(self, img_size=224, patch_size=16, encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                 decoder_num_classes=1536, decoder_embed_dim=384, decoder_depth_video=4, decoder_num_heads=6,
                 decoder_depth_image=2, feat_decoder_embed_dim=384, feat_decoder_num_heads=6, mlp_ratio=4.0,
                 qkv_bias=True, drop_path_rate=0.0, init_values=0.0, tubelet_size=2, num_frames=16,
                 target_feature_dim=1024, with_image_decoder=True, pos="3d"):
        super().__init__()
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, encoder_embed_dim, num_frames, tubelet_size)
        pe = self.patch_embed
        self.tubelet_size, self.num_frames = tubelet_size, num_frames
        self.pos_kind = pos
        self.pos_embed = nn.Parameter(make_pos_embed(pos, encoder_embed_dim, pe), requires_grad=False)
        dpr = torch.linspace(0, drop_path_rate, encoder_depth).tolist()
        self.blocks = nn.ModuleList([Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias, drop_path=dpr[i],
                                           init_values=init_values, norm_layer=norm) for i in range(encoder_depth)])
        self.norm = norm(encoder_embed_dim)

        self.with_image_decoder = with_image_decoder
        if with_image_decoder:
            self.mask_token_img = nn.Parameter(torch.zeros(1, 1, feat_decoder_embed_dim))
            self.down_img = nn.Linear(encoder_embed_dim, feat_decoder_embed_dim)
            self.decoder_img = PretrainVisionTransformerDecoder(
                target_feature_dim, feat_decoder_embed_dim, decoder_depth_image, feat_decoder_num_heads, mlp_ratio,
                qkv_bias, drop_path_rate, norm, init_values)
            self.pos_embed_img = nn.Parameter(make_pos_embed(pos, feat_decoder_embed_dim, pe), requires_grad=False)
            trunc_normal_(self.mask_token_img, std=0.02)

        self.mask_token_vid = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.down_vid = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.decoder_vid = PretrainVisionTransformerDecoder(
            decoder_num_classes, decoder_embed_dim, decoder_depth_video, decoder_num_heads, mlp_ratio, qkv_bias,
            drop_path_rate, norm, init_values)
        self.pos_embed_vid = nn.Parameter(make_pos_embed(pos, decoder_embed_dim, pe), requires_grad=False)
        trunc_normal_(self.mask_token_vid, std=0.02)
        self.apply(_xavier_init)

    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "mask_token"}

    def forward_encoder(self, x, mask):
        x = self.patch_embed(x) + self.pos_embed.type_as(x).detach()
        B, _, C = x.shape
        x = x[~mask].reshape(B, -1, C)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def _decode(self, x, mask, down, pos_embed, mask_token, decoder):
        x = down(x)
        B, _, C = x.shape
        pos = pos_embed.type_as(x).detach().expand(B, -1, -1)
        pos_mask = pos[mask].reshape(B, -1, C)
        x = torch.cat([x + pos[~mask].reshape(B, -1, C), mask_token + pos_mask], dim=1)
        return decoder(x, pos_mask.shape[1])

    def forward(self, x, mask):
        x = self.forward_encoder(x, mask)
        x_img = (self._decode(x, mask, self.down_img, self.pos_embed_img, self.mask_token_img, self.decoder_img)
                 if self.with_image_decoder else None)
        x_vid = self._decode(x, mask, self.down_vid, self.pos_embed_vid, self.mask_token_vid, self.decoder_vid)
        return x_img, x_vid


def build_student(size, img_size, num_frames, tubelet, **kw):
    return PretrainMaskedVideoStudent(img_size=img_size, num_frames=num_frames, tubelet_size=tubelet,
                                      **STUDENT_SIZES[size], **kw)


# ---------------------------------------------------------------- teacher (modeling_surgery_teacher.py)
class TeacherAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.attn_drop, self.proj, self.proj_drop = nn.Dropout(0.0), nn.Linear(dim, dim), nn.Dropout(0.0)

    def forward(self, x):
        B, N, C = x.shape
        h = lambda t: t.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q, k, v = h(self.q(x)), h(self.k(x)), h(self.v(x))
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))


class TeacherBlock(nn.Module):
    # official teacher blocks use the default nn.LayerNorm (eps 1e-5), not the 1e-6 passed for the final norm
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(dim), TeacherAttention(dim, num_heads)
        self.norm2, self.mlp = nn.LayerNorm(dim), Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class SurgeryTeacher(nn.Module):
    """EndoSSL ViT-L/16 as the official code runs it: patch-only pos_embed, cls token prepended, NO final norm
    applied (the checkpoint's norm.* weights load but are unused), returns the 196 patch tokens."""

    def __init__(self, img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        n = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n, embed_dim))
        self.blocks = nn.ModuleList([TeacherBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)  # loaded, never applied (as in the official forward)

    def forward(self, x):  # B x 3 x 224 x 224, ImageNet-normalized
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2) + self.pos_embed.to(x.dtype)
        x = torch.cat([self.cls_token.to(x.dtype).expand(x.shape[0], -1, -1), x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return x[:, 1:]


def load_teacher(ckpt_path):
    t = SurgeryTeacher()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state" in sd:
        sd = {(k[6:] if k.startswith("model") else k): v for k, v in sd["model_state"].items()}
    t.load_state_dict(sd, strict=True)
    return t.eval().requires_grad_(False)


# ---------------------------------------------------------------- loss (engine_for_pretraining_surgery.py)
def surgvista_loss(model, videos, mask, teacher_feats, patch_size=16, w_img=0.05, w_vid=1.0):
    """videos: B x 3 x T x H x W ImageNet-normalized student input. teacher_feats: B x (T/tubelet * L) x D raw teacher
    tokens of frames 0, tubelet, 2*tubelet, ... (time_stride_loss=True; norm_feature=False, as in pretrain.sh),
    or None for the plain-VideoMAE ablation. Returns dict(loss, loss_vid, loss_img)."""
    t = model.tubelet_size
    out_img, out_vid = model(videos, mask)
    with torch.no_grad():
        mean = torch.as_tensor(IMAGENET_MEAN, device=videos.device)[None, :, None, None, None]
        std = torch.as_tensor(IMAGENET_STD, device=videos.device)[None, :, None, None, None]
        v = rearrange(videos * std + mean, "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c",
                      p0=t, p1=patch_size, p2=patch_size)
        v = (v - v.mean(dim=-2, keepdim=True)) / (v.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        v = rearrange(v, "b n p c -> b n (p c)")
        B, _, C = v.shape
        labels = v[mask].reshape(B, -1, C)
    loss_vid = F.mse_loss(out_vid, labels)
    res = {"loss_vid": loss_vid, "loss": w_vid * loss_vid}
    if teacher_feats is not None:
        D = out_img.shape[-1]
        loss_img = F.smooth_l1_loss(out_img, teacher_feats[mask].reshape(B, -1, D))
        res["loss_img"] = loss_img
        res["loss"] = res["loss"] + w_img * loss_img
    return res


def teacher_targets(teacher, videos_for_teacher, tubelet=2):
    """Official time_stride_loss: run the teacher on frames 0, t, 2t, ... -> B x (T/t * 196) x D."""
    T = videos_for_teacher.shape[2]
    f = teacher(rearrange(videos_for_teacher[:, :, ::tubelet], "b c t h w -> (b t) c h w"))
    return rearrange(f, "(b t) l c -> b (t l) c", t=T // tubelet)


class TubeMaskingGenerator:
    """Official: int(ratio * H*W) masked per frame slot, same spatial mask for every temporal slot. 1 = masked."""

    def __init__(self, input_size, mask_ratio):
        self.frames, h, w = input_size
        self.n = h * w
        self.n_mask = int(mask_ratio * self.n)

    def __call__(self):
        m = np.hstack([np.zeros(self.n - self.n_mask), np.ones(self.n_mask)])
        np.random.shuffle(m)
        return np.tile(m, (self.frames, 1)).flatten()
