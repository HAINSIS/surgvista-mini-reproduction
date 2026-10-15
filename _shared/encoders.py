"""Fine-tuning encoder + converters that map every candidate starting checkpoint onto one parameter layout.

VideoEncoder = official downstream/model/unifiedmodel.py VisionTransformer as configured by unified_base
(1-D sinusoid pos table, VideoMAE/VideoMAE V2/UMT weights) and unified_base_st (3-D sin-cos, SurgVISTA/MVD weights):
mean pooling -> fc_norm -> dropout(0.5) -> zero-initialised linear head (init_scale 0), LayerNorm eps 1e-6.

load_start_weights() understands:
  * SurgVISTA / MVD / our pretrain_official.py checkpoints: {"model": {...}} with student keys
    (patch_embed.*, blocks.*, norm.*, decoders...); "encoder." prefixes are stripped as in the official init_()
  * HF VideoMAE pretraining checkpoints (MCG-NJU/videomae-base): videomae.* / decoder.* / encoder_to_decoder / mask_token
  * HF VideoMAE V2 (OpenGVLab/VideoMAEv2-Base): model.*
For the pretraining student it also returns the matching video-decoder weights when the source has them.
"""
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn

from surgvista import Block, PatchEmbed, make_pos_embed, sinusoid_table, trunc_normal_  # noqa: F401


class VideoEncoder(nn.Module):
    def __init__(self, num_classes=100, img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
                 num_frames=16, tubelet_size=2, drop_path_rate=0.1, fc_drop_rate=0.5, pos="3d", init_scale=0.0):
        super().__init__()
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim, num_frames, tubelet_size)
        self.register_buffer("pos_embed", make_pos_embed(pos, embed_dim, self.patch_embed), persistent=False)
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, 4.0, True, drop_path=dpr[i], norm_layer=norm)
                                     for i in range(depth)])
        self.fc_norm = norm(embed_dim)
        self.fc_dropout = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes)
        trunc_normal_(self.head.weight, std=0.02)
        self.apply(self._init)
        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_tokens(self, x):
        x = self.patch_embed(x) + self.pos_embed.type_as(x)
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward_features(self, x):
        return self.fc_norm(self.forward_tokens(x).mean(1))

    def forward(self, x):
        return self.head(self.fc_dropout(self.forward_features(x)))


# ---------------------------------------------------------------- checkpoint conversion
def _from_hf_videomae(sd):
    """MCG-NJU/videomae-* (transformers VideoMAEForPreTraining) -> official student names."""
    out = {}
    for k, v in sd.items():
        n = k
        n = n.replace("videomae.embeddings.patch_embeddings.projection", "patch_embed.proj")
        n = n.replace("videomae.encoder.layer.", "blocks.").replace("decoder.decoder_layers.", "decoder_vid.blocks.")
        n = n.replace("attention.output.dense", "attn.proj").replace("intermediate.dense", "mlp.fc1")
        n = n.replace("output.dense", "mlp.fc2").replace("layernorm_before", "norm1").replace("layernorm_after", "norm2")
        n = n.replace("attention.attention.q_bias", "attn.q_bias").replace("attention.attention.v_bias", "attn.v_bias")
        n = n.replace("videomae.layernorm", "norm").replace("decoder.norm", "decoder_vid.norm")
        n = n.replace("decoder.head", "decoder_vid.head").replace("encoder_to_decoder", "down_vid")
        n = "mask_token_vid" if n == "mask_token" else n
        out[n] = v
    # fuse separate q/k/v weights into qkv (VideoMAE has no k bias)
    for prefix in sorted({k.rsplit("attention.attention.", 1)[0] for k in out if "attention.attention.query.weight" in k}):
        q, k_, v = (out.pop(f"{prefix}attention.attention.{c}.weight") for c in ["query", "key", "value"])
        out[f"{prefix}attn.qkv.weight"] = torch.cat([q, k_, v])
    return out


def _from_hf_videomaev2(sd):
    out = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    out.pop("fc_norm.weight", None); out.pop("fc_norm.bias", None)  # fine-tune fc_norm starts fresh, as official
    return out


def read_checkpoint(path):
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(path))
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        for key in ["model", "module", "state_dict"]:
            if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
    if any(k.startswith("videomae.") for k in sd):
        return _from_hf_videomae(sd), "hf-videomae"
    if any(k.startswith("model.blocks.") for k in sd):
        return _from_hf_videomaev2(sd), "hf-videomaev2"
    return {(k[8:] if k.startswith("encoder.") else k): v for k, v in sd.items()}, "official"


def load_start_weights(model, path, verbose=True):
    """Load encoder weights (and decoder weights, for a pretraining student) from any supported checkpoint.
    Shape-mismatched or absent keys are reported, never silently forced."""
    sd, fmt = read_checkpoint(path)
    own = model.state_dict()
    use = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    skipped_shape = sorted(k for k, v in sd.items() if k in own and own[k].shape != v.shape)
    missing = sorted(k for k in own if k not in use)
    model.load_state_dict(use, strict=False)
    report = {"path": str(path), "format": fmt, "loaded": len(use), "missing": missing,
              "shape_mismatch": skipped_shape, "unused": sorted(k for k in sd if k not in own)}
    if verbose:
        enc_missing = [k for k in missing if k.startswith(("patch_embed", "blocks"))]
        print(f"start weights {Path(path).name} ({fmt}): loaded {len(use)} tensors; "
              f"encoder missing {len(enc_missing)}; shape mismatch {skipped_shape[:4]}; other missing {len(missing) - len(enc_missing)}")
    return report


START_POS = {"hf-videomae": "1d", "hf-videomaev2": "1d", "official": "3d"}  # official unified_base vs unified_base_st


def checkpoint_pos(path):
    """Position-table kind for a checkpoint: stored by pretrain_official.py ("pos"), else by format."""
    if Path(path).suffix != ".safetensors":
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and raw.get("pos") in ("1d", "3d"):
            return raw["pos"]
    return START_POS[read_checkpoint(path)[1]]
