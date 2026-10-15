"""03_pretrain check: prove _shared/surgvista.py computes exactly what the official SurgVISTA code computes.

Loads the official classes from the clone (<work>/ref/SurgVISTA @ 00ae4bc), bypassing only their stale import
path (`pretrain_SurgMAEKD.utils_` -> the clone's `pretrain_SurgVISTA/utils_.py`), as an independent audit of our first version did.
Same weights + same inputs + same mask -> compare outputs.

Writes output/official_equivalence.json and output/teacher_output_comparison.json.
    work/.venv/Scripts/python.exe 03_pretrain/check_official.py
"""
import importlib.util
import json
import sys
import types
import warnings
from pathlib import Path

import cv2
import numpy as np
import timm
import torch

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
import surgvista as ours  # noqa: E402
from surgvista_mini import load_cfg  # noqa: E402

cfg = load_cfg()
REF = Path(cfg["work_root"]) / "ref" / "SurgVISTA"
CKPT = Path(cfg["work_root"]) / "ckpt" / "teacher_endossl_vitl.pth"
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --- import shim: only the package path changes, no official code is edited
off_utils_ = load_module("official_utils_", REF / "pretrain_SurgVISTA" / "utils_.py")
pkg = types.ModuleType("pretrain_SurgMAEKD")
pkg.utils_ = off_utils_
sys.modules["pretrain_SurgMAEKD"] = pkg
sys.modules["pretrain_SurgMAEKD.utils_"] = off_utils_
sys.modules.setdefault("utils", types.ModuleType("utils"))  # teacher imports `utils` only for its loader helper
mg = types.ModuleType("datasets.transforms.masking_generator")
mg_src = load_module("official_masking", REF / "datasets" / "transforms" / "masking_generator.py")
mg.TubeMaskingGenerator = mg_src.TubeMaskingGenerator
for n in ["datasets", "datasets.transforms"]:
    sys.modules[n] = types.ModuleType(n)
sys.modules["datasets.transforms.masking_generator"] = mg
off_student = load_module("official_student", REF / "pretrain_SurgVISTA" / "modeling_surgery_student.py")
off_teacher = load_module("official_teacher", REF / "pretrain_SurgVISTA" / "modeling_surgery_teacher.py")


def diff(a, b):
    a, b = a.double(), b.double()
    return {"max_abs": float((a - b).abs().max()), "rel_l2": float((a - b).norm() / b.norm())}


report = {"official_commit": "00ae4bc641371f5b6e1efb899450bfb61f304100", "device": dev,
          "torch": torch.__version__, "timm": timm.__version__}

# 1. 3D sin-cos position embedding
report["pos_embed"] = {
    f"dim{d}": float((ours.get_3d_sincos_pos_embed(d, 14, 8) - off_utils_.get_3d_sincos_pos_embed(d, 14, 8)).abs().max())
    for d in (384, 768, 1024)}

# 2. tube masking: same count of masked tokens per slot as the official generator
gen_o, gen_m = mg_src.TubeMaskingGenerator((8, 14, 14), 0.85), ours.TubeMaskingGenerator((8, 14, 14), 0.85)
report["tube_mask"] = {"official_masked": int(gen_o().sum()), "ours_masked": int(gen_m().sum()), "total": 8 * 196}

# 3. student: official registered models vs ours, same state dict, eval mode
kw = dict(decoder_depth_video=4, decoder_depth_image=2, feat_decoder_embed_dim=384, feat_decoder_num_heads=6,
          target_feature_dim=1024, tubelet_size=2, num_frames=16, drop_path_rate=0.1)
for size in ["small", "base"]:
    torch.manual_seed(0); np.random.seed(0)
    off = getattr(off_student, f"pretrain_masked_video_student_{size}_patch16_224")(**kw).to(dev).eval()
    mine = ours.build_student(size, 224, 16, 2, decoder_depth_video=4, decoder_depth_image=2,
                              target_feature_dim=1024, drop_path_rate=0.1).to(dev).eval()
    res = mine.load_state_dict(off.state_dict(), strict=True)
    x = torch.randn(2, 3, 16, 224, 224, device=dev)
    mask = torch.from_numpy(np.stack([gen_o(), gen_o()])).to(dev).bool()
    with torch.no_grad():
        oi, ov = off(x, mask)
        mi, mv = mine(x, mask)
    report[f"student_{size}"] = {
        "strict_load": str(res), "params": sum(p.numel() for p in mine.parameters()),
        "same_param_names": sorted(off.state_dict()) == sorted(mine.state_dict()),
        "image_decoder_out": diff(mi, oi), "video_decoder_out": diff(mv, ov)}
    del off, mine
    torch.cuda.empty_cache()

# 4. teacher: official MaskedAutoencoderViT vs ours vs the phase-1 timm teacher, same checkpoint, 2 real frames
raw = Path(cfg["data"]["raw_dir"]) / "data" / "005"
imgs = []
for p in sorted(raw.glob("*.png"))[:2]:
    im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]; s = min(h, w)
    im = cv2.resize(im[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2], (224, 224), interpolation=cv2.INTER_LINEAR)
    imgs.append(im)
x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().div(255).to(dev)
x = (x - torch.tensor(ours.IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)) / torch.tensor(ours.IMAGENET_STD, device=dev).view(1, 3, 1, 1)

sd = torch.load(CKPT, map_location="cpu", weights_only=False)
t_off = off_teacher.surgery_teacher_vit_large_patch16()
r_off = t_off.load_state_dict(sd, strict=True)
t_off = t_off.to(dev).eval()
t_mine = ours.load_teacher(CKPT).to(dev)


def phase1_timm_teacher(final_norm=True, block_eps=1e-6):
    """Exactly how 03_pretrain/extract_teacher.py (phase 1) built the teacher, optionally corrected."""
    m = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0, no_embed_class=True)
    s = dict(sd)
    for i in range(len(m.blocks)):
        p = f"blocks.{i}.attn."
        s[p + "qkv.weight"] = torch.cat([s.pop(p + c + ".weight") for c in "qkv"])
        s[p + "qkv.bias"] = torch.cat([s.pop(p + c + ".bias") for c in "qkv"])
    m.load_state_dict(s, strict=False)
    for blk in m.blocks:
        blk.norm1.eps = blk.norm2.eps = block_eps
    m = m.to(dev).eval()
    return lambda z: (m.forward_features(z) if final_norm else
                      m.blocks(m._pos_embed(m.patch_embed(z))))[:, m.num_prefix_tokens:]


with torch.no_grad():
    f_off = t_off(x)
    f_mine = t_mine(x)
    f_p1 = phase1_timm_teacher()(x)
    f_p1_fixed = phase1_timm_teacher(final_norm=False, block_eps=1e-5)(x)
teacher = {
    "checkpoint": str(CKPT), "inputs": "2 frames from data/005, centre square crop, 224 px, ImageNet norm",
    "official_strict_load": str(r_off),
    "ours_vs_official": diff(f_mine, f_off),
    "phase1_timm_vs_official": diff(f_p1, f_off),
    "phase1_timm_no_final_norm_eps1e-5_vs_official": diff(f_p1_fixed, f_off),
    "official_feature_abs_max": float(f_off.abs().max()), "official_feature_std": float(f_off.std())}
report["teacher"] = teacher

out = HERE / "output"; out.mkdir(exist_ok=True)
(out / "official_equivalence.json").write_text(json.dumps(report, indent=2))
(out / "teacher_output_comparison.json").write_text(json.dumps(teacher, indent=2))
print(json.dumps(report, indent=2))
ok = (max(report["pos_embed"].values()) < 1e-6 and report["tube_mask"]["official_masked"] == report["tube_mask"]["ours_masked"]
      and all(report[f"student_{s}"][k]["rel_l2"] < 1e-5 for s in ["small", "base"] for k in ["image_decoder_out", "video_decoder_out"])
      and teacher["ours_vs_official"]["rel_l2"] < 1e-6)
print("EQUIVALENT TO OFFICIAL" if ok else "MISMATCH - see report")
sys.exit(0 if ok else 1)
