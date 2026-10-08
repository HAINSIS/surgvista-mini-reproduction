"""Linear probe on FROZEN features: measures representation quality without letting a 22 M-param
encoder memorise 40 training videos (which is what end-to-end fine-tuning does at this data scale).

  python 04_finetune/linear_probe.py

Encoders: random init, ImageNet (inflated ViT-S), the 03_pretrain encoders whose checkpoint exists (surgvista,
surgvista_in, videomae), plus the surgical IMAGE foundation model itself (EndoSSL ViT-L teacher, last frame of the clip; features already cached).
Per class: standardised features -> logistic regression (class-balanced); mAP on the 10 held-out videos.
Writes output/linear_probe.json, output/linear_probe.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
from clips import clip_index, gather  # noqa: E402
from surgvista_mini import MEAN, STD, VideoViT, inflate_imagenet_vit_s, load_cfg  # noqa: E402

cfg = load_cfg(); mc = cfg["model"]; dev = "cuda"
cache = Path(cfg["data"]["cache"]); ck = Path(cfg["work_root"]) / "ckpt"; out = HERE / "output"
df = pd.read_csv(HERE.parent / "02_data" / "output" / "index.csv")
classes = json.load(open(HERE.parent / "02_data" / "output" / "classes.json"))
groups, off = {}, 0
for k, short in zip(["instruments", "verbs", "targets", "triplets"], ["I", "V", "T", "IVT"]):
    groups[short] = (off, off + len(classes[k])); off += len(classes[k])
Y = np.load(cache / "labels.npy")
cidx = clip_index(df, mc["num_frames"], mc["sampling_rate"])
tr = df.split.values == "train"; te = ~tr
frames = torch.from_numpy(np.load(cache / "frames112.npy")).to(dev)
mean, std = MEAN.to(dev), STD.to(dev)


def video_feats(init):
    torch.manual_seed(0)
    m = VideoViT(cfg["data"]["img_size"], mc["patch"], mc["tubelet"], mc["num_frames"], mc["embed_dim"], mc["depth"], mc["heads"])
    if init == "imagenet":
        inflate_imagenet_vit_s(m)
    elif init != "scratch":
        print(init, m.load_state_dict(torch.load(ck / f"{init}_encoder.pth"), strict=True))
    m.eval().to(dev); F = []
    with torch.no_grad():
        for i in range(0, len(df), 32):
            x = gather(frames, torch.from_numpy(cidx[i:i + 32]).to(dev))
            F.append(m.forward_features((x - mean) / std).mean(1).cpu())
    return torch.cat(F).numpy()


feats = {a: video_feats(a) for a in ["scratch", "imagenet", "videomae", "surgvista", "surgvista_in"]
         if a in ("scratch", "imagenet") or (ck / f"{a}_encoder.pth").exists()}
feats["endossl_teacher"] = np.load(cache / "teacher_feats.npy", mmap_mode="r").astype(np.float32).mean(1)

res = {}
for name, Fx in feats.items():
    sc = StandardScaler().fit(Fx[tr]); Xtr, Xte = sc.transform(Fx[tr]), sc.transform(Fx[te])
    P = np.zeros((te.sum(), Y.shape[1]))
    for c in range(Y.shape[1]):
        if 0 < Y[tr, c].sum() < tr.sum():
            P[:, c] = LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000).fit(Xtr, Y[tr, c]).predict_proba(Xte)[:, 1]
    r = {}
    for g, (a, b) in groups.items():
        keep = Y[te, a:b].sum(0) > 0
        r[f"mAP_{g}"] = float(average_precision_score(Y[te, a:b][:, keep], P[:, a:b][:, keep], average="macro"))
    res[name] = r; print(name, {k: round(100 * v, 1) for k, v in r.items()}, flush=True)

# chance level = class prevalence in the test set (expected AP of a random ranking)
r = {}
for g, (a, b) in groups.items():
    keep = Y[te, a:b].sum(0) > 0
    r[f"mAP_{g}"] = float(Y[te, a:b][:, keep].mean(0).mean())
res["chance"] = r
json.dump(res, open(out / "linear_probe.json", "w"), indent=1)
LAB = {"chance": "Chance (test prevalence)", "scratch": "Random-init video ViT-S", "imagenet": "ImageNet ViT-S (natural images)",
       "videomae": "VideoMAE-mini (surgical video, no distill.)", "surgvista": "SurgVISTA-mini from scratch", "surgvista_in": "SurgVISTA-mini from ImageNet init",
       "endossl_teacher": "EndoSSL ViT-L (surgical image FM = teacher, 300 M params)"}
lines = ["# Linear probe on frozen features — CholecT50, 10 held-out videos", "",
         "| Encoder | mAP_I | mAP_V | mAP_T | mAP_IVT |", "|---|---|---|---|---|"]
for k in ["chance", "scratch", "imagenet", "videomae", "surgvista", "surgvista_in", "endossl_teacher"]:
    if k in res:
        lines.append(f"| {LAB[k]} | " + " | ".join(f"{100 * res[k][f'mAP_{g}']:.1f}" for g in groups) + " |")
(out / "linear_probe.md").write_text("\n".join(lines), encoding="utf8"); print("\n".join(lines))
