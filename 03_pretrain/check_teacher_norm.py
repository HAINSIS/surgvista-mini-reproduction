"""03_pretrain check: which input convention does the EndoSSL ViT-L teacher expect?

The official SurgVISTA code feeds it ImageNet-normalised crops; EndoSSL's own repo is inconsistent (0-255 in
inference.py, 0-1 in its RandAugment path) and its pretraining code is unreleased. Empirical test: frozen teacher
features (mean of the 196 patch tokens, official forward = no final norm) of single frames, whole frame resized to
224x224, under three conventions -> same linear probe -> official CholecT50 metric on val/test frames.
Only informs a possible "better" variant; the faithful reproduction keeps the official convention.

    work/.venv/Scripts/python.exe 03_pretrain/check_teacher_norm.py
Writes output/teacher_normalisation_check.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
import surgvista as sv  # noqa: E402
import triplet_metrics as tm  # noqa: E402
from surgvista_mini import load_cfg  # noqa: E402

CFG = load_cfg(); P2 = CFG["phase2"]
H, W = P2["frame_hw"]
MEAN = torch.tensor(sv.IMAGENET_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(sv.IMAGENET_STD).view(1, 3, 1, 1)
CONVENTIONS = {"imagenet_norm (official SurgVISTA)": lambda x: (x - MEAN) / STD,
               "raw_0_1": lambda x: x, "raw_0_255": lambda x: x * 255.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conventions", nargs="+", default=list(CONVENTIONS), help="subset of: " + ", ".join(CONVENTIONS))
    ap.add_argument("--strides", nargs=3, type=int, default=[10, 5, 5], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--out", default="teacher_normalisation_check.json")
    a = ap.parse_args()
    convs = {k: v for k, v in CONVENTIONS.items() if k in a.conventions or k.split(" ")[0] in a.conventions}
    df = pd.read_csv(HERE.parent / "02_data" / "output" / "index_full.csv")
    lab = np.load(Path(CFG["data"]["cache"]) / "labels_full.npz")["ivt"].astype(np.float32)
    mm = np.memmap(Path(CFG["data"]["cache"]) / f"frames_{W}x{H}.u8", np.uint8, "r", shape=(len(df), H, W, 3))
    pick = {}
    for split, stride in zip(["train", "val", "test"], a.strides):
        r = np.where((df.split.values == split) & df.cached.values)[0]
        pick[split] = r[::stride]
    teacher = sv.load_teacher(Path(CFG["work_root"]) / "ckpt" / "teacher_endossl_vitl.pth").cuda()
    feats = {c: {} for c in convs}
    with torch.no_grad():
        for split, rows in pick.items():
            for i in range(0, len(rows), 16):
                x = np.stack([np.asarray(Image.fromarray(np.asarray(mm[r])).resize((224, 224), Image.BILINEAR)) for r in rows[i:i + 16]])
                x = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0
                for c, f in convs.items():
                    feats[c].setdefault(split, []).append(teacher(f(x).cuda()).mean(1).cpu().numpy())
            print(split, len(rows), "frames", flush=True)
    report = {"frames": {k: int(len(v)) for k, v in pick.items()}, "strides": a.strides, "results": {}}
    for c in convs:
        X = {s: np.concatenate(feats[c][s]) for s in pick}
        torch.manual_seed(0)
        head = nn.Sequential(nn.BatchNorm1d(1024, affine=False), nn.Linear(1024, 100)).cuda()
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        Xt, Yt = torch.tensor(X["train"]).cuda(), torch.tensor(lab[pick["train"]]).cuda()
        best, state = -1, None
        for ep in range(200):
            head.train()
            for idx in torch.randperm(len(Xt), device="cuda").split(256):
                loss = nn.functional.binary_cross_entropy_with_logits(head(Xt[idx]), Yt[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            if (ep + 1) % 10 == 0:
                head.eval()
                with torch.no_grad():
                    v = tm.official_map(head(torch.tensor(X["val"]).cuda()).cpu().numpy(), lab[pick["val"]],
                                        df.video.values[pick["val"]])["mAP_ivt"]
                if v > best:
                    best, state = v, {k: t.clone() for k, t in head.state_dict().items()}
        head.load_state_dict(state); head.eval()
        with torch.no_grad():
            tl = head(torch.tensor(X["test"]).cuda()).cpu().numpy()
        res = tm.official_map(tl, lab[pick["test"]], df.video.values[pick["test"]])
        report["results"][c] = {"val_best_mAP_ivt": round(best, 2),
                                "test": {k: round(v, 2) for k, v in res.items() if k.startswith("mAP")},
                                "feature_std": float(X["train"].std())}
        print(c, report["results"][c], flush=True)
    (HERE / "output" / a.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
