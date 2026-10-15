"""04_finetune, phase 2: frozen-feature linear probe — the cheap, fixed yardstick for the scaling curve.

Not part of the paper's protocol (the paper fine-tunes end to end); used only to compare checkpoints of OUR
pretraining at equal evaluation cost. Same clips, same official metric as finetune_official.py.

  feature = mean of the encoder's final tokens (eval transform: whole frame resized to 224x224)
  probe   = BatchNorm1d(affine=False) + Linear(768, 100), BCE, AdamW; epoch picked on val mAP_IVT; test once

  python 04_finetune/probe.py --name vmk400_ep0 --ckpt <work>/weights/videomae_base_k400_1600e.safetensors
  python 04_finetune/probe.py --name sv_ep2 --ckpt <work>/ckpt/p2_<run>/checkpoint-1.pth
Writes 04_finetune/output/probe_<name>.json; features cached in <cache>/probe_feats/<name>_<split>.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared")); sys.path.insert(0, str(HERE))
import encoders as E  # noqa: E402
import surgvista as sv  # noqa: E402
import triplet_metrics as tm  # noqa: E402
from finetune_official import CFG, TripletClips  # noqa: E402


@torch.no_grad()
def features(enc, ds, bs=8, workers=4, dev="cuda"):
    enc.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=workers, shuffle=False)
    f, rows = [], []
    for x, _, r in dl:
        f.append(enc.forward_tokens(x.to(dev, non_blocking=True)).mean(1).float().cpu().numpy()); rows.append(r.numpy().copy())  # copy: holding worker tensors keeps a file handle open each (Linux limit 1024)
    return np.concatenate(f), np.concatenate(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pos", choices=["auto", "1d", "3d"], default="auto")
    ap.add_argument("--train-stride", type=int, default=8)
    ap.add_argument("--val-stride", type=int, default=4)
    ap.add_argument("--test-stride", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", help="cpu works when features are already cached")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    sv.FAST_EVAL_ATTENTION = True  # eval only; rel diff ~1e-6 vs explicit attention

    fmt = E.read_checkpoint(a.ckpt)[1]
    pos = E.checkpoint_pos(a.ckpt) if a.pos == "auto" else a.pos
    dev = a.device
    enc = E.VideoEncoder(pos=pos, drop_path_rate=0.0, fc_drop_rate=0.0).to(dev)
    rep = E.load_start_weights(enc, a.ckpt)
    assert not [k for k in rep["missing"] if k.startswith(("patch_embed", "blocks"))], "encoder weights missing"

    fdir = Path(CFG["data"]["cache"]) / "probe_feats"; fdir.mkdir(exist_ok=True)
    data = {}
    for split, stride in [("train", a.train_stride), ("val", a.val_stride), ("test", a.test_stride)]:
        p = fdir / f"{a.name}_{split}_s{stride}.npz"
        ds = TripletClips(split, False, stride)
        if not p.exists():
            f, r = features(enc, ds, dev=dev)
            np.savez(p, f=f, r=r)
        z = np.load(p)
        o = np.lexsort((ds.frames[z["r"]], ds.videos[z["r"]]))
        data[split] = (z["f"][o], ds.labels[z["r"][o]], ds.videos[z["r"][o]])
        print(split, "clips", len(o), flush=True)

    Xtr, Ytr, _ = data["train"]
    head = nn.Sequential(nn.BatchNorm1d(Xtr.shape[1], affine=False), nn.Linear(Xtr.shape[1], 100)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    Xt, Yt = torch.tensor(Xtr).to(dev), torch.tensor(Ytr).to(dev)
    best, best_state, curve = -1, None, []
    for ep in range(a.epochs):
        head.train()
        for idx in torch.randperm(len(Xt), device=dev).split(256):
            loss = nn.functional.binary_cross_entropy_with_logits(head(Xt[idx]), Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 10 == 0:
            head.eval()
            with torch.no_grad():
                vx, vy, vv = data["val"]
                m = tm.official_map(head(torch.tensor(vx).to(dev)).cpu().numpy(), vy, vv)["mAP_ivt"]
            curve.append((ep + 1, round(m, 3)))
            if m > best:
                best, best_state = m, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state); head.eval()
    tx, ty, tv = data["test"]
    with torch.no_grad():
        tl = head(torch.tensor(tx).to(dev)).cpu().numpy()
    res = {k: v for k, v in tm.official_map(tl, ty, tv).items() if k.startswith("mAP")}
    out = {"name": a.name, "ckpt": a.ckpt, "format": fmt, "pos": pos, "val_best_mAP_ivt": best, "val_curve": curve,
           "test": res, "test_ci95": tm.bootstrap_ci(tl, ty, tv, n=200),
           "ci_method": "Bayesian block bootstrap (60-frame blocks) + bias-corrected percentile, 200 reps",
           "strides": {"train": a.train_stride, "val": a.val_stride, "test": a.test_stride},
           "clips": {k: len(v[0]) for k, v in data.items()}}
    (HERE / "output").mkdir(exist_ok=True)
    np.savez_compressed(HERE / "output" / f"probe_{a.name}_test_preds.npz", predictions=tl, labels=ty, video=tv)
    (HERE / "output" / f"probe_{a.name}.json").write_text(json.dumps(out, indent=2))
    print("PROBE", a.name, json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
