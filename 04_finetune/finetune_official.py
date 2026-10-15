"""04_finetune, phase 2: CholecT50 triplet fine-tuning with the OFFICIAL SurgVISTA recipe.

Mirrors isyangshu/SurgVISTA@00ae4bc downstream/scripts/triplet/train_triplet_video.sh + run_triplet_training.py +
engine_for_triplet.py + datasets/datasets4finetuning/CholecT50_triplet.py:
  * model : _shared/encoders.VideoEncoder (unified_base / unified_base_st), mean pool -> fc_norm -> dropout 0.5 ->
            zero-init head (100 triplets), drop_path 0.1; 3-D pos for SurgVISTA/MVD weights, 1-D for VideoMAE
  * clips : every frame of a train video is a sample; 16 frames back at stride 4, clamped at the video start
  * train aug : RandAugment rand-m7-n4-mstd0.5-inc1 (same ops for all frames, bicubic), Inception crop
                (scale 0.7-1, ratio 3/4-4/3) to 224, h-flip 0.5, random erasing p 0.25 (pixel); eval: resize to 224x224
  * optim : AdamW (0.9, 0.999), wd 0.05, layer decay 0.75, lr 5e-4 / min_lr 1e-6 / warmup_lr 1e-6 all x total_batch/64,
            5 warm-up epochs, per-step cosine, 20 epochs, total batch 64, BCEWithLogits, no grad clipping
  * select: best val mAP_IVT (official metric, per video) -> evaluate that checkpoint once on the test videos
Budget knobs for the GTX 1070 (each one is recorded in run_meta.json and must be reported):
  --train-stride k : use every k-th frame as a training sample (official: 1)
  --val-stride k   : validate on every k-th val frame (official: 1); test always uses every frame
Outputs: <work_root>/ckpt/ft_<name>/{best,last}.pth, run_meta.json; 04_finetune/output/ft_<name>_{log.csv,metrics.json,test_preds.npz}

  python 04_finetune/finetune_official.py --name vmk400_pilot --init <work>/weights/videomae_base_k400_1600e.safetensors \
      --batch-size 2 --update-freq 32 --no-amp --train-stride 8 --val-stride 4
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data.auto_augment import rand_augment_transform
from timm.data.random_erasing import RandomErasing

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
import encoders as E  # noqa: E402
import surgvista as sv  # noqa: E402
import triplet_metrics as tm  # noqa: E402
from surgvista_mini import load_cfg  # noqa: E402

CFG = load_cfg()
P2 = CFG["phase2"]
FT = P2["finetune"]


class TripletClips(torch.utils.data.Dataset):
    def __init__(self, split, train, stride=1, frames="cache", keep_empty=False):
        # frames: 'cache' = 512x320 memmap (pretraining resolution); 'png' = the original 854x480 frames, as the
        # official CholecT50 fine-tuning loader reads them (datasets4finetuning/CholecT50_triplet.py)
        # keep_empty=False: like the official label builder (generate_pkl_cholect50.py skips triplet id -1), frames
        # with no triplet are never a sample (train/val/test); they still appear as earlier frames inside clips
        df = pd.read_csv(HERE.parent / "02_data" / "output" / "index_full.csv")
        self.H, self.W = P2["frame_hw"]
        self.mm_path = Path(CFG["data"]["cache"]) / f"frames_{self.W}x{self.H}.u8"
        self.n_total = len(df)
        self.video_start = np.arange(len(df)) - df.groupby("video").cumcount().values
        self.T, self.rate = FT["num_frames"], FT["sampling_rate"]
        self.labels = np.load(Path(CFG["data"]["cache"]) / "labels_full.npz")["ivt"].astype(np.float32)
        cand = np.where(df.split.values == split)[0]
        cand = cand[(cand - self.video_start[cand]) % stride == 0]
        if not keep_empty:
            cand = cand[self.labels[cand].sum(1) > 0]
        cached = df.cached.values
        self.rows = np.array([r for r in cand if cached[self.clip_rows(r)].all()])
        self.videos, self.frames = df.video.values, df.frame.values
        self.train = train
        self.size = FT["input_size"]
        self.mean = torch.tensor(sv.IMAGENET_MEAN).view(3, 1, 1, 1)
        self.std = torch.tensor(sv.IMAGENET_STD).view(3, 1, 1, 1)
        if train:
            self.aa = rand_augment_transform(FT["aa"], {"translate_const": int(self.size * 0.45),
                                                         "interpolation": Image.BICUBIC})
            self.erase = RandomErasing(FT["reprob"], mode="pixel", max_count=1, num_splits=1, device="cpu")
        self.mm = None
        self.frames_src = frames
        self.raw = Path(CFG["data"]["raw_dir"])
        self.rel = df.rel.values

    def clip_rows(self, r):
        f, ids = r - self.video_start[r], []
        for _ in range(self.T):
            ids.append(f)
            if f - self.rate >= 0:
                f -= self.rate
        return self.video_start[r] + np.array(sorted(ids))

    def __len__(self):
        return len(self.rows)

    NATIVE_WH = (854, 480)  # CholecT50 frame size; the HF copy has some frames at other sizes

    def _png(self, k):
        """Original frame, as the official loader reads it. The official loader stacks a clip's frames directly, so it
        assumes one size; frames stored at another size in the HF copy are brought to 854x480 (PIL bilinear) first."""
        im = Image.open(self.raw / self.rel[k]).convert("RGB")
        return im if im.size == self.NATIVE_WH else im.resize(self.NATIVE_WH, Image.BILINEAR)

    def _train_clip(self, frames):
        state = random.getstate()
        imgs = []
        for f in frames:  # identical RandAugment ops/magnitudes for every frame of the clip
            random.setstate(state)
            imgs.append(np.asarray(self.aa(Image.fromarray(f)), np.float32))
        x = torch.from_numpy(np.stack(imgs)).permute(3, 0, 1, 2) / 255.0  # C T H W
        x = (x - self.mean) / self.std
        H, W = x.shape[-2:]
        i, j, h, w = _inception_crop_params(H, W, FT["crop_scale"], FT["crop_ratio"])
        x = F.interpolate(x[:, :, i:i + h, j:j + w], size=(self.size, self.size), mode="bilinear", align_corners=False)
        if random.random() < 0.5:
            x = x.flip(-1)
        return self.erase(x.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)

    def _eval_clip(self, frames):
        x = np.stack([np.asarray(Image.fromarray(f).resize((self.size, self.size), Image.BILINEAR), np.float32) for f in frames])
        x = torch.from_numpy(x).permute(3, 0, 1, 2) / 255.0
        return (x - self.mean) / self.std

    def __getitem__(self, i):
        if self.mm is None and self.frames_src == "cache":
            self.mm = np.memmap(self.mm_path, np.uint8, "r", shape=(self.n_total, self.H, self.W, 3))
        r = self.rows[i]
        if self.frames_src == "png":
            frames = [np.asarray(self._png(k)) for k in self.clip_rows(r)]
        else:
            frames = [np.asarray(self.mm[k]) for k in self.clip_rows(r)]
        x = self._train_clip(frames) if self.train else self._eval_clip(frames)
        return x, torch.from_numpy(self.labels[r]), r


def _inception_crop_params(H, W, scale, ratio):
    """SlowFast _get_param_spatial_crop (used by the official random_resized_crop)."""
    for _ in range(10):
        area = H * W * random.uniform(*scale)
        logr = (math.log(ratio[0]), math.log(ratio[1]))
        ar = math.exp(random.uniform(*logr))
        w, h = int(round(math.sqrt(area * ar))), int(round(math.sqrt(area / ar)))
        if 0 < w <= W and 0 < h <= H:
            return random.randint(0, H - h), random.randint(0, W - w), h, w
    in_ratio = W / H  # fallback: centre crop
    if in_ratio < min(ratio):
        w, h = W, int(round(W / min(ratio)))
    elif in_ratio > max(ratio):
        h, w = H, int(round(H * max(ratio)))
    else:
        w, h = W, H
    return (H - h) // 2, (W - w) // 2, h, w


def seed_worker(_):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s); random.seed(s)


def layer_decay_groups(model, wd, decay):
    n = model.get_num_layers() + 2
    scales = [decay ** (n - 1 - i) for i in range(n)]
    skip = model.no_weight_decay()
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = p.ndim == 1 or name.endswith(".bias") or name in skip
        lid = 0 if name.startswith(("patch_embed", "pos_embed", "cls_token")) else (
            int(name.split(".")[1]) + 1 if name.startswith("blocks.") else n - 1)
        key = (lid, no_decay)
        g = groups.setdefault(key, {"params": [], "weight_decay": 0.0 if no_decay else wd, "lr_scale": scales[lid]})
        g["params"].append(p)
    return list(groups.values())


def cosine(base, final, epochs, niter, warm_epochs, start):
    warm = np.linspace(start, base, warm_epochs * niter) if warm_epochs > 0 else np.array([])
    it = np.arange(epochs * niter - len(warm))
    return np.concatenate((warm, final + 0.5 * (base - final) * (1 + np.cos(np.pi * it / len(it)))))


@torch.no_grad()
def predict(model, ds, dev, amp, bs=8, workers=4):
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=workers, shuffle=False)
    logits, rows = [], []
    for x, _, r in dl:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits.append(model(x.to(dev, non_blocking=True)).float().cpu().numpy())
        rows.append(r.numpy().copy())  # copy: holding worker tensors keeps a file handle open each (Linux limit 1024)
    rows = np.concatenate(rows)
    return np.concatenate(logits), rows


def score(ds, logits, rows):
    order = np.lexsort((ds.frames[rows], ds.videos[rows]))
    return tm.official_map(logits[order], ds.labels[rows][order], ds.videos[rows][order]), order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--init", required=True, help="start checkpoint (.pth / .safetensors) or 'scratch'")
    ap.add_argument("--pos", choices=["auto", "1d", "3d"], default="auto")
    ap.add_argument("--batch-size", type=int, default=FT["batch_size"])
    ap.add_argument("--update-freq", type=int, default=FT["update_freq"])
    ap.add_argument("--epochs", type=int, default=FT["epochs"])
    ap.add_argument("--warmup-epochs", type=int, default=FT["warmup_epochs"])
    ap.add_argument("--train-stride", type=int, default=1)
    ap.add_argument("--val-stride", type=int, default=1)
    ap.add_argument("--workers", type=int, default=FT["num_workers"])
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", choices=["cache", "png"], default="cache",
                    help="png = original 854x480 frames, as the official CholecT50 loader (default: 512x320 cache)")
    ap.add_argument("--keep-empty", action="store_true",
                    help="keep frames with no triplet as samples (runs before 2026-10-13); official code drops them")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke tests only")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    sv.FAST_EVAL_ATTENTION = True  # only affects model.eval() passes (val/test); training uses explicit attention
    dev, amp = torch.device("cuda"), FT["amp"] and not a.no_amp
    pos = a.pos
    if pos == "auto":
        pos = "3d" if a.init == "scratch" else E.checkpoint_pos(a.init)
    model = E.VideoEncoder(num_classes=100, num_frames=FT["num_frames"], drop_path_rate=FT["drop_path"],
                           fc_drop_rate=FT["fc_drop_rate"], pos=pos).to(dev)
    init_report = None if a.init == "scratch" else E.load_start_weights(model, a.init)

    tr = TripletClips("train", True, a.train_stride, a.frames, a.keep_empty)
    va = TripletClips("val", False, a.val_stride, a.frames, a.keep_empty)
    te = TripletClips("test", False, 1, a.frames, a.keep_empty)
    dl = torch.utils.data.DataLoader(tr, batch_size=a.batch_size, shuffle=True, num_workers=a.workers, drop_last=True,
                                     pin_memory=True, worker_init_fn=seed_worker, persistent_workers=a.workers > 0)
    total = a.batch_size * a.update_freq
    niter = len(tr) // total
    lr, min_lr, warm_lr = (FT[k] * total / 64 for k in ["lr", "min_lr", "warmup_lr"])
    sched = cosine(lr, min_lr, a.epochs, niter, a.warmup_epochs, warm_lr)
    opt = torch.optim.AdamW(layer_decay_groups(model, FT["weight_decay"], FT["layer_decay"]), lr=lr,
                            betas=tuple(FT["betas"]), eps=FT["opt_eps"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    crit = torch.nn.BCEWithLogitsLoss()

    ck = Path(CFG["work_root"]) / "ckpt" / f"ft_{a.name}"; ck.mkdir(parents=True, exist_ok=True)
    out = HERE / "output"; out.mkdir(exist_ok=True)
    meta = {"args": vars(a), "pos": pos, "init_report": init_report and {k: init_report[k] for k in ["format", "loaded", "missing", "shape_mismatch"]},
            "finetune_cfg": FT, "train_clips": len(tr), "val_clips": len(va), "test_clips": len(te),
            "total_batch": total, "iters_per_epoch": niter, "lr": lr, "min_lr": min_lr, "amp": amp,
            "deviations": {"train_stride": a.train_stride, "val_stride": a.val_stride, "epochs": a.epochs,
                           "frame_source": ("original 854x480 PNGs (official)" if a.frames == "png" else "512x320 cache (official: original PNGs)"),
                           "attention": "SDPA" if sv.FAST_ATTENTION else "explicit",
                           "empty_frames": ("kept as all-negative samples (official: dropped)" if a.keep_empty
                                            else "dropped, as the official label builder")},
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    (ck / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ["pos", "train_clips", "val_clips", "test_clips", "total_batch", "iters_per_epoch", "lr"]}), flush=True)
    assert niter > 0

    start_ep, best, log = 0, -1.0, []
    if (ck / "last.pth").exists():
        s = torch.load(ck / "last.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["optimizer"]); scaler.load_state_dict(s["scaler"])
        start_ep, best, log = s["epoch"] + 1, s["best"], s["log"]
        print("resumed from epoch", s["epoch"])

    t0, steps = time.time(), 0
    for ep in range(start_ep, a.epochs):
        model.train(); opt.zero_grad(set_to_none=True); run = 0.0
        for step, (x, y, _) in enumerate(dl):
            if step // a.update_freq >= niter:
                break
            it = ep * niter + step // a.update_freq
            for g in opt.param_groups:
                g["lr"] = sched[it] * g["lr_scale"]
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = crit(model(x.to(dev, non_blocking=True)), y.to(dev, non_blocking=True))
            scaler.scale(loss / a.update_freq).backward()
            if (step + 1) % a.update_freq == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); steps += 1
            run += loss.item()
            if step % 200 == 0:
                print(f"ep {ep} step {step}/{niter * a.update_freq} lr {sched[it]:.2e} loss {loss.item():.4f} "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if a.max_steps and steps >= a.max_steps:
                break
        vl, vr = predict(model, va, dev, amp)
        vm, _ = score(va, vl, vr)
        row = {"epoch": ep, "minutes": round((time.time() - t0) / 60, 1), "train_loss": round(run / max(step + 1, 1), 5),
               **{f"val_{k}": round(v, 3) for k, v in vm.items() if k.startswith("mAP")}}
        log.append(row); print(row, flush=True)
        pd.DataFrame(log).to_csv(out / f"ft_{a.name}_log.csv", index=False)
        if vm["mAP_ivt"] > best:
            best = vm["mAP_ivt"]
            torch.save({"model": model.state_dict(), "epoch": ep, "val": vm}, ck / "best.pth")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": ep, "best": best, "log": log}, ck / "last.pth")
        if a.max_steps and steps >= a.max_steps:
            break

    # final test with the best-on-val checkpoint (official), all test frames, once
    s = torch.load(ck / "best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(s["model"])
    tl, tr_rows = predict(model, te, dev, amp)
    res, order = score(te, tl, tr_rows)
    tl, tr_rows = tl[order], tr_rows[order]
    ci = tm.bootstrap_ci(tl, te.labels[tr_rows], te.videos[tr_rows], frames=te.frames[tr_rows])
    metrics = {"name": a.name, "best_val_epoch": s["epoch"], "val_at_best": {k: v for k, v in s["val"].items() if k.startswith("mAP")},
               "test": {k: v for k, v in res.items() if k.startswith("mAP")}, "test_ci95": ci, "meta": meta}
    (out / f"ft_{a.name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    classes = json.load(open(HERE.parent / "02_data" / "output" / "classes_official.json"))["triplet"]
    np.savez_compressed(out / f"ft_{a.name}_test_preds.npz", predictions=tl, labels=te.labels[tr_rows],
                        video=te.videos[tr_rows], frame=te.frames[tr_rows], class_names=np.array(classes))
    print("TEST", json.dumps(metrics["test"]), "CI95", json.dumps(ci), flush=True)


if __name__ == "__main__":
    main()
