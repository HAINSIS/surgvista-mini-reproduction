"""03_pretrain, phase 2: SurgVISTA pretraining with the OFFICIAL recipe.

Mirrors isyangshu/SurgVISTA@00ae4bc run_pretraining.py + engine_for_pretraining_surgery.py + pretrain.sh:
  * model  : _shared/surgvista.py (bit-exact with the official student/teacher, see check_official.py)
  * data   : every frame of every pretraining video is one sample; the clip is that frame and 15 earlier ones at
             stride 4, clamped at the video start (official _video_batch_loader); same multi-scale crop for all
             frames (scales 1/.875/.75/.66 of the 320-px side, max_distort 1, 13 fixed offsets), PIL bilinear
             resize to 224 for student and teacher, no flip, ImageNet normalisation
  * target : teacher on frames 0,2,4,... (time_stride_loss), raw tokens (no norm_feature), SmoothL1 x 0.05;
             normalised-pixel MSE x 1.0
  * optim  : AdamW(0.9, 0.95, eps 1e-8), wd 0.05 constant, no-decay for 1-D / bias / exact skip names;
             lr, min_lr, warmup_lr scaled by total_batch / 256; per-step cosine with linear warm-up from warmup_lr;
             no gradient clipping; fp16 autocast + loss scaling when amp is on
  * the plain-VideoMAE ablation (--no-distill) drops the teacher and the image decoder, nothing else

Single GPU or torchrun (DDP). Auto-resumes from <ckpt>/checkpoint-last.pth.

  # smoke test on the GTX 1070 (fp32, tiny batch, a few steps)
  python 03_pretrain/pretrain_official.py --name smoke --model small --batch-size 2 --update-freq 1 --no-amp --max-steps 20
  # the real run (cloud), official values from config phase2.pretrain
  torchrun --nproc_per_node 8 03_pretrain/pretrain_official.py --name settingA
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
import encoders as E  # noqa: E402
import surgvista as sv  # noqa: E402
from surgvista_mini import load_cfg  # noqa: E402

CFG = load_cfg()
P2 = CFG["phase2"]


# ---------------------------------------------------------------- data
class PretrainClips(torch.utils.data.Dataset):
    def __init__(self, pc, splits, clip_stride=1):
        df = pd.read_csv(HERE.parent / "02_data" / "output" / "index_full.csv")
        self.H, self.W = P2["frame_hw"]
        self.mm_path = Path(CFG["data"]["cache"]) / f"frames_{self.W}x{self.H}.u8"
        self.n_total = len(df)
        start = df.groupby("video").cumcount().values  # == frame id (frames are 0..n-1 per video)
        self.video_start = np.arange(len(df)) - start
        self.T, self.rate = pc["num_frames"], pc["sampling_rate"]
        keep = df.split.isin(splits).values & df.cached.values
        keep &= (np.arange(len(df)) - self.video_start) % clip_stride == 0  # budget knob; official: 1
        # a sample is usable only if all frames of its clip are cached (matters while the download runs)
        ok = [r for r in np.where(keep)[0] if df.cached.values[self.clip_rows(r)].all()]
        self.rows = np.array(ok)
        self.size, self.tsize = pc["input_size"], pc["teacher_input_size"]
        self.scales = pc["scales"]
        self.masker = sv.TubeMaskingGenerator((self.T // pc["tubelet"], self.size // 16, self.size // 16), pc["mask_ratio"])
        self.mean = np.array(sv.IMAGENET_MEAN, np.float32)[:, None, None, None]
        self.std = np.array(sv.IMAGENET_STD, np.float32)[:, None, None, None]
        self.mm = None

    def clip_rows(self, r):
        f, ids = r - self.video_start[r], []
        for _ in range(self.T):  # official: step back by rate while it stays >= 0, else repeat
            ids.append(f)
            if f - self.rate >= 0:
                f -= self.rate
        return self.video_start[r] + np.array(sorted(ids))

    def crop_box(self):
        """GroupMultiScaleTwoResizedCrop._sample_crop_size with fix_crop=more_fix_crop=True, max_distort=1."""
        base = min(self.W, self.H)
        sizes = [int(base * s) for s in self.scales]
        sizes = [self.size if abs(x - self.size) < 3 else x for x in sizes]
        pairs = [(w, h) for i, h in enumerate(sizes) for j, w in enumerate(sizes) if abs(i - j) <= 1]
        cw, ch = random.choice(pairs)
        ws, hs = (self.W - cw) // 4, (self.H - ch) // 4
        offs = [(0, 0), (4 * ws, 0), (0, 4 * hs), (4 * ws, 4 * hs), (2 * ws, 2 * hs), (0, 2 * hs), (4 * ws, 2 * hs),
                (2 * ws, 4 * hs), (2 * ws, 0), (ws, hs), (3 * ws, hs), (ws, 3 * hs), (3 * ws, 3 * hs)]
        ox, oy = random.choice(offs)
        return ox, oy, cw, ch

    def _to_tensor(self, frames):
        x = np.stack([np.asarray(f, np.float32) for f in frames]).transpose(3, 0, 1, 2) / 255.0  # C T H W
        return torch.from_numpy((x - self.mean) / self.std)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.mm is None:
            self.mm = np.memmap(self.mm_path, np.uint8, "r", shape=(self.n_total, self.H, self.W, 3))
        ox, oy, cw, ch = self.crop_box()
        crops = [Image.fromarray(np.ascontiguousarray(self.mm[r, oy:oy + ch, ox:ox + cw])) for r in self.clip_rows(self.rows[i])]
        stu = self._to_tensor([c.resize((self.size, self.size), Image.BILINEAR) for c in crops])
        tea = stu if self.tsize == self.size else self._to_tensor([c.resize((self.tsize, self.tsize), Image.BILINEAR) for c in crops])
        return stu, tea, torch.from_numpy(self.masker()).bool()


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s); random.seed(s)


# ---------------------------------------------------------------- optimisation (official helpers, re-implemented)
def param_groups(model, wd):
    skip = model.no_weight_decay()
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim == 1 or n.endswith(".bias") or n in skip) else decay).append(p)
    return [{"params": no_decay, "weight_decay": 0.0, "lr_scale": 1.0},
            {"params": decay, "weight_decay": wd, "lr_scale": 1.0}]


def cosine_scheduler(base, final, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0.0):
    warm_iters = warmup_epochs * niter_per_ep
    warm = np.linspace(start_warmup_value, base, warm_iters) if warmup_epochs > 0 else np.array([])
    iters = np.arange(epochs * niter_per_ep - warm_iters)
    main = final + 0.5 * (base - final) * (1 + np.cos(np.pi * iters / len(iters)))
    s = np.concatenate((warm, main))
    assert len(s) == epochs * niter_per_ep
    return s


# ---------------------------------------------------------------- main
def main():
    pc = dict(P2["pretrain"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run name -> <work_root>/ckpt/p2_<name>/, output/p2_<name>_log.csv")
    ap.add_argument("--model", default=pc["model"], choices=list(sv.STUDENT_SIZES))
    ap.add_argument("--batch-size", type=int, default=pc["batch_size"])
    ap.add_argument("--update-freq", type=int, default=pc["update_freq"])
    ap.add_argument("--epochs", type=int, default=pc["epochs"])
    ap.add_argument("--warmup-epochs", type=int, default=pc["warmup_epochs"])
    ap.add_argument("--workers", type=int, default=pc["num_workers"])
    ap.add_argument("--no-amp", action="store_true", help="fp32 (Pascal GPUs have no fast fp16)")
    ap.add_argument("--no-distill", action="store_true", help="plain VideoMAE ablation")
    ap.add_argument("--splits", nargs="+", default=P2["pretrain_splits"])
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (smoke tests)")
    ap.add_argument("--seed", type=int, default=pc["seed"])
    ap.add_argument("--init", default=None, help="start weights (we warm-start; pretraining from scratch is out of our compute budget)")
    ap.add_argument("--pos", choices=["auto", "1d", "3d"], default="auto", help="auto: 1d for VideoMAE weights, else 3d")
    ap.add_argument("--clip-stride", type=int, default=1, help="every k-th frame as a sample (official: 1)")
    ap.add_argument("--save-every", type=int, default=0, help="also keep checkpoint-<ep>.pth every N epochs")
    a = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    dev = torch.device("cuda")
    seed = a.seed + rank
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    amp = pc["amp"] and not a.no_amp

    ds = PretrainClips(pc, a.splits, a.clip_stride)
    sampler = (torch.utils.data.DistributedSampler(ds, world, rank, shuffle=True) if ddp
               else torch.utils.data.RandomSampler(ds))
    dl = torch.utils.data.DataLoader(ds, sampler=sampler, batch_size=a.batch_size, num_workers=a.workers,
                                     pin_memory=True, drop_last=True, worker_init_fn=seed_worker,
                                     persistent_workers=a.workers > 0)

    model = sv.build_student(a.model, pc["input_size"], pc["num_frames"], pc["tubelet"],
                             decoder_depth_video=pc["decoder_depth_video"], decoder_depth_image=pc["decoder_depth_image"],
                             feat_decoder_embed_dim=pc["feat_decoder_embed_dim"],
                             feat_decoder_num_heads=pc["feat_decoder_num_heads"],
                             target_feature_dim=pc["distillation_target_dim"], drop_path_rate=pc["drop_path"],
                             with_image_decoder=not a.no_distill,
                             pos=(("3d" if a.init is None else E.checkpoint_pos(a.init)) if a.pos == "auto" else a.pos))
    init_report = E.load_start_weights(model, a.init) if a.init else None
    model = model.to(dev)
    raw_model = model
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index or 0])
    teacher = None if a.no_distill else sv.load_teacher(Path(CFG["work_root"]) / "ckpt" / "teacher_endossl_vitl.pth").to(dev)

    total_batch = a.batch_size * a.update_freq * world
    niter = len(ds) // total_batch
    lr, min_lr, warmup_lr = (pc[k] * total_batch / 256 for k in ["lr", "min_lr", "warmup_lr"])
    lr_sched = cosine_scheduler(lr, min_lr, a.epochs, niter, a.warmup_epochs, warmup_lr)
    wd_sched = cosine_scheduler(pc["weight_decay"], pc["weight_decay"], a.epochs, niter)
    opt = torch.optim.AdamW(param_groups(raw_model, pc["weight_decay"]), lr=lr, betas=tuple(pc["betas"]), eps=pc["opt_eps"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    ck = Path(CFG["work_root"]) / "ckpt" / f"p2_{a.name}"; ck.mkdir(parents=True, exist_ok=True)
    log_path = HERE / "output" / f"p2_{a.name}_log.csv"; log_path.parent.mkdir(exist_ok=True)
    start_ep, log = 0, []
    if (ck / "checkpoint-last.pth").exists():
        s = torch.load(ck / "checkpoint-last.pth", map_location="cpu", weights_only=False)
        raw_model.load_state_dict(s["model"]); opt.load_state_dict(s["optimizer"]); scaler.load_state_dict(s["scaler"])
        start_ep = s["epoch"] + 1
        log = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []
        if rank == 0:
            print(f"resumed from epoch {s['epoch']}")
    if rank == 0:
        meta = {"args": vars(a), "pretrain_cfg": pc, "pos": raw_model.pos_kind,
                "init_report": init_report and {k: init_report[k] for k in ["format", "loaded", "missing", "shape_mismatch"]}, "samples": len(ds), "total_batch": total_batch, "iters_per_epoch": niter,
                "lr": lr, "min_lr": min_lr, "warmup_lr": warmup_lr, "amp": amp, "world": world,
                "params_M": sum(p.numel() for p in raw_model.parameters()) / 1e6,
                "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
        (ck / "run_meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2), flush=True)
    assert niter > 0, "dataset smaller than one total batch"

    w_img, w_vid = pc["image_teacher_loss_weight"], pc["video_reconstruction_loss_weight"]
    step_global, t0 = start_ep * niter, time.time()
    for ep in range(start_ep, a.epochs):
        if ddp:
            sampler.set_epoch(ep)
        model.train()
        acc, n = {"loss": 0.0, "loss_vid": 0.0, "loss_img": 0.0}, 0
        opt.zero_grad(set_to_none=True)
        for step, (x, xt, mask) in enumerate(dl):
            if step // a.update_freq >= niter:
                break
            it = ep * niter + step // a.update_freq
            for g in opt.param_groups:
                g["lr"] = lr_sched[it] * g["lr_scale"]
                if g["weight_decay"] > 0:
                    g["weight_decay"] = wd_sched[it]
            x, xt, mask = x.to(dev, non_blocking=True), xt.to(dev, non_blocking=True), mask.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                with torch.no_grad():
                    tf = sv.teacher_targets(teacher, xt, pc["tubelet"]) if teacher is not None else None
                res = sv.surgvista_loss(model if not ddp else _DDPView(model, raw_model), x, mask, tf,
                                        w_img=w_img, w_vid=w_vid)
            if not math.isfinite(res["loss"].item()):
                sys.exit(f"loss is {res['loss'].item()}, stopping (as official)")
            scaler.scale(res["loss"] / a.update_freq).backward()
            if (step + 1) % a.update_freq == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                step_global += 1
            for k in acc:
                acc[k] += res[k].item() if k in res else 0.0
            n += 1
            if rank == 0 and step % 50 == 0:
                print(f"ep {ep} step {step} it {it} lr {lr_sched[it]:.3e} loss {res['loss'].item():.4f} "
                      f"vid {res['loss_vid'].item():.4f} img {float(res.get('loss_img', 0)):.4f} "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if a.max_steps and step_global >= a.max_steps:
                break
        if rank == 0:
            log.append({"epoch": ep, "minutes": round((time.time() - t0) / 60, 2), "lr": float(lr_sched[it]),
                        **{k: round(v / max(n, 1), 5) for k, v in acc.items()}})
            pd.DataFrame(log).to_csv(log_path, index=False)
            print(log[-1], flush=True)
            state = {"model": raw_model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": ep, "args": vars(a), "pos": raw_model.pos_kind}
            torch.save(state, ck / "checkpoint-last.pth")
            if (ep + 1) % pc["save_ckpt_freq"] == 0 or ep + 1 == a.epochs or (a.save_every and (ep + 1) % a.save_every == 0):
                torch.save({"model": raw_model.state_dict(), "epoch": ep, "pos": raw_model.pos_kind}, ck / f"checkpoint-{ep}.pth")
        if a.max_steps and step_global >= a.max_steps:
            break
    if ddp:
        dist.destroy_process_group()


class _DDPView:
    """surgvista_loss calls model(x, mask) and reads .tubelet_size; route the call through DDP for gradient sync."""

    def __init__(self, ddp_model, raw):
        self._ddp, self.tubelet_size = ddp_model, raw.tubelet_size

    def __call__(self, x, mask):
        return self._ddp(x, mask)


if __name__ == "__main__":
    main()
