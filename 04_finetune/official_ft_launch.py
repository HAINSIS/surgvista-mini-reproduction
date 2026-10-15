"""Run the OFFICIAL SurgVISTA triplet fine-tuning script (downstream_triplet/run_triplet_training.py) unchanged.

The clone at <work_root>/ref/SurgVISTA is never edited. This launcher only makes it importable/runnable here;
every shim is listed, none touches the model, data, optimiser, schedule or metric:
  1. sys.path: the clone root + downstream/ (the script appends the authors' own /home/syangcw/... paths).
  2. `downstream.ivtmetrics`: the engine imports it from there, but the clone ships it at
     downstream/downstream_triplet/ivtmetrics (and it imports itself as `ivtmetrics`) -> registered under both
     names (same files).
  3. `decord` / `tensorboardX`: stubbed only if not installed. decord is imported but never used by the CholecT50
     loader; tensorboardX only writes TensorBoard logs.
  3b. `numpy.lib.function_base` (removed in numpy 2): the loaders import `disp` from it and never call it ->
     stub module when missing.
  3c. timm >= 0.9 renamed some optimisers that datasets/transforms/optim_factory.py imports by their old names
     (e.g. Nadam). Missing names get a placeholder that raises if used; the recipe uses AdamW (or DeepSpeed's).
  3d. timm >= 0.9 create_model passes pretrained_cfg/... to the model function, which the official
     VisionTransformer rejects -> timm.models.create_model restored to the timm 0.4.x behaviour the code was written
     for: drop None kwargs, call the registered model function with pretrained + the remaining kwargs.
  3e. `downstream.model.unifiedmodel_image` (image-backbone models) is imported but not in the public repo; the
     triplet video runs never use it -> placeholder module whose unified_base_image raises if called.
  4. NPJ_TORCH_ADAM=1: adds "torch_adam": true to the DeepSpeed optimiser config, so DeepSpeed runs torch's
     AdamW instead of its fused CUDA kernel (same update rule; for machines that cannot JIT-build FusedAdam).
     The config's "bias_correction": true is a FusedAdam-only argument that torch's AdamW rejects (DeepSpeed 0.19.7
     passes the params through); torch's AdamW always applies bias correction, so it is dropped (asserted True).
  5. Without --enable_deepspeed (DeepSpeed not installable): the script's final test loads the DeepSpeed-format
     best checkpoint (checkpoint-best/mp_rank_00_model_states.pt); the best model is also saved under that name.
     Precision then differs from the paper's run: torch AMP instead of DeepSpeed fp16.
  6. (DeepSpeed) today's DeepSpeed switches autocast off inside its engine; the official val/test passes rely on
     autocast (fp32 videos, fp16 weights) as the authors' 2023 DeepSpeed allowed -> the engine leaves autocast alone.
  7. (DeepSpeed) gradient scale: with update_freq > 1 the official script AND DeepSpeed both divide the loss by it;
     the authors ran update_freq 1 (2 GPUs x 32). DeepSpeed's own division is switched off (scale_wrt_gas=False), so
     the gradient is the mean over the 64 clips, as theirs. See patch_deepspeed().
  8. torch >= 2.6 loads checkpoints with weights_only=True by default; the official final test loads its own
     DeepSpeed best checkpoint (numpy scalars inside) as torch < 2.6 did -> default restored to weights_only=False.
Launch through torchrun (the final test wraps the model in DistributedDataParallel), from a folder that holds
pretrain_params/ (the script reads start weights from that relative path):
    cd <run dir> && torchrun --nproc_per_node=1 <this file> <official args...>
"""
import json
import os
import runpy
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
from surgvista_mini import load_cfg  # noqa: E402

REF = Path(load_cfg()["work_root"]) / "ref" / "SurgVISTA"
SCRIPT = REF / "downstream" / "downstream_triplet" / "run_triplet_training.py"


def setup():
    for p in (str(REF / "downstream"), str(REF)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    for mod in ("decord", "tensorboardX"):
        try:
            __import__(mod)
        except ImportError:
            stub = types.ModuleType(mod)
            if mod == "tensorboardX":
                class SummaryWriter:  # no-op: TensorBoard logging only
                    def __init__(self, *a, **k): pass
                    def __getattr__(self, name): return lambda *a, **k: None
                stub.SummaryWriter = SummaryWriter
            sys.modules[mod] = stub
            print(f"[launcher] {mod} not installed: stubbed (not used for results)", flush=True)
    try:
        import numpy.lib.function_base  # noqa: F401
    except ImportError:
        fb = types.ModuleType("numpy.lib.function_base")
        fb.disp = print
        sys.modules["numpy.lib.function_base"] = fb
        print("[launcher] numpy.lib.function_base missing (numpy 2): stubbed (`disp`, never called)", flush=True)
    import importlib
    import re
    src = (REF / "datasets" / "transforms" / "optim_factory.py").read_text(encoding="utf8")
    for mod, name in re.findall(r"^from (timm\.optim\.\w+) import (\w+)", src, re.M):
        try:
            m = importlib.import_module(mod)
        except ImportError:
            m = sys.modules[mod] = types.ModuleType(mod)
        if not hasattr(m, name):
            def _missing(*a, _n=name, **k):
                raise NotImplementedError(f"{_n}: not in this timm version (not used by the recipe)")
            setattr(m, name, _missing)
            print(f"[launcher] {mod}.{name} missing in timm {importlib.import_module('timm').__version__}: placeholder", flush=True)
    import timm.models

    def create_model(model_name, pretrained=False, **kwargs):  # timm 0.4.12 semantics (no checkpoint_path use)
        for k in ("checkpoint_path", "scriptable", "exportable", "no_jit"):
            kwargs.pop(k, None)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return timm.models.model_entrypoint(model_name)(pretrained=pretrained, **kwargs)
    timm.models.create_model = create_model
    if not (REF / "downstream" / "model" / "unifiedmodel_image.py").exists():
        im = types.ModuleType("downstream.model.unifiedmodel_image")

        def unified_base_image(*a, **k):
            raise NotImplementedError("unifiedmodel_image.py is not in the public SurgVISTA repo")
        im.unified_base_image = unified_base_image
        sys.modules["downstream.model.unifiedmodel_image"] = im
        print("[launcher] downstream.model.unifiedmodel_image not in the repo: placeholder (image models only)", flush=True)
    sys.path.append(str(REF / "downstream" / "downstream_triplet"))  # the package imports itself as `ivtmetrics`
    import ivtmetrics as ivt
    sys.modules["downstream.ivtmetrics"] = ivt
    import downstream.utils as U

    if os.environ.get("NPJ_TORCH_ADAM") == "1":
        orig_cfg = U.create_ds_config

        def create_ds_config(args):
            orig_cfg(args)
            cfg = json.load(open(args.deepspeed_config))
            prm = cfg["optimizer"]["params"]
            prm["torch_adam"] = True
            assert prm.pop("bias_correction", True) is True  # torch AdamW: always bias-corrected
            json.dump(cfg, open(args.deepspeed_config, "w"), indent=2)
            print("[launcher] DeepSpeed optimiser: torch AdamW (torch_adam=true)", flush=True)
        U.create_ds_config = create_ds_config

    # 8. torch >= 2.6 defaults torch.load to weights_only=True; the official script loads its own best checkpoint
    #    (DeepSpeed format, holds numpy scalars) with the pre-2.6 default. Restore it for files this run wrote/loads.
    import torch
    if "weights_only" in torch.load.__code__.co_varnames:
        orig_load = torch.load

        def load(*a, **k):
            k.setdefault("weights_only", False)
            return orig_load(*a, **k)
        torch.load = load

    if "--enable_deepspeed" in sys.argv:
        patch_deepspeed()

    if os.environ.get("NPJ_OFFICIAL_NODIST") == "1":  # smoke tests on Windows only (no NCCL): no torchrun
        import torch

        class _NoDDP(torch.nn.Module):
            def __init__(self, module, **k):
                super().__init__()
                self.module = module

            def forward(self, *a, **k):
                return self.module(*a, **k)
        torch.nn.parallel.DistributedDataParallel = _NoDDP
        torch.distributed.barrier = lambda *a, **k: None
        orig_init = U.init_distributed_mode

        def init_distributed_mode(args):  # torchrun would set args.gpu; the final test passes it to DDP
            orig_init(args)
            args.gpu = getattr(args, "gpu", 0)
        U.init_distributed_mode = init_distributed_mode
        print("[launcher] NPJ_OFFICIAL_NODIST=1: single process, DDP/barrier bypassed (smoke test only)", flush=True)

    orig_save = U.save_model

    def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
        orig_save(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema)
        if loss_scaler is not None and str(epoch) == "best" and U.is_main_process():
            d = Path(args.output_dir) / "checkpoint-best"
            d.mkdir(parents=True, exist_ok=True)
            import torch
            torch.save({"model": model_without_ddp.state_dict()}, d / "mp_rank_00_model_states.pt")
    U.save_model = save_model


def patch_deepspeed():
    """6./7. in the header: make today's DeepSpeed behave as the 2023 one the authors ran, for their 2 x 32 setup."""
    import contextlib
    import importlib
    import inspect
    import deepspeed  # noqa: F401
    from deepspeed.runtime.engine import DeepSpeedEngine

    # 6. autocast: the official val/test passes run model(videos) under torch.cuda.amp.autocast() with fp32 videos and
    #    rely on autocast to cast them to the fp16 weights. DeepSpeed >= 0.16 switches autocast off inside its engine
    #    unless its own "torch_autocast" config is used (we use its "fp16" section, as the official config) ->
    #    "Input type (float) and bias type (c10::Half)". Restore the old behaviour: the engine leaves autocast alone.
    #    Training is unaffected (the official train step feeds .half() samples without autocast).
    patched = []
    for name in ("deepspeed.runtime.engine", "deepspeed.runtime.torch_autocast"):
        try:
            m = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(m, "autocast_if_enabled"):
            m.autocast_if_enabled = lambda *a, **k: contextlib.nullcontext()
            patched.append(name)
    if patched:
        print(f"[launcher] DeepSpeed engine leaves the caller's autocast on (as DeepSpeed 0.5-0.9): {patched}", flush=True)
    else:  # fallback: cast val/test inputs to fp16 (pure fp16 eval instead of autocast)
        import downstream_triplet.engine_for_triplet as EFT

        class _Half:
            def __init__(self, loader): self.loader = loader
            def __len__(self): return len(self.loader)
            def __iter__(self): return ((b[0].half(), *b[1:]) for b in self.loader)

        for fn in ("validation_one_epoch", "final_triplet_test"):
            orig = getattr(EFT, fn)
            setattr(EFT, fn, lambda loader, *a, _o=orig, **k: _o(_Half(loader), *a, **k))
        print("[launcher] autocast_if_enabled not found in this DeepSpeed: val/test inputs cast to fp16 instead", flush=True)

    # 7. gradient scale: the official script divides the loss by update_freq AND DeepSpeed's backward divides it by
    #    gradient_accumulation_steps again. The authors ran update_freq 1 (2 GPUs x 32 = 64), so neither happened and
    #    the gradient was the mean over 64 clips. With our 8 x 8 split the double division makes it 8x smaller, which
    #    in fp16 underflows and lets Adam's eps damp updates. Keep only the script's division (= the authors' gradient).
    def inner(f):  # DeepSpeed wraps backward in an NVTX profiling wrapper (*args, **kwargs) without __wrapped__
        f = inspect.unwrap(f)
        for c in f.__closure__ or ():
            v = c.cell_contents
            if callable(v) and getattr(v, "__name__", "") == "backward":
                return v
        return f
    if "scale_wrt_gas" in inspect.signature(inner(DeepSpeedEngine.backward)).parameters:
        orig_backward = DeepSpeedEngine.backward

        def backward(self, loss, *a, **k):
            k["scale_wrt_gas"] = False
            return orig_backward(self, loss, *a, **k)
        DeepSpeedEngine.backward = backward
        print("[launcher] DeepSpeed backward: scale_wrt_gas=False (gradient = mean over the total batch of 64, as the "
              "official 2 x 32 run)", flush=True)
    else:
        raise SystemExit("[launcher] DeepSpeedEngine.backward has no scale_wrt_gas in this DeepSpeed version: the loss "
                         "would be divided by the update frequency twice (8x too small gradients). Not running.")


def main():
    setup()
    sys.argv = [str(SCRIPT)] + sys.argv[1:]
    runpy.run_path(str(SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
