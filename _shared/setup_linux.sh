#!/usr/bin/env bash
# One-time setup of the SurgVISTA reproduction on a Linux + NVIDIA machine. Safe to re-run (each step checks).
# Expects the transfer bundle unpacked so that this project folder and the work folder sit side by side:
#   npj_bundle/npj-digital-medicine/   (code; run this script from here)
#   npj_bundle/npj-dm/                 (work root: weights, teacher, pretrained checkpoints)
#     bash _shared/setup_linux.sh                 # or: NPJ_WORK_ROOT=/big/disk/npj-dm bash _shared/setup_linux.sh
# Mirrors first (mainland China friendly): PyPI via Tsinghua, PyTorch via download.pytorch.org then Aliyun,
# Hugging Face via hf-mirror.com, code via GitHub.
set -u
cd "$(dirname "$0")/.." || exit 1
PROJ=$(pwd)
export NPJ_WORK_ROOT=${NPJ_WORK_ROOT:-$PROJ/work}   # heavy files: work/ inside the repo (git-ignored)
W=$NPJ_WORK_ROOT; mkdir -p "$W"/{weights,ckpt,ref,logs,data/cholect50,cache}
PIP_MIRROR=${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}
ok()   { echo -e "  \033[32mOK\033[0m   $*"; }
warn() { echo -e "  \033[33mWARN\033[0m $*"; }
die()  { echo -e "  \033[31mFAIL\033[0m $*"; exit 1; }
echo "project: $PROJ"; echo "work root: $W"

echo "[1/7] GPU"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: install the NVIDIA driver first"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
CU=cu126; [ "${CC:-0}" -ge 120 ] && CU=cu128          # Blackwell (RTX 50xx, sm_120) needs CUDA 12.8 wheels
ok "compute capability $CC -> PyTorch $CU wheels"
FREE_GB=$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge 130 ] && ok "free disk at work root: ${FREE_GB} GB" || warn "only ${FREE_GB} GB free at $W; need ~130 GB (58 GB frames + 50 GB cache + checkpoints)"

echo "[2/7] uv + Python environment"
if ! command -v uv >/dev/null; then
  python3 -m pip install --user -q uv -i "$PIP_MIRROR" || die "could not install uv"
  export PATH=$HOME/.local/bin:$PATH
fi
ok "uv $(uv --version | cut -d' ' -f2)"
# Reuse an existing CUDA PyTorch >= 2.1 (e.g. the lab machine's conda base has 2.6.0) instead of downloading ~3 GB:
# the venv then sees the base packages (--system-site-packages) and nothing in the base environment is modified.
REUSE=$(python3 -c "import torch,sys; v=tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); print('yes' if v>=(2,1) and torch.cuda.is_available() else 'no')" 2>/dev/null || echo no)
REUSE=${REUSE_TORCH:-$REUSE}
if [ ! -x "$W/.venv/bin/python" ]; then
  PYV=$(python3 -c 'import sys;print(sys.version_info[:2]>=(3,10))' 2>/dev/null)
  if [ "$REUSE" = "yes" ]; then uv venv --seed --system-site-packages --python "$(command -v python3)" "$W/.venv"
  elif [ "$PYV" = "True" ]; then uv venv --seed --python python3 "$W/.venv"; else uv venv --seed --python 3.11 "$W/.venv"; fi
fi
[ "$REUSE" = "yes" ] && ok "reusing the existing PyTorch from $(command -v python3) (no 3 GB download)"
PY=$W/.venv/bin/python; [ -x "$PY" ] || die "venv not created"; ok "$($PY --version) at $W/.venv"

echo "[3/7] PyTorch ($CU) + packages"
if ! $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  uv pip install --python "$PY" torch==2.9.1 torchvision==0.24.1 --index-url "https://download.pytorch.org/whl/$CU" \
    || uv pip install --python "$PY" torch==2.9.1 torchvision==0.24.1 --find-links "https://mirrors.aliyun.com/pytorch-wheels/$CU/" -i "$PIP_MIRROR" \
    || die "PyTorch install failed"
fi
# timm / torchvision with --no-deps so they never pull a second PyTorch; their small deps are listed explicitly
if ! $PY -c "import torchvision" 2>/dev/null; then
  TV=$($PY -c "import torch; v=torch.__version__.split('+')[0]; m={'2.1':'0.16','2.2':'0.17','2.3':'0.18','2.4':'0.19','2.5':'0.20','2.6':'0.21','2.7':'0.22','2.8':'0.23','2.9':'0.24'}; print(m['.'.join(v.split('.')[:2])])")
  uv pip install --python "$PY" -q --no-deps -i "$PIP_MIRROR" "torchvision==${TV}.*" || die "torchvision install failed"
fi
uv pip install --python "$PY" -q --no-deps -i "$PIP_MIRROR" timm==1.0.30 || die "timm install failed"
uv pip install --python "$PY" -q -i "$PIP_MIRROR" einops==0.8.2 "numpy>=2,<3" pandas scikit-learn opencv-python-headless pillow \
  matplotlib PyYAML tqdm safetensors urllib3 scipy huggingface_hub \
  || die "package install failed"
$PY -c "import torch; print('  torch', torch.__version__, '| CUDA ok:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0), '| archs', torch.cuda.get_arch_list()[-2:])"

echo "[4/7] official code (pinned commit) for the equivalence check and ivtmetrics"
# GitHub may be unreachable (it is on the lab machine): the bundle carries the clone in npj-dm/ref/SurgVISTA.
if [ ! -d "$W/ref/SurgVISTA" ]; then
  git clone -q https://github.com/isyangshu/SurgVISTA.git "$W/ref/SurgVISTA" 2>/dev/null     || die "no official code at $W/ref/SurgVISTA and GitHub unreachable: copy npj-dm/ref/SurgVISTA from the bundle"
fi
if [ -d "$W/ref/SurgVISTA/.git" ]; then git -C "$W/ref/SurgVISTA" checkout -q 00ae4bc641371f5b6e1efb899450bfb61f304100 2>/dev/null; fi
[ -f "$W/ref/SurgVISTA/pretrain_SurgVISTA/modeling_surgery_student.py" ] && ok "official SurgVISTA code present (commit 00ae4bc)" || die "official code incomplete"

echo "[5/7] weights"
[ -s "$W/ckpt/teacher_endossl_vitl.pth" ] && ok "teacher (EndoSSL ViT-L)" || warn "missing $W/ckpt/teacher_endossl_vitl.pth  <- Google Drive file id 1bazW3W5Cglzfeeh_pdJALLktfxpateJ6 (1.2 GB, 'vit_large_patch16_224_surgery.pth'), rename to teacher_endossl_vitl.pth"
[ -s "$W/weights/mvd_base_k400_400e_from_vitl.pth" ] && ok "MVD-K400 start weights" || warn "missing $W/weights/mvd_base_k400_400e_from_vitl.pth  <- Google Drive file id 1MFgpUuHTtFMiormgenmxypU9gqRDbbwb (343 MB, 'mvd_b_from_l_ckpt_399.pth'), rename"
for f in "MCG-NJU/videomae-base:model.safetensors:videomae_base_k400_1600e.safetensors" \
         "OpenGVLab/VideoMAEv2-Base:model.safetensors:videomaev2_base_hybrid1m_800e.safetensors"; do
  IFS=: read -r repo file out <<< "$f"
  [ -s "$W/weights/$out" ] || curl -sL --noproxy '*' --retry 3 -o "$W/weights/$out" "https://hf-mirror.com/$repo/resolve/main/$file"
  [ -s "$W/weights/$out" ] && ok "$out" || warn "could not download $out"
done
[ -s "$W/ckpt/p2_svc_mvd/checkpoint-3.pth" ] && ok "laptop pretraining checkpoints present (pilot will skip that step)" || warn "no laptop pretraining checkpoints: the pilot will redo the 4-epoch pretraining (fast on this GPU)"

echo "[6/7] dataset labels (frames are downloaded by the pilot, resumable)"
[ -s "$W/data/cholect50/samples.json" ] || curl -sL --noproxy '*' --retry 3 -o "$W/data/cholect50/samples.json" \
  https://hf-mirror.com/datasets/Voxel51/cholect50/resolve/main/samples.json
$PY -c "import json;print('  samples.json:', len(json.load(open('$W/data/cholect50/samples.json'))['samples']), 'frames listed (expect 100863)')" || die "samples.json unreadable"
rm -f 02_data/output/.cache_complete   # the laptop's marker; this machine builds its own cache

echo "[7/7] verification"
$PY 03_pretrain/check_official.py 2>/dev/null | tail -1 | grep -q "EQUIVALENT TO OFFICIAL" && ok "model is bit-exact with the official code" || warn "check_official.py did not pass: run it directly to see why"
$PY - <<'EOF'
import sys, time, torch
sys.path.insert(0, "_shared")
import surgvista as sv
m = sv.build_student("base", 224, 16, 2, decoder_depth_video=4, decoder_depth_image=2, target_feature_dim=1024).cuda()
opt = torch.optim.AdamW(m.parameters(), 1e-5)
for amp in (False, True):
    bs = 16 if amp else 8
    x = torch.randn(bs, 3, 16, 224, 224, device="cuda"); mask = torch.from_numpy(sv.TubeMaskingGenerator((8, 14, 14), 0.85)()).bool().cuda().expand(bs, -1).contiguous()
    tf = torch.randn(bs, 8 * 196, 1024, device="cuda"); sc = torch.amp.GradScaler("cuda", enabled=amp)
    try:
        for i in range(8):
            if i == 3: torch.cuda.synchronize(); t = time.time()
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = sv.surgvista_loss(m, x, mask, tf)["loss"]
            opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update()
        torch.cuda.synchronize(); dt = (time.time() - t) / 5
        print(f"  student fwd+bwd, {'fp16' if amp else 'fp32'} batch {bs}: {bs/dt:.1f} clips/s (teacher not included), peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    except torch.OutOfMemoryError:
        print(f"  {'fp16' if amp else 'fp32'} batch {bs}: out of memory")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
EOF
echo
echo "Setup done. Next steps (README.md, How to rerun): download and index CholecT50, check_official.py,"
echo "pretraining, then: tmux new -s official && cd $PROJ && export NPJ_WORK_ROOT=$W && USE_DS=0 bash _shared/run_official_ft.sh"
