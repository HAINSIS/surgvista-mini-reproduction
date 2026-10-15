#!/usr/bin/env bash
# OFFICIAL SurgVISTA fine-tuning code on our CholecT50 copy. Linux + NVIDIA only (the official
# script needs torch.distributed/NCCL; one GPU is fine). Visible + resumable like run_phase3.sh:
#   tmux new -s official   then:   bash _shared/run_official_ft.sh        (log: $WORK/logs/official_ft.log)
#   without DeepSpeed (official AMP path, runs named off_*_amp):   USE_DS=0 bash _shared/run_official_ft.sh
#   while phase 3 still pretrains:   WAIT_PHASE3=1 bash _shared/run_official_ft.sh   (takes over when it ends)
#   0. deps: DeepSpeed (the paper's fp16 engine) + tensorboardX, from the pip mirror; FusedAdam build check
#   1. data: 04_finetune/official_ft_prepare.py -> $WORK/official_ft/CholecT50 (official layout + official pickles,
#      checked sample-for-sample against ours: 04_finetune/output/official_ft_data_check.json)
#   2. MVD-K400 fine-tune with the official script (calibration: paper 35.3 AP_IVT)
#   3. fine-tune of our long SurgVISTA pretraining (phase 3 step 2, checkpoint-29) with the official script
# The official script runs unchanged via 04_finetune/official_ft_launch.py (shims listed there); arguments are
# downstream/scripts/triplet/train_triplet_video.sh except: 1 GPU, batch FT_BS x update_freq FT_UF = 64 total
# (official 2 GPUs x 32), paths. Results: 04_finetune/output/ft_off_<name>_{metrics.json,log.csv,test_preds.npz}.
cd "$(dirname "$0")/.." || exit 1
set -o pipefail
export PYTHONUNBUFFERED=1
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
WORK=${NPJ_WORK_ROOT:-work}; [ -d "$WORK" ] || [ ! -d ../npj-dm ] || WORK=$(cd ../npj-dm && pwd)
if [ -x "$WORK/.venv/bin/python" ]; then PY=${PY:-$WORK/.venv/bin/python}; else PY=${PY:-$WORK/.venv/Scripts/python.exe}; fi
PIP_MIRROR=${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}
WORKERS=${WORKERS:-12}; USE_DS=${USE_DS:-1}
# batch per step x accumulation = 64. DeepSpeed fp16 fits 8 on 24 GB; the AMP path keeps softmax in fp32 -> 4
if [ "$USE_DS" = 1 ]; then FT_BS=${FT_BS:-8}; FT_UF=${FT_UF:-8}; else FT_BS=${FT_BS:-4}; FT_UF=${FT_UF:-16}; fi
START=${START:-$WORK/weights/mvd_base_k400_400e_from_vitl.pth}
SVC=${SVC:-$WORK/ckpt/p2_svc_mvd_long/checkpoint-29.pth}
OFF=$WORK/official_ft; DATA=$OFF/CholecT50; RUN=$OFF/run
LAUNCH=$(pwd)/04_finetune/official_ft_launch.py
LOG=$WORK/logs/official_ft.log; mkdir -p "$WORK/logs" "$RUN/pretrain_params"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
refresh_guide() {
  if "$PY" 04_finetune/summarize_p2.py > "$WORK/logs/guide_refresh.out" 2>&1; then
    say "guide: results table + page rebuilt; republish from the laptop"
  else say "guide: refresh failed, see $WORK/logs/guide_refresh.out"; fi
}

# 0. dependencies
if [ "$USE_DS" = 1 ] && ! "$PY" -c "import deepspeed" 2>/dev/null; then
  say "installing deepspeed + tensorboardX from $PIP_MIRROR (needs the installed torch: --no-build-isolation)"
  # --no-deps: deepspeed requires torch, and uv does not count the torch this venv borrows from the system
  # (--system-site-packages), so it would download and install a second, newer torch (several GB of CUDA libs).
  # Its other requirements are listed here explicitly.
  uv pip install --python "$PY" -i "$PIP_MIRROR" tensorboardX hjson msgpack ninja psutil py-cpuinfo "pydantic>=2" \
    nvidia-ml-py tqdm packaging 2>&1 | tee -a "$LOG"
  uv pip install --python "$PY" -i "$PIP_MIRROR" --no-build-isolation --no-deps deepspeed==0.19.7 2>&1 | tee -a "$LOG" \
    || { say "deepspeed install FAILED -> running without it (torch AMP instead of the paper's DeepSpeed fp16)"; USE_DS=0; }
fi
if [ "$USE_DS" = 1 ]; then
  if "$PY" - <<'EOF' >> "$LOG" 2>&1
import torch
from deepspeed.ops.adam import FusedAdam
p = torch.nn.Parameter(torch.zeros(4, device="cuda")); o = FusedAdam([p]); p.grad = torch.ones_like(p); o.step()
EOF
  then say "DeepSpeed FusedAdam OK"
  else export NPJ_TORCH_ADAM=1; say "FusedAdam cannot build here -> DeepSpeed with torch AdamW (same update rule)"; fi
  DSFLAG=--enable_deepspeed
else DSFLAG=; fi
# run names: off_<x> = DeepSpeed fp16 (the paper's engine); off_<x>_amp = the official script's own non-DeepSpeed path
# (torch AMP + NativeScaler), for when DeepSpeed is unavailable or misbehaves (USE_DS=0)
SUF=; [ "$USE_DS" = 1 ] || SUF=_amp
say "official fine-tune: batch ${FT_BS}x${FT_UF}, deepspeed=$USE_DS, workers $WORKERS"

# 1. data in the official layout + official pickles
if [ ! -e "$DATA/labels_pkl/test/1fpstest.pickle" ]; then
  say "start: data layout (official_ft_prepare.py)"
  "$PY" 04_finetune/official_ft_prepare.py 2>&1 | tee -a "$LOG" || { say "FAILED: data layout"; exit 1; }
else say "skip (done): data layout"; fi

# WAIT_PHASE3=1: start this while phase 3 step 2 (long pretraining) still runs. Steps 0-1 run now (CPU); here it
# waits for the pretraining to finish, then stops phase 3 step 3 (our fine-tune code), which this script replaces.
if [ "${WAIT_PHASE3:-0}" = 1 ]; then
  say "waiting for phase 3 pretraining to finish ($SVC)"
  while [ ! -e "$SVC" ] || pgrep -f "pretrain_official.py --name svc_mvd_long" > /dev/null; do sleep 60; done
  say "pretraining finished; stopping phase 3 step 3 if it starts (up to 5 min)"
  for _ in $(seq 30); do
    if pkill -f "finetune_official.py --name svc_mvd_long"; then
      say "stopped phase 3 step 3 (our fine-tune code); the official code replaces it"; sleep 20; break
    fi
    sleep 10
  done
fi

# start weights under the names the official script expects (pretrain_params/ relative to the run folder)
ln -sf "$START" "$RUN/pretrain_params/mvd_base_k400_400e.pth"

official() {  # official <out dir> <data dir> <batch> <update_freq> <pretrained_data> <pretrained_method> [extra args]
  out=$1 data=$2 b=$3 u=$4 pd=$5 pm=$6; shift 6
  rm -rf "$out"; mkdir -p "$out"
  ( cd "$RUN" && "$PY" -m torch.distributed.run --nproc_per_node=1 --master_port 12332 "$LAUNCH" \
      --batch_size $b --update_freq $u --epochs 20 --save_ckpt_freq 10 --model unified_base_st \
      --pretrained_data "$pd" --pretrained_method "$pm" --lr 5e-4 --layer_decay 0.75 --warmup_epochs 5 \
      --data_path "$data" --eval_data_path "$data" --nb_classes 100 --data_strategy online \
      --output_mode key_frame --num_frames 16 --sampling_rate 4 --data_set CholecT50 --data_fps 1fps \
      --output_dir "$out" --log_dir "$out" --num_workers $WORKERS --dist_eval $DSFLAG --no_auto_resume "$@" ) \
      2>&1 | tee -a "$LOG" "$out/console.log"
}

# 1b. smoke run of the whole official loop on a tiny subset (3 videos, every 5th sample, 1 epoch): train -> val ->
#     best checkpoint -> test, with the real DeepSpeed setup. ~5 min; the long runs only start once it passes.
SMOKE=$OFF/smoke
if [ ! -e "$SMOKE/.passed_ds$USE_DS" ]; then
  say "start: smoke run of the official loop (tiny subset, 1 epoch)"
  rm -rf "$SMOKE"
  "$PY" 04_finetune/official_ft_prepare.py --videos 78 5 111 --root "$SMOKE/CholecT50" --thin 5 2>&1 | tee -a "$LOG" \
    || { say "FAILED: smoke data"; exit 1; }
  official "$SMOKE/out" "$SMOKE/CholecT50" $FT_BS $FT_UF k400 mvd --epochs 1 --warmup_epochs 0 \
    && grep -q "Final mAP-ivt" "$SMOKE"/out/*/log.txt \
    || { say "FAILED: smoke run (see $SMOKE/out/console.log)"; exit 1; }
  touch "$SMOKE/.passed_ds$USE_DS"; say "done: smoke run passed (numbers meaningless: ~130 training clips)"
else say "skip (done): smoke run"; fi

finetune() {  # finetune <name> <pretrained_data> <pretrained_method>
  name=$1; out=$OFF/out_$name
  if [ -e "04_finetune/output/ft_${name}_metrics.json" ]; then say "skip (done): $name"; return 0; fi
  bs=$FT_BS; uf=$FT_UF
  # a finished run (final test written) is only collected again, never retrained (training output is never deleted)
  if grep -qs "Final mAP-ivt" "$out"/*/log.txt; then say "found finished run: $name (collecting only)"
  else while :; do
    say "start: $name (official run_triplet_training.py, batch ${bs}x${uf})"
    official "$out" "$DATA" $bs $uf "$2" "$3" && break
    if grep -q "out of memory" "$out/console.log" && [ "$bs" -gt 1 ]; then
      bs=$((bs / 2)); uf=$((uf * 2)); say "OOM -> retry with batch ${bs}x${uf} (same total 64)"
    else say "FAILED: $name (see $out/console.log)"; exit 1; fi
  done; fi
  "$PY" 04_finetune/official_ft_collect.py --name "$name" --base "$out" 2>&1 | tee -a "$LOG" || { say "FAILED: collect $name"; exit 1; }
  say "done: $name"; refresh_guide
}

# 2. calibration: MVD-K400 with the official code
finetune off_mvd$SUF k400 mvd

# 3. our long SurgVISTA pretraining with the official code
if [ -e "$SVC" ]; then
  ln -sf "$SVC" "$RUN/pretrain_params/SurgVISTA.pth"
  finetune off_svc_mvd_long$SUF SurgVISTA SurgVISTA
else say "skip: $SVC not found (phase 3 step 2 not finished?)"; fi
say "OFFICIAL FINE-TUNES COMPLETE"
