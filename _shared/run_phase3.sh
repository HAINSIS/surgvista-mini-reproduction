#!/usr/bin/env bash
# Phase 3 on the RTX 4090. Visible + resumable.
#   1. calibration check: MVD fine-tune on the ORIGINAL 854x480 frames, as the official CholecT50 loader reads them
#      (pilot fine-tunes used the 512x320 pretraining cache and scored ~5 points under the paper's MVD)
#   2. long SurgVISTA pretraining from MVD: every frame (80k clips/epoch) for LONG_EPOCHS epochs (pilot: 20k x 4)
#   3. official fine-tune of the long-pretrained encoder, same settings as step 1
# Fused attention (SDPA) is on (NPJ_SDPA=1): identical math, ~2x faster and much lighter on fp16 GPUs.
# Run it VISIBLY:  tmux new -s phase3   then:  bash _shared/run_phase3.sh   (log: $WORK/logs/phase3.log)
cd "$(dirname "$0")/.." || exit 1
set -o pipefail
export PYTHONUNBUFFERED=1
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
WORK=${NPJ_WORK_ROOT:-work}; [ -d "$WORK" ] || [ ! -d ../npj-dm ] || WORK=$(cd ../npj-dm && pwd)
if [ -x "$WORK/.venv/bin/python" ]; then PY=${PY:-$WORK/.venv/bin/python}; else PY=${PY:-$WORK/.venv/Scripts/python.exe}; fi
export NPJ_SDPA=${NPJ_SDPA:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
AMP_FLAG=${AMP_FLAG-}; PT_BS=${PT_BS:-16}; PT_UF=${PT_UF:-4}; FT_BS=${FT_BS:-8}; FT_UF=${FT_UF:-8}; WORKERS=${WORKERS:-12}
FT_FRAMES=${FT_FRAMES:-png}; LONG_EPOCHS=${LONG_EPOCHS:-30}; LONG_WARMUP=${LONG_WARMUP:-3}
START=${START:-$WORK/weights/mvd_base_k400_400e_from_vitl.pth}
CK=$WORK/ckpt
LOG=$WORK/logs/phase3.log; mkdir -p $WORK/logs
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
refresh_guide() {
  case "$1" in
    *04_finetune/output/probe_*|*04_finetune/output/ft_*)
      if "$PY" 04_finetune/summarize_p2.py > "$WORK/logs/guide_refresh.out" 2>&1; then
        say "guide: results table + page rebuilt (06_present/output/reproduction_guide.html); republish from the laptop"
      else
        say "guide: refresh failed, see $WORK/logs/guide_refresh.out"
      fi ;;
  esac
}
run() {  # run <done-file> <cmd...>: skip if done-file exists, stop on failure
  done_file=$1; shift
  if [ -e "$done_file" ]; then say "skip (done): $*"; return 0; fi
  say "start: $*"
  "$@" 2>&1 | tee -a "$LOG" || { say "FAILED: $*"; exit 1; }
  say "done: $*"
  refresh_guide "$done_file"
}
say "phase 3: SDPA=$NPJ_SDPA frames=$FT_FRAMES FT batch ${FT_BS}x${FT_UF} PT batch ${PT_BS}x${PT_UF} long epochs $LONG_EPOCHS"
FTARGS="--batch-size $FT_BS --update-freq $FT_UF $AMP_FLAG --train-stride 1 --val-stride 1 --workers $WORKERS --frames $FT_FRAMES"

# 1. calibration check on the official frame source
run 04_finetune/output/ft_mvd_${FT_FRAMES}_metrics.json "$PY" 04_finetune/finetune_official.py --name mvd_${FT_FRAMES} --init "$START" $FTARGS

# 2. long SurgVISTA pretraining from MVD (every frame)
LAST=$((LONG_EPOCHS - 1))
run $CK/p2_svc_mvd_long/checkpoint-$LAST.pth "$PY" 03_pretrain/pretrain_official.py --name svc_mvd_long --init "$START" \
    --model base --batch-size $PT_BS --update-freq $PT_UF $AMP_FLAG --clip-stride 1 --epochs $LONG_EPOCHS \
    --warmup-epochs $LONG_WARMUP --save-every 10 --workers $WORKERS

# 3. fine-tune the long-pretrained encoder, same settings as step 1
run 04_finetune/output/ft_svc_mvd_long_${FT_FRAMES}_metrics.json "$PY" 04_finetune/finetune_official.py \
    --name svc_mvd_long_${FT_FRAMES} --init $CK/p2_svc_mvd_long/checkpoint-$LAST.pth $FTARGS
say "PHASE 3 COMPLETE"
