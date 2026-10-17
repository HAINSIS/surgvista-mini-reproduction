#!/usr/bin/env bash
# Regularisation screen against fine-tuning overfitting, with the OFFICIAL fine-tuning script.
# VALIDATION ONLY: the screen root's test pickle is a copy of val (04_finetune/screen_prepare.py), so the script's
# automatic final test scores validation again and no test video is ever loaded.
#   tmux new -s screen   then:   bash _shared/run_screen.sh          (log: $WORK/logs/screen.log)
#   later arms:   ARMS="s_combo:--lr 2.5e-4 --drop_path 0.2" bash _shared/run_screen.sh
#   more data per epoch:   THIN=2 ARMS="s_stride2:" bash _shared/run_screen.sh      (~4.8 h)
#   the single test of an arm declared final:   TEST_ARM=s_base bash _shared/run_screen.sh   (~5 min)
# Every arm: MVD-K400 start, 10 epochs (2 warm-up), every 4th training clip, seed 0 (official default), checkpoint
# every epoch -> averages of the last 3 and last 5 epochs scored with the official --eval. Each arm adds its own
# deviations from the official recipe (lr 5e-4, layer decay 0.75, drop path 0.1). ~2.3 h per arm on the RTX 4090.
# Results: 04_finetune/output/screen_<arm>.json + screen_results.md. Re-run safe: finished steps are skipped.
cd "$(dirname "$0")/.." || exit 1
set -o pipefail
export PYTHONUNBUFFERED=1
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
WORK=${NPJ_WORK_ROOT:-work}; [ -d "$WORK" ] || [ ! -d ../npj-dm ] || WORK=$(cd ../npj-dm && pwd)
if [ -x "$WORK/.venv/bin/python" ]; then PY=${PY:-$WORK/.venv/bin/python}; else PY=${PY:-$WORK/.venv/Scripts/python.exe}; fi
WORKERS=${WORKERS:-12}; FT_BS=${FT_BS:-4}; FT_UF=${FT_UF:-16}   # official AMP path (DeepSpeed fp16 did not train here)
THIN=${THIN:-4}; EPOCHS=${EPOCHS:-10}; WARMUP=${WARMUP:-2}; LASTS=${LASTS:-"3 5"}   # LASTS: epochs averaged
ARMS=${ARMS:-"s_base:;s_lr25:--lr 2.5e-4;s_ld65:--layer_decay 0.65;s_dp2:--drop_path 0.2"}  # "name:deviations;..."
START=${START:-$WORK/weights/mvd_base_k400_400e_from_vitl.pth}
OFF=${OFF:-$WORK/official_ft}; SRC=${SRC:-$OFF/CholecT50}
if [ "$THIN" = 4 ]; then SDATA=${SDATA:-$OFF/screen/CholecT50}; else SDATA=${SDATA:-$OFF/screen_thin$THIN/CholecT50}; fi  # one root per thinning
RUN=$OFF/run; SOUT=$OFF/screen_out
LAUNCH=$(pwd)/04_finetune/official_ft_launch.py
RES=${NPJ_SCREEN_OUT:-04_finetune/output}; export NPJ_SCREEN_OUT=$RES   # smoke tests: point elsewhere
LOG=$WORK/logs/screen.log; mkdir -p "$WORK/logs" "$RUN/pretrain_params" "$SOUT"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
ln -sf "$START" "$RUN/pretrain_params/mvd_base_k400_400e.pth"

official() {  # official <out dir> <data dir> [extra args]   (as run_official_ft.sh; extra args override)
  local out=$1 data=$2; shift 2   # local: screen() keeps its own $out (the arm folder)
  rm -rf "$out"; mkdir -p "$out"
  if [ "${NPJ_OFFICIAL_NODIST:-0}" = 1 ]; then RUNNER=("$PY" "$LAUNCH")   # Windows smoke test: no torchrun/NCCL
  else RUNNER=("$PY" -m torch.distributed.run --nproc_per_node=1 --master_port 12332 "$LAUNCH"); fi
  ( cd "$RUN" && "${RUNNER[@]}" \
      --batch_size $FT_BS --update_freq $FT_UF --epochs 20 --save_ckpt_freq 10 --model unified_base_st \
      --pretrained_data k400 --pretrained_method mvd --lr 5e-4 --layer_decay 0.75 --warmup_epochs 5 \
      --data_path "$data" --eval_data_path "$data" --nb_classes 100 --data_strategy online \
      --output_mode key_frame --num_frames 16 --sampling_rate 4 --data_set CholecT50 --data_fps 1fps \
      --output_dir "$out" --log_dir "$out" --num_workers $WORKERS --dist_eval --no_auto_resume "$@" ) \
      2>&1 | tee -a "$LOG" "$out/console.log"
}
done_run() { grep -qs "Final mAP-ivt" "$1"/*/log.txt; }

# TEST_ARM=<arm>: the single TEST evaluation of a finished arm, once its recipe is declared final. Scores the arm's
# best-val checkpoint (the rule used for every headline number) with the official --eval on the REAL split, collects
# it as ft_scr_<arm>_* and compares it paired with the official-recipe MVD run. Refuses to test an arm twice.
if [ -n "${TEST_ARM:-}" ]; then
  name=scr_$TEST_ARM; tout=$OFF/screen_test/$TEST_ARM
  [ -e "04_finetune/output/ft_${name}_metrics.json" ] && { say "REFUSED: $TEST_ARM already has a test result (ft_${name}_metrics.json); the test set is scored once per final setup"; exit 1; }
  [ -e "$RES/screen_$TEST_ARM.json" ] || { say "FAILED: $TEST_ARM has no screen summary ($RES/screen_$TEST_ARM.json)"; exit 1; }
  ck=("$SOUT/$TEST_ARM"/unified*/checkpoint-best/mp_rank_00_model_states.pt)
  [ ${#ck[@]} = 1 ] && [ -e "${ck[0]}" ] || { say "FAILED: no unique best checkpoint for $TEST_ARM (${ck[*]})"; exit 1; }
  if done_run "$tout"; then say "found finished test evaluation: $TEST_ARM (collecting only)"
  else
    say "start: TEST of $TEST_ARM (best-val checkpoint ${ck[0]}), official --eval on $SRC"
    official "$tout" "$SRC" --eval --resume "${ck[0]}" && done_run "$tout" || { say "FAILED: test of $TEST_ARM"; exit 1; }
  fi
  "$PY" 04_finetune/official_ft_collect.py --name "$name" --base "$tout" --screen-json "$RES/screen_$TEST_ARM.json" 2>&1 | tee -a "$LOG"     || { say "FAILED: collect $name"; exit 1; }
  "$PY" 04_finetune/compare_runs.py --a off_mvd_amp --b "$name" 2>&1 | tee -a "$LOG" || say "comparison failed (needs ft_off_mvd_amp_test_preds.npz)"
  say "TEST DONE: $TEST_ARM -> 04_finetune/output/ft_${name}_metrics.json, compare_off_mvd_amp_vs_${name}.md"
  exit 0
fi

# 1. screen root: every THIN-th training clip, val unchanged, test pickle = val
if [ ! -e "$SDATA/screen_meta.json" ]; then
  [ -e "$SRC/labels_pkl/test/1fpstest.pickle" ] || { say "FAILED: $SRC not prepared (run _shared/run_official_ft.sh step 1)"; exit 1; }
  "$PY" 04_finetune/screen_prepare.py --src "$SRC" --dst "$SDATA" --thin "$THIN" 2>&1 | tee -a "$LOG" \
    || { say "FAILED: screen root"; exit 1; }
else say "skip (done): screen root $SDATA"; fi

screen() {  # screen <arm> [deviations...]
  local name=$1; shift; local out=$SOUT/$name ev k
  if [ -e "$RES/screen_${name}.json" ]; then say "skip (done): $name"; return 0; fi
  if done_run "$out"; then say "found finished training: $name"
  else
    local nth="every ${THIN}th"; [ "$THIN" = 2 ] && nth="every 2nd"; [ "$THIN" = 1 ] && nth="every"
    say "start: $name ($EPOCHS epochs, warm-up $WARMUP, $nth clip; deviations: ${*:-none})"
    official "$out" "$SDATA" --epochs "$EPOCHS" --warmup_epochs "$WARMUP" --save_ckpt_freq 1 "$@" \
      && done_run "$out" || { say "FAILED: $name (see $out/console.log)"; exit 1; }
  fi
  # shellcheck disable=SC2086
  "$PY" 04_finetune/screen_collect.py average --out "$out" --last $LASTS 2>&1 | tee -a "$LOG" || { say "FAILED: average $name"; exit 1; }
  for k in $LASTS; do
    ev=$out/eval_avg_last$k
    [ -e "$out/avg_last$k.pth" ] || continue   # fewer epochs than k
    done_run "$ev" && continue
    say "start: $name average of the last $k epochs, official --eval on val"
    official "$ev" "$SDATA" --eval --resume "$out/avg_last$k.pth" "$@" && done_run "$ev" \
      || { say "FAILED: eval avg_last$k $name (see $ev/console.log)"; exit 1; }
  done
  "$PY" 04_finetune/screen_collect.py summarize --out "$out" --name "$name" --args "$*" --data "$SDATA" 2>&1 | tee -a "$LOG" \
    || { say "FAILED: summarize $name"; exit 1; }
  "$PY" 04_finetune/screen_collect.py clean --out "$out" --name "$name" 2>&1 | tee -a "$LOG"
  say "done: $name"
}

# 2. smoke run of the whole screen loop on the tiny smoke subset (videos 78/5/111, every 5th sample; made by
#    run_official_ft.sh): 2 epochs -> per-epoch checkpoints -> average -> official --eval -> summary -> clean.
#    ~5 min; results go to $OFF/screen_smoke/results, never into 04_finetune/output. The long arms start only after it.
SMK=$OFF/screen_smoke
if [ ! -e "$SMK/.passed" ]; then
  [ -e "$OFF/smoke/CholecT50/labels_pkl/test/1fpstest.pickle" ] || { say "FAILED: no smoke subset ($OFF/smoke; run_official_ft.sh makes it)"; exit 1; }
  say "start: smoke run of the screen loop (tiny subset, 2 epochs)"
  rm -rf "$SMK"
  "$PY" 04_finetune/screen_prepare.py --src "$OFF/smoke/CholecT50" --dst "$SMK/CholecT50" --thin 1 2>&1 | tee -a "$LOG"     || { say "FAILED: smoke screen root"; exit 1; }
  ( SDATA=$SMK/CholecT50 SOUT=$SMK/out RES=$SMK/results EPOCHS=2 WARMUP=0 LASTS=2; export NPJ_SCREEN_OUT=$RES
    mkdir -p "$RES"; screen smoke --lr 2.5e-4 ) || { say "FAILED: smoke run of the screen loop"; exit 1; }
  grep -q '"avg_last2"' "$SMK/results/screen_smoke.json" || { say "FAILED: smoke summary has no averaged model"; exit 1; }
  grep -q 'lr: 0.000250\|"lr": 0.00025\|lr=0.00025' "$SMK"/out/smoke/console.log || say "WARNING: could not confirm --lr reached the script"
  touch "$SMK/.passed"; say "done: smoke run passed (numbers meaningless: ~130 training clips)"
else say "skip (done): smoke run of the screen loop"; fi

IFS=';' read -ra ARM_LIST <<< "$ARMS"
for arm in "${ARM_LIST[@]}"; do
  name=${arm%%:*}; dev=${arm#*:}
  # shellcheck disable=SC2086
  screen "$name" $dev
done
say "SCREEN COMPLETE: $RES/screen_results.md"
