#!/usr/bin/env bash
# Delivery after the official fine-tunes (). Run on the machine that ran them
# (Linux 4090), from anywhere, once `off_svc_mvd_long_amp` is done:   bash _shared/deliver.sh
#   1. paired comparison of the two official-code runs on the same test frames (04_finetune/compare_runs.py)
#   2. second-by-second demo of both fine-tuned models on held-out test video 111 (05_demo/demo_official.py)
#   3. results table + guide refresh (04_finetune/summarize_p2.py)
#   4. one zip with predictions, metrics, comparison, demo files and logs: ~/Desktop/npj_delivery_<date>.zip
# Visible + logged like the other scripts (log: $WORK/logs/deliver.log). DEMO_VIDEO=<id> picks another test video.
cd "$(dirname "$0")/.." || exit 1
set -o pipefail
export PYTHONUNBUFFERED=1
WORK=${NPJ_WORK_ROOT:-work}; [ -d "$WORK" ] || [ ! -d ../npj-dm ] || WORK=$(cd ../npj-dm && pwd)
if [ -x "$WORK/.venv/bin/python" ]; then PY=${PY:-$WORK/.venv/bin/python}; else PY=${PY:-$WORK/.venv/Scripts/python.exe}; fi
A=${A:-off_mvd_amp}; B=${B:-off_svc_mvd_long_amp}; DEMO_VIDEO=${DEMO_VIDEO:-111}
LOG=$WORK/logs/deliver.log; mkdir -p "$WORK/logs"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
step() { say "start: $*"; "$@" 2>&1 | tee -a "$LOG" || { say "FAILED: $*"; exit 1; }; say "done: $*"; }

for r in $A $B; do
  [ -e "04_finetune/output/ft_${r}_metrics.json" ] || { say "missing 04_finetune/output/ft_${r}_metrics.json (run not finished/collected)"; exit 1; }
done
step "$PY" 04_finetune/compare_runs.py --a "$A" --b "$B"
for r in $A $B; do
  ck=$(ls "$WORK"/official_ft/out_"$r"/*/checkpoint-best/mp_rank_00_model_states.pt 2>/dev/null | head -1)
  [ -n "$ck" ] || { say "no best checkpoint for $r under $WORK/official_ft/out_$r"; exit 1; }
  step "$PY" 05_demo/demo_official.py --ckpt "$ck" --video "$DEMO_VIDEO" --name "$r"
done
"$PY" 04_finetune/summarize_p2.py 2>&1 | tee -a "$LOG" || say "results table refresh failed (non-fatal; see above)"

ZIP=~/Desktop/npj_delivery_$(date +%Y-%m-%d).zip
rm -f "$ZIP"
files=$(ls 04_finetune/output/ft_off_*_{metrics.json,log.csv,test_preds.npz} 04_finetune/output/compare_* \
           04_finetune/output/official_ft_data_check.json 04_finetune/output/rescore_official_frames.* \
           04_finetune/output/p2_results.* 05_demo/output/demo_off_* 2>/dev/null)
cp "$WORK/logs/official_ft.log" "$WORK/logs/phase3.log" 04_finetune/output/ 2>/dev/null
"$PY" - "$ZIP" $files 04_finetune/output/official_ft.log 04_finetune/output/phase3.log <<'EOF'
import sys, zipfile, os
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for f in sys.argv[2:]:
        if os.path.exists(f):
            z.write(f, "npj_delivery/" + f)
print(f"wrote {sys.argv[1]} ({os.path.getsize(sys.argv[1]) / 1e6:.1f} MB, {len(sys.argv) - 2} files)")
EOF
say "DELIVERY READY: $ZIP"
