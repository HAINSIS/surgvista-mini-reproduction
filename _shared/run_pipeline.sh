#!/usr/bin/env sh
# Runs stages 02 -> 05 end to end (bash / Git Bash). Each step's log lands in its stage output/ folder.
# Interpreter: python on PATH, or set PYTHON=/path/to/python.
# Epoch budgets live in config.yaml; minute caps here are only a safety net.
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
mkdir -p work/ckpt 05_demo/output

[ -f work/cache/frames112.npy ]      || $PY 02_data/build_index.py     > 02_data/output/build_index.log 2>&1
[ -f work/cache/teacher_feats.npy ]  || $PY 03_pretrain/extract_teacher.py > 03_pretrain/output/extract_teacher.log 2>&1
$PY 03_pretrain/pretrain.py --variant surgvista --minutes 20 > 03_pretrain/output/surgvista_pretrain.log 2>&1
for init in surgvista imagenet scratch; do
  $PY 04_finetune/finetune.py --init $init --minutes 12 > 04_finetune/output/${init}_finetune.log 2>&1
done
# extra arm: same SurgVISTA pretraining, started from ImageNet weights (domain-adaptive continued pretraining)
$PY 03_pretrain/pretrain.py --variant surgvista --init-from imagenet --minutes 20 > 03_pretrain/output/surgvista_in_pretrain.log 2>&1
$PY 04_finetune/finetune.py --init surgvista_in --minutes 12 > 04_finetune/output/surgvista_in_finetune.log 2>&1
$PY 04_finetune/linear_probe.py > 04_finetune/output/linear_probe.log 2>&1
$PY 04_finetune/summarize.py    > 04_finetune/output/summarize.log 2>&1
$PY 05_demo/demo.py --init imagenet --video 111  > 05_demo/output/demo_imagenet.log 2>&1
echo PIPELINE_DONE
