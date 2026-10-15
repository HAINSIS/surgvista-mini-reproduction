# SurgVISTA reproduction on CholecT50 triplet recognition

A reproduction of **SurgVISTA**, from *Large-scale self-supervised video foundation model for intelligent surgery*
(Yang et al., npj Digital Medicine 2026; [arXiv 2506.02692](https://arxiv.org/abs/2506.02692); official code
[isyangshu/SurgVISTA](https://github.com/isyangshu/SurgVISTA)), evaluated on CholecT50 surgical action-triplet
recognition with the official split, metric and fine-tuning script.

![demo](docs/demo_frame.png)

*Second-by-second recognition on held-out test video 111: each prediction uses only the past 60 seconds
(16 frames, 4 seconds apart). Top 3 triplets with probabilities; ground truth in green.*

## Results

Test AP_IVT on the official CholecT50 challenge split (test videos 92, 96, 103, 110, 111; 7,815 test frames),
official ivtmetrics (`compute_video_AP`, null classes excluded). Both models use the same fine-tuning: the
authors' own script, unchanged.

| Model | AP_IVT (95% CI) | AP_I | AP_V | AP_T | AP_IV | AP_IT |
|---|---|---|---|---|---|---|
| MVD-K400 | 32.1 (30.7 to 33.4) | 77.8 | 49.9 | 42.7 | 31.5 | 36.3 |
| MVD-K400 + SurgVISTA pretraining (30 epochs) | 30.2 (28.8 to 31.6) | 80.9 | 49.6 | 42.4 | 32.5 | 34.2 |
| *Paper: MVD-K400* | *35.3* | | | | | |
| *Paper: SurgVISTA (3.55 M frames)* | *41.1 (38.0 to 44.1)* | | | | | |

Paired comparison on the same test frames (`04_finetune/output/compare_off_mvd_amp_vs_off_svc_mvd_long_amp.md`):

| Metric | Difference (SurgVISTA 30 ep minus MVD) | 95% CI |
|---|---|---|
| AP_IVT | -1.86 | -3.00 to -0.66 |
| AP_I | +3.03 | +0.66 to +4.43 |
| AP_V | -0.30 | -2.25 to +0.85 |
| AP_T | -0.30 | -1.79 to +1.36 |
| AP_IV | +0.93 | -0.40 to +1.92 |
| AP_IT | -2.06 | -3.20 to -0.85 |

Confidence intervals: Bayesian block bootstrap over test frames (200 resamples, an Exp(1) weight per one-minute
block within each test video, the official metric recomputed, Efron bias-corrected percentile). The comparison
gives both runs the same weights in every resample. The per-frame predictions of both models are in
`04_finetune/output/ft_off_*_test_preds.npz`, so every number can be recomputed.

## What is reproduced

| | Paper | Here |
|---|---|---|
| Model | ViT-B, 16 frames × 224 px, joint space-time attention | same; outputs identical to the official classes (max difference 0.0, `03_pretrain/check_official.py`) |
| Pretraining objective | tube mask 0.85; pixel decoder (MSE) + EndoSSL ViT-L distillation decoder (SmoothL1, weight 0.05) | same, official `pretrain.sh` values |
| Pretraining data | 3.55 M frames, about 200 epochs, from scratch, 8×H800 for 10 days | 40 CholecT50 training videos (80k clips) × 30 epochs, warm-started from MVD-K400, one RTX 4090 D |
| Fine-tuning | official recipe | the authors' script, unchanged (`04_finetune/official_ft_launch.py`) |
| Evaluation | CholecT50 challenge split, ivtmetrics | same; sample lists checked against the official label builder: 0 differences |

The pretraining starts from MVD-K400, the paper's strongest public baseline, which has the same architecture
(`unified_base_st` in the official code).

## How to rerun

Needs Linux with one 24 GB NVIDIA GPU (fine-tuning about 15.7 h per model, pretraining about 13.3 h), the official
code cloned to `work/ref/SurgVISTA` at commit `00ae4bc641371f5b6e1efb899450bfb61f304100`, CholecT50 (we used the
Hugging Face copy `Voxel51/cholect50`), and the EndoSSL ViT-L teacher and MVD-K400 weights (links and checksums in
`02_data/data_sources.md`). Heavy files live in `work/` (or set `NPJ_WORK_ROOT`).

```bash
bash _shared/setup_linux.sh                        # Python env (uv), packages, checks
python 02_data/download_full.py --workers 128      # CholecT50 frames (resumable)
python 02_data/build_full.py --workers 12          # index, official class order, challenge split, labels
python 03_pretrain/check_official.py               # prints EQUIVALENT TO OFFICIAL

# SurgVISTA pretraining from MVD-K400, every training frame, 30 epochs
NPJ_SDPA=1 python 03_pretrain/pretrain_official.py --name svc_mvd_long --init work/weights/mvd_base_k400_400e_from_vitl.pth \
    --model base --batch-size 16 --update-freq 4 --clip-stride 1 --epochs 30 --warmup-epochs 3 --save-every 10 --workers 12

# both fine-tunes with the authors' script (official mixed-precision path), results collected automatically
USE_DS=0 bash _shared/run_official_ft.sh

# paired comparison + second-by-second demo on test video 111
bash _shared/deliver.sh
```

How the official fine-tuning script runs unchanged: `04_finetune/official_ft_prepare.py` lays the data out in the
official format and runs the official label builder; `04_finetune/official_ft_launch.py` lists every compatibility
shim for current libraries (none touches model, data, optimiser or metric); `04_finetune/official_ft_collect.py`
re-scores the official predictions and writes results only if they equal the script's own numbers.

## Repository layout

```
_shared/       model (surgvista.py), fine-tuning encoder (encoders.py), official metric + CIs (triplet_metrics.py),
               config, setup and run scripts
02_data/       CholecT50 download and index (official split and class order)
03_pretrain/   equivalence check against the official code, SurgVISTA pretraining
04_finetune/   official-script fine-tuning (prepare, launch, collect), our own fine-tune, comparison; results in output/
05_demo/       second-by-second demo; per-second predictions and timelines in output/
logs/          full logs of the pretraining and both official fine-tunes
docs/          demo screenshot
```

The first, small-scale version of this project (ViT-S, a subset of frames) is in the history at commit `eafa313`.
