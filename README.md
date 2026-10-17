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
official ivtmetrics (`compute_video_AP`, null classes excluded). Every model is fine-tuned with the authors' own
script, unchanged; the 5-epoch rows change only the schedule (see "Fine-tuning schedule" below).

| Model | Fine-tuning | AP_IVT (95% CI) | AP_I | AP_V | AP_T | AP_IV | AP_IT |
|---|---|---|---|---|---|---|---|
| MVD-K400 | 5 epochs, every clip | **34.9 (33.4 to 36.3)** | 85.5 | 54.0 | 44.6 | 34.8 | 37.1 |
| MVD-K400 | official, 20 epochs | 32.1 (31.1 to 33.6) | 77.8 | 49.9 | 42.7 | 31.5 | 36.3 |
| MVD-K400 + SurgVISTA pretraining (30 epochs) | official, 20 epochs | 30.2 (29.3 to 31.7) | 80.9 | 49.6 | 42.4 | 32.5 | 34.2 |
| *Paper: MVD-K400* | *official* | *35.3* | *85.2* | *53.1* | *46.7* | *36.0* | *38.1* |
| *Paper: SurgVISTA (3.55 M frames)* | *official* | *41.1 (38.0 to 44.1)* | | | | | |

Paired comparison on the same test frames (`04_finetune/output/compare_off_mvd_amp_vs_off_svc_mvd_long_amp.md`):

| Metric | Difference (SurgVISTA 30 ep minus MVD) | 95% CI |
|---|---|---|
| AP_IVT | -1.86 | -2.86 to -0.83 |
| AP_I | +3.03 | +1.02 to +5.07 |
| AP_V | -0.30 | -1.84 to +0.82 |
| AP_T | -0.30 | -1.66 to +0.92 |
| AP_IV | +0.93 | -0.09 to +1.67 |
| AP_IT | -2.06 | -3.06 to -1.07 |

Confidence intervals: Bayesian block bootstrap over test frames (200 resamples, an Exp(1) weight per one-minute
block of real time, i.e. (video, frame number ÷ 60), the official metric recomputed, Efron bias-corrected percentile). The comparison
gives both runs the same weights in every resample. The per-frame predictions of both models are in
`04_finetune/output/ft_*_test_preds.npz`, so every number can be recomputed.

### Fine-tuning schedule

With the official 20 epochs, validation AP_IVT peaks at epoch 3 while the training loss keeps falling and the
validation loss rises: neighbouring 1-second clips share most of their frames, so each epoch repeats every scene.
I compared schedules and regularisers on the 5 validation videos only (`_shared/run_screen.sh`; the screen's test
list is a copy of validation, so no test video is loaded), then scored each chosen recipe once on test:

| Recipe (official script, MVD-K400 start) | Passes over the training clips | Best val AP_IVT | Test AP_IVT |
|---|---|---|---|
| Official, 20 epochs | 20 | 43.2 | 32.1 |
| 10 epochs, every 4th clip | 2.5 | 43.9 | 30.2 |
| Same, learning rate 2.5e-4 / layer decay 0.65 / drop path 0.2 | 2.5 | 36.7 / 39.9 / 39.8 | |
| 10 epochs, every 2nd clip, seeds 0 / 1 / 2 | 5 | 44.30 / 44.28 / 44.26 | 34.8 / 32.7 / 31.4 |
| **5 epochs (1 warm-up), every clip** | **5** | **46.1** | **34.9** |

Seeds move validation by 0.04 and test by up to 3.4 points (5 test videos). Paired with the official 20-epoch run,
the 5-epoch model gains +2.84 AP_IVT (+1.53 to +4.08). Details: `04_finetune/output/screen_results.md`,
`04_finetune/output/compare_off_mvd_amp_vs_scr_*.md`.

## What is reproduced

| | Paper | Here |
|---|---|---|
| Model | ViT-B, 16 frames × 224 px, joint space-time attention | same; outputs identical to the official classes (max difference 0.0, `03_pretrain/check_official.py`) |
| Pretraining objective | tube mask 0.85; pixel decoder (MSE) + EndoSSL ViT-L distillation decoder (SmoothL1, weight 0.05) | same, official `pretrain.sh` values |
| Pretraining data | 3.55 M frames, about 200 epochs, from scratch, 8×H800 for 10 days | 40 CholecT50 training videos (80k clips) × 30 epochs, warm-started from MVD-K400, one RTX 4090 D |
| Fine-tuning | official recipe | the authors' script, unchanged (`04_finetune/official_ft_launch.py`); official 20 epochs and 5 epochs |
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

# 5-epoch schedule (validation only, about 4 h), then its single test evaluation and paired comparison
THIN=1 EPOCHS=5 WARMUP=1 LASTS="2 3" ARMS="s_all5:" bash _shared/run_screen.sh
TEST_ARM=s_all5 bash _shared/run_screen.sh

# paired comparison + second-by-second demo on test video 111
bash _shared/deliver.sh

# live streaming on any video or camera, with per-frame latency (frames read one at a time, past 60 s only)
python 05_demo/stream_demo.py --ckpt hainsis_triplet_vitb_5ep.pt --mp4 your_video.mp4 --show

# mixed-precision overflow record of both fine-tunes, from the training log
python 04_finetune/overflow_report.py
```

How the official fine-tuning script runs unchanged: `04_finetune/official_ft_prepare.py` lays the data out in the
official format and runs the official label builder; `04_finetune/official_ft_launch.py` lists every compatibility
shim for current libraries (none touches model, data, optimiser or metric); `04_finetune/official_ft_collect.py`
re-scores the official predictions and writes results only if they equal the script's own numbers.

## Final weights, real-time running and training health

| Weights (345,276,750 bytes each, fp32) | SHA-256 |
|---|---|
| `hainsis_triplet_vitb_5ep.pt` (5-epoch fine-tune, test 34.9) | `e13540aea70edcc1a370ad3d58ffd1572af5cef104e117b256bdc06337c69eac` |
| `mvd_k400_cholect50_finetuned.pt` | `91338838e0dbc1f8e4886b50210157fe8b43ef6d6d97f830a5870226e28591fc` |
| `surgvista30_cholect50_finetuned.pt` | `8be65a9f024a7117102a343e344cfb7d94179b59df9c405aaee332c7c4b9413b` |

The weight files are too large for GitHub and are shared separately. [MODEL_CARD.md](MODEL_CARD.md) describes the
input processing, the modules and the failure cases seen on the test video.

- **Streaming** (`05_demo/stream_demo.py`): a producer thread delivers frames on schedule (cameras: newest frame
  only, 1-frame driver buffer); each frame is timed from arrival to the overlay on screen, with queue wait and
  backlog. Real-time replay of test video 111 at 1 frame per second with the live window, GTX 1070 laptop GPU
  (batch 1, fp32), 300 s: end-to-end mean 147 ms (5-epoch model; 95th percentile 156 ms), 153 ms (MVD, official 20 epochs) and 191 ms (SurgVISTA). An overload test at 8 frames per
  second shows the backlog growing. The streamed predictions equal the offline demo's. Records:
  `05_demo/output/stream_*`. Real camera capture to display is not yet validated.
- **Mixed precision** (`04_finetune/output/overflow_report.md`): the scaler skipped 3 of 22,800 optimizer steps (MVD)
  and 4 of 22,800 (SurgVISTA), all in epochs 15 to 19; no non-finite loss; the evaluated checkpoints (epochs 4 and 7)
  predate every skipped step.

## Repository layout

```
_shared/       model (surgvista.py), fine-tuning encoder (encoders.py), official metric + CIs (triplet_metrics.py),
               config, setup and run scripts
02_data/       CholecT50 download and index (official split and class order)
03_pretrain/   equivalence check against the official code, SurgVISTA pretraining
04_finetune/   official-script fine-tuning (prepare, launch, collect), our own fine-tune, comparison; results in output/
05_demo/       second-by-second demo, streaming demo with latency; per-second predictions and timelines in output/
logs/          full logs of the pretraining and both official fine-tunes
docs/          demo screenshot
```

The first, small-scale version of this project (ViT-S, a subset of frames) is in the history at commit `eafa313`.
