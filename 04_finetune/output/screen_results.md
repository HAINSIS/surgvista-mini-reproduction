# Fine-tuning screen (validation only)

MVD-K400 start, official fine-tuning script, official hyperparameters except the args column (seed 0 unless `--seed` is given). Epochs (warm-up 2, or 1 for 5-epoch arms) and training clips per arm below. All numbers are on the 5 validation videos; the test videos are never loaded. Seed spread on validation was ±0.02 (s_stride2, 3 seeds).

| Arm | Training clips | Epochs | Args | Best epoch val AP_IVT (epoch) | Mean val, last 3 | Weight averages of the last k epochs | Best epoch: I / V / T / IT |
|---|---|---|---|---|---|---|---|
| s_all5 | every | 5 | `official` | 46.10 (4) | 45.49 | last 2: 46.05, last 3: 45.92 | 83.03 / 64.44 / 53.13 / 47.38 |
| s_base | every 4th | 10 | `official` | 43.89 (8) | 43.69 | last 3: 43.78, last 5: 42.89 | 79.62 / 65.02 / 54.94 / 45.99 |
| s_dp2 | every 4th | 10 | `--drop_path 0.2` | 39.80 (7) | 39.70 | last 3: 39.78, last 5: 39.16 | 75.13 / 61.11 / 52.46 / 43.22 |
| s_ld65 | every 4th | 10 | `--layer_decay 0.65` | 39.90 (7) | 39.71 | last 3: 39.71, last 5: 39.74 | 74.99 / 60.45 / 52.93 / 41.22 |
| s_lr25 | every 4th | 10 | `--lr 2.5e-4` | 36.72 (9) | 36.63 | last 3: 36.76, last 5: 36.13 | 72.91 / 61.48 / 50.00 / 38.12 |
| s_stride2 | every 2nd | 10 | `official` | 44.30 (6) | 43.57 | last 3: 43.44, last 5: 43.43 | 82.90 / 64.18 / 53.05 / 46.36 |
| s_stride2_s1 | every 2nd | 10 | `--seed 1` | 44.28 (6) | 43.46 | last 3: 43.49, last 5: 44.07 | 82.52 / 64.47 / 54.31 / 45.46 |
| s_stride2_s2 | every 2nd | 10 | `--seed 2` | 44.26 (7) | 43.62 | last 3: 43.57, last 5: 44.40 | 84.78 / 65.83 / 54.44 / 45.90 |
