# Mixed-precision overflow record (official fine-tunes, torch GradScaler)

Each halving of the loss scale = one optimizer step skipped because of an overflow (inf/NaN gradients); the weights are not changed on that step. 1,140 optimizer steps per epoch.

## MVD-K400 fine-tune (off_mvd_amp)

Skipped optimizer steps: **3 of 22,800** (0.01%). Loss scale 65,536 → 16,777,216. Non-finite loss values logged: 0 (the official script stops on a non-finite loss).

| Epoch | Skipped steps (halvings) | Scale doublings | Loss scale min to max | Epoch-average grad_norm (training log) |
|---|---|---|---|---|
| 1 | 0 | 0 | 65,536 to 65,536 | 0.255 |
| 2 | 0 | 1 | 65,536 to 131,072 | 0.073 |
| 3 | 0 | 0 | 131,072 to 131,072 | 0.090 |
| 4 | 0 | 1 | 131,072 to 262,144 | 0.092 |
| 5 | 0 | 0 | 262,144 to 262,144 | 0.089 |
| 6 | 0 | 1 | 262,144 to 524,288 | 0.086 |
| 7 | 0 | 0 | 524,288 to 524,288 | 0.082 |
| 8 | 0 | 1 | 524,288 to 1,048,576 | 0.079 |
| 9 | 0 | 1 | 1,048,576 to 2,097,152 | 0.075 |
| 10 | 0 | 0 | 2,097,152 to 2,097,152 | 0.075 |
| 11 | 0 | 1 | 2,097,152 to 4,194,304 | 0.072 |
| 12 | 0 | 0 | 4,194,304 to 4,194,304 | 0.071 |
| 13 | 0 | 1 | 4,194,304 to 8,388,608 | 0.070 |
| 14 | 0 | 0 | 8,388,608 to 8,388,608 | 0.068 |
| 15 | 1 | 1 | 8,388,608 to 16,777,216 | inf |
| 16 | 0 | 0 | 8,388,608 to 8,388,608 | 0.066 |
| 17 | 1 | 1 | 8,388,608 to 16,777,216 | inf |
| 18 | 1 | 1 | 8,388,608 to 16,777,216 | inf |
| 19 | 0 | 0 | 8,388,608 to 8,388,608 | 0.064 |
| 20 | 0 | 1 | 8,388,608 to 16,777,216 | 0.064 |

## SurgVISTA 30-epoch fine-tune (off_svc_mvd_long_amp)

Skipped optimizer steps: **4 of 22,800** (0.02%). Loss scale 65,536 → 4,194,304. Non-finite loss values logged: 0 (the official script stops on a non-finite loss).

| Epoch | Skipped steps (halvings) | Scale doublings | Loss scale min to max | Epoch-average grad_norm (training log) |
|---|---|---|---|---|
| 1 | 0 | 0 | 65,536 to 65,536 | 0.265 |
| 2 | 0 | 1 | 65,536 to 131,072 | 0.075 |
| 3 | 0 | 0 | 131,072 to 131,072 | 0.095 |
| 4 | 0 | 1 | 131,072 to 262,144 | 0.100 |
| 5 | 0 | 0 | 262,144 to 262,144 | 0.101 |
| 6 | 0 | 1 | 262,144 to 524,288 | 0.096 |
| 7 | 0 | 0 | 524,288 to 524,288 | 0.092 |
| 8 | 0 | 1 | 524,288 to 1,048,576 | 0.089 |
| 9 | 0 | 1 | 1,048,576 to 2,097,152 | 0.087 |
| 10 | 0 | 0 | 2,097,152 to 2,097,152 | 0.084 |
| 11 | 0 | 1 | 2,097,152 to 4,194,304 | 0.083 |
| 12 | 0 | 0 | 4,194,304 to 4,194,304 | 0.081 |
| 13 | 0 | 1 | 4,194,304 to 8,388,608 | 0.081 |
| 14 | 0 | 0 | 8,388,608 to 8,388,608 | 0.079 |
| 15 | 1 | 1 | 8,388,608 to 16,777,216 | inf |
| 16 | 1 | 1 | 8,388,608 to 16,777,216 | inf |
| 17 | 0 | 0 | 8,388,608 to 8,388,608 | 0.077 |
| 18 | 1 | 1 | 8,388,608 to 16,777,216 | inf |
| 19 | 1 | 0 | 4,194,304 to 8,388,608 | inf |
| 20 | 0 | 0 | 4,194,304 to 4,194,304 | 0.073 |

