# Test AP on the official frame set (labelled frames only)

Official label builder skips frames with triplet id -1; our fine-tunes kept them. Same predictions, two frame sets. Paper MVD AP_IVT 35.3, SurgVISTA 41.1.

| run | frames | AP_IVT all | AP_IVT labelled only | 95% CI | I | V | T | IV | IT |
|---|---|---|---|---|---|---|---|---|---|
| mvd | 7815/10374 | 30.3 | **31.3** | 29.9-32.9 | 82.2 | 50.6 | 42.5 | 32.0 | 34.5 |
| mvd_png | 7815/10374 | 28.6 | **29.5** | 27.4-31.2 | 83.5 | 52.1 | 42.0 | 32.2 | 32.8 |
| svc_mvd_ep4 | 7815/10374 | 29.8 | **31.3** | 29.7-32.9 | 80.6 | 51.0 | 40.8 | 33.0 | 36.2 |

Training still included the empty frames in these runs; runs from 2026-10-13 on drop them in training, validation and test, like the official code (`--keep-empty` restores the old behaviour).
