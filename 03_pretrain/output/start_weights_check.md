# Start-weight conversion check (2026-10-09)

Functional test of `_shared/encoders.py` conversion of HF `MCG-NJU/videomae-base` (VideoMAE-K400, 1600 ep,
pretraining checkpoint incl. pixel decoder) into the SurgVISTA student. 24 random CholecT50 train clips, official
crop + tube mask 0.85, normalised-pixel MSE on masked cubes (the official video loss), ViT-B, eval mode.

| student weights | recon loss |
|---|---|
| random init | 1.3124 |
| VideoMAE-K400 converted (1-D pos table, as the weights were trained) | **0.4356** |
| same, q/k weights deliberately swapped | 0.5257 |
| same, but 3-D sin-cos pos (wrong for these weights) | 1.0065 |

Conclusions: the conversion is correct (a wrong mapping loses most of the benefit); VideoMAE-initialised runs
must keep the 1-D table (`pos="1d"`, official `unified_base`); a natural-video model already reconstructs surgical
video far better than random init, so starting from it saves compute.
Also: 1-D table equals the official `get_sinusoid_encoding_table` exactly (max abs diff 0.0).
