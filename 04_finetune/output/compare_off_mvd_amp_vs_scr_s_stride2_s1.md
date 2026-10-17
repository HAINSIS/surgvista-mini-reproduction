# scr_s_stride2_s1 vs off_mvd_amp (paired, same 7,815 test frames)

Paired Bayesian block bootstrap (200 replicates, one-minute blocks, same weights for both runs), 95% bias-corrected percentile CI of the difference. Test-frame uncertainty only: no seed or video variance.

| Metric | off_mvd_amp | scr_s_stride2_s1 | Difference (b - a) | 95% CI | Replicates with b > a |
|---|---|---|---|---|---|
| AP_IVT | 32.08 | 32.73 | **+0.65** | -0.30 to +1.59 | 90% |
| AP_I | 77.84 | 79.73 | **+1.88** | +0.26 to +3.69 | 97% |
| AP_V | 49.93 | 51.42 | **+1.49** | +0.43 to +2.53 | 98% |
| AP_T | 42.74 | 44.24 | **+1.50** | +0.21 to +2.78 | 99% |
| AP_IV | 31.54 | 32.86 | **+1.32** | +0.60 to +2.23 | 100% |
| AP_IT | 36.30 | 37.24 | **+0.95** | -0.22 to +1.66 | 99% |

Triplet classes better in scr_s_stride2_s1: 21 of 39.

| Largest gains | test frames | a | b | diff |
|---|---|---|---|---|
| grasper,retract,gut | 13 | 18.1 | 34.7 | +16.6 |
| grasper,grasp,cystic_duct | 174 | 22.9 | 39.2 | +16.3 |
| grasper,grasp,peritoneum | 323 | 17.5 | 28.7 | +11.2 |
| clipper,clip,cystic_duct | 161 | 74.2 | 83.5 | +9.4 |
| bipolar,dissect,gallbladder | 5 | 81.3 | 90.0 | +8.7 |
| grasper,grasp,gut | 24 | 3.3 | 10.0 | +6.7 |
| bipolar,coagulate,abdominal_wall_cavity | 60 | 62.0 | 67.4 | +5.4 |
| scissors,cut,cystic_artery | 44 | 40.0 | 45.3 | +5.3 |

| Largest losses | test frames | a | b | diff |
|---|---|---|---|---|
| bipolar,dissect,cystic_duct | 37 | 50.5 | 27.1 | -23.4 |
| bipolar,dissect,cystic_artery | 82 | 26.5 | 15.9 | -10.5 |
| bipolar,coagulate,peritoneum | 12 | 57.4 | 49.3 | -8.1 |
| hook,dissect,cystic_artery | 239 | 31.4 | 24.8 | -6.7 |
| hook,dissect,cystic_duct | 884 | 55.1 | 48.8 | -6.2 |
| grasper,retract,gallbladder | 1186 | 37.7 | 32.4 | -5.3 |
| irrigator,aspirate,fluid | 179 | 33.5 | 28.3 | -5.2 |
| grasper,retract,liver | 1085 | 69.3 | 64.5 | -4.8 |
