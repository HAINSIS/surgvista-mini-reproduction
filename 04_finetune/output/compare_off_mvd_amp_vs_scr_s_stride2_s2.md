# scr_s_stride2_s2 vs off_mvd_amp (paired, same 7,815 test frames)

Paired Bayesian block bootstrap (200 replicates, one-minute blocks, same weights for both runs), 95% bias-corrected percentile CI of the difference. Test-frame uncertainty only: no seed or video variance.

| Metric | off_mvd_amp | scr_s_stride2_s2 | Difference (b - a) | 95% CI | Replicates with b > a |
|---|---|---|---|---|---|
| AP_IVT | 32.08 | 31.40 | **-0.68** | -1.74 to +0.14 | 10% |
| AP_I | 77.84 | 79.84 | **+2.00** | +0.21 to +3.70 | 100% |
| AP_V | 49.93 | 51.28 | **+1.35** | -0.05 to +2.80 | 94% |
| AP_T | 42.74 | 43.92 | **+1.18** | -0.26 to +2.59 | 96% |
| AP_IV | 31.54 | 32.06 | **+0.53** | -0.71 to +1.33 | 86% |
| AP_IT | 36.30 | 34.20 | **-2.09** | -3.08 to -1.16 | 0% |

Triplet classes better in scr_s_stride2_s2: 18 of 39.

| Largest gains | test frames | a | b | diff |
|---|---|---|---|---|
| irrigator,irrigate,liver | 3 | 4.8 | 25.7 | +20.9 |
| grasper,grasp,omentum | 6 | 1.9 | 18.8 | +16.9 |
| grasper,retract,gut | 13 | 18.1 | 32.5 | +14.3 |
| grasper,grasp,cystic_duct | 174 | 22.9 | 33.8 | +10.9 |
| clipper,clip,cystic_duct | 161 | 74.2 | 81.6 | +7.4 |
| bipolar,dissect,cystic_artery | 82 | 26.5 | 32.8 | +6.3 |
| bipolar,coagulate,abdominal_wall_cavity | 60 | 62.0 | 68.0 | +6.0 |
| grasper,grasp,gallbladder | 2754 | 41.9 | 45.8 | +3.9 |

| Largest losses | test frames | a | b | diff |
|---|---|---|---|---|
| bipolar,dissect,cystic_duct | 37 | 50.5 | 19.2 | -31.3 |
| clipper,clip,cystic_artery | 96 | 61.8 | 45.5 | -16.3 |
| bipolar,coagulate,peritoneum | 12 | 57.4 | 44.4 | -13.0 |
| hook,dissect,cystic_artery | 239 | 31.4 | 20.0 | -11.4 |
| scissors,cut,cystic_artery | 44 | 40.0 | 30.4 | -9.6 |
| hook,dissect,gallbladder | 2147 | 72.8 | 64.2 | -8.6 |
| bipolar,dissect,gallbladder | 5 | 81.3 | 74.5 | -6.8 |
| bipolar,coagulate,blood_vessel | 74 | 65.0 | 58.4 | -6.6 |
