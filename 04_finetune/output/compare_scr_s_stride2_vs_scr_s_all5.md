# scr_s_all5 vs scr_s_stride2 (paired, same 7,815 test frames)

Paired Bayesian block bootstrap (200 replicates, one-minute blocks, same weights for both runs), 95% bias-corrected percentile CI of the difference. Test-frame uncertainty only: no seed or video variance.

| Metric | scr_s_stride2 | scr_s_all5 | Difference (b - a) | 95% CI | Replicates with b > a |
|---|---|---|---|---|---|
| AP_IVT | 34.82 | 34.92 | **+0.09** | -1.05 to +1.31 | 50% |
| AP_I | 85.45 | 85.47 | **+0.02** | -1.27 to +1.19 | 59% |
| AP_V | 53.69 | 53.96 | **+0.27** | -0.82 to +1.33 | 61% |
| AP_T | 45.08 | 44.57 | **-0.52** | -1.57 to +0.43 | 26% |
| AP_IV | 35.79 | 34.75 | **-1.03** | -2.32 to +0.09 | 2% |
| AP_IT | 40.05 | 37.07 | **-2.98** | -3.93 to -1.85 | 0% |

Triplet classes better in scr_s_all5: 19 of 39.

| Largest gains | test frames | a | b | diff |
|---|---|---|---|---|
| irrigator,irrigate,liver | 3 | 4.1 | 50.8 | +46.6 |
| grasper,grasp,omentum | 6 | 1.0 | 24.5 | +23.6 |
| bipolar,dissect,cystic_duct | 37 | 26.0 | 46.1 | +20.1 |
| bipolar,coagulate,blood_vessel | 74 | 55.5 | 65.4 | +9.9 |
| grasper,grasp,cystic_duct | 174 | 39.7 | 49.5 | +9.9 |
| clipper,clip,cystic_artery | 96 | 70.7 | 77.8 | +7.1 |
| grasper,pack,gallbladder | 96 | 38.5 | 45.3 | +6.8 |
| hook,dissect,cystic_artery | 239 | 22.0 | 28.5 | +6.5 |

| Largest losses | test frames | a | b | diff |
|---|---|---|---|---|
| bipolar,coagulate,peritoneum | 12 | 80.2 | 30.9 | -49.2 |
| bipolar,dissect,gallbladder | 5 | 100.0 | 77.3 | -22.7 |
| grasper,retract,gut | 13 | 43.5 | 24.7 | -18.8 |
| bipolar,grasp,specimen_bag | 3 | 19.2 | 3.9 | -15.3 |
| bipolar,dissect,cystic_artery | 82 | 31.6 | 19.5 | -12.1 |
| scissors,cut,cystic_artery | 44 | 61.2 | 49.6 | -11.6 |
| grasper,grasp,peritoneum | 323 | 23.7 | 16.7 | -7.0 |
| clipper,clip,cystic_duct | 161 | 78.9 | 76.3 | -2.5 |
