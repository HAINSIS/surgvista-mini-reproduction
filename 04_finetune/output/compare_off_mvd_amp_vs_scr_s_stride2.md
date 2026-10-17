# scr_s_stride2 vs off_mvd_amp (paired, same 7,815 test frames)

Paired Bayesian block bootstrap (200 replicates, one-minute blocks, same weights for both runs), 95% bias-corrected percentile CI of the difference. Test-frame uncertainty only: no seed or video variance.

| Metric | off_mvd_amp | scr_s_stride2 | Difference (b - a) | 95% CI | Replicates with b > a |
|---|---|---|---|---|---|
| AP_IVT | 32.08 | 34.82 | **+2.75** | +1.42 to +3.53 | 100% |
| AP_I | 77.84 | 85.45 | **+7.60** | +5.93 to +9.55 | 100% |
| AP_V | 49.93 | 53.69 | **+3.77** | +1.91 to +4.95 | 100% |
| AP_T | 42.74 | 45.08 | **+2.34** | +0.83 to +3.61 | 100% |
| AP_IV | 31.54 | 35.79 | **+4.25** | +2.81 to +5.42 | 100% |
| AP_IT | 36.30 | 40.05 | **+3.76** | +2.51 to +4.83 | 100% |

Triplet classes better in scr_s_stride2: 21 of 39.

| Largest gains | test frames | a | b | diff |
|---|---|---|---|---|
| grasper,retract,gut | 13 | 18.1 | 43.5 | +25.4 |
| bipolar,coagulate,peritoneum | 12 | 57.4 | 80.2 | +22.8 |
| scissors,cut,cystic_artery | 44 | 40.0 | 61.2 | +21.2 |
| bipolar,dissect,gallbladder | 5 | 81.3 | 100.0 | +18.7 |
| grasper,grasp,cystic_duct | 174 | 22.9 | 39.7 | +16.7 |
| bipolar,grasp,specimen_bag | 3 | 4.6 | 19.2 | +14.6 |
| scissors,cut,cystic_duct | 78 | 64.1 | 77.0 | +13.0 |
| irrigator,aspirate,fluid | 179 | 33.5 | 43.1 | +9.6 |

| Largest losses | test frames | a | b | diff |
|---|---|---|---|---|
| bipolar,dissect,cystic_duct | 37 | 50.5 | 26.0 | -24.5 |
| grasper,retract,liver | 1085 | 69.3 | 59.4 | -9.9 |
| bipolar,coagulate,blood_vessel | 74 | 65.0 | 55.5 | -9.5 |
| hook,dissect,cystic_artery | 239 | 31.4 | 22.0 | -9.4 |
| grasper,pack,gallbladder | 96 | 45.7 | 38.5 | -7.2 |
| grasper,retract,gallbladder | 1186 | 37.7 | 33.0 | -4.7 |
| hook,dissect,gallbladder | 2147 | 72.8 | 69.1 | -3.8 |
| hook,dissect,peritoneum | 54 | 7.6 | 3.8 | -3.8 |
