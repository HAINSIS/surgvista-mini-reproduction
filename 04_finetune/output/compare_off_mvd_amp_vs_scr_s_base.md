# scr_s_base vs off_mvd_amp (paired, same 7,815 test frames)

Paired Bayesian block bootstrap (200 replicates, one-minute blocks, same weights for both runs), 95% bias-corrected percentile CI of the difference. Test-frame uncertainty only: no seed or video variance.

| Metric | off_mvd_amp | scr_s_base | Difference (b - a) | 95% CI | Replicates with b > a |
|---|---|---|---|---|---|
| AP_IVT | 32.08 | 30.18 | **-1.90** | -2.83 to -1.18 | 0% |
| AP_I | 77.84 | 76.39 | **-1.45** | -3.85 to +0.18 | 16% |
| AP_V | 49.93 | 49.23 | **-0.70** | -2.38 to +0.78 | 20% |
| AP_T | 42.74 | 43.26 | **+0.52** | -0.93 to +1.69 | 85% |
| AP_IV | 31.54 | 30.70 | **-0.84** | -1.79 to -0.08 | 4% |
| AP_IT | 36.30 | 34.49 | **-1.81** | -2.61 to -0.97 | 0% |

Triplet classes better in scr_s_base: 15 of 39.

| Largest gains | test frames | a | b | diff |
|---|---|---|---|---|
| bipolar,coagulate,peritoneum | 12 | 57.4 | 89.3 | +31.9 |
| grasper,grasp,gut | 24 | 3.3 | 13.4 | +10.1 |
| hook,cut,peritoneum | 92 | 8.9 | 18.7 | +9.8 |
| irrigator,aspirate,fluid | 179 | 33.5 | 36.3 | +2.8 |
| clipper,clip,blood_vessel | 14 | 3.5 | 6.2 | +2.7 |
| hook,dissect,peritoneum | 54 | 7.6 | 10.2 | +2.6 |
| grasper,grasp,gallbladder | 2754 | 41.9 | 43.9 | +2.0 |
| bipolar,dissect,gallbladder | 5 | 81.3 | 82.8 | +1.4 |

| Largest losses | test frames | a | b | diff |
|---|---|---|---|---|
| scissors,cut,cystic_duct | 78 | 64.1 | 36.8 | -27.2 |
| bipolar,dissect,cystic_duct | 37 | 50.5 | 31.0 | -19.5 |
| scissors,cut,cystic_artery | 44 | 40.0 | 23.7 | -16.3 |
| grasper,retract,liver | 1085 | 69.3 | 58.4 | -10.9 |
| grasper,pack,gallbladder | 96 | 45.7 | 35.4 | -10.3 |
| clipper,clip,cystic_duct | 161 | 74.2 | 64.5 | -9.7 |
| clipper,clip,cystic_artery | 96 | 61.8 | 53.6 | -8.3 |
| bipolar,coagulate,abdominal_wall_cavity | 60 | 62.0 | 55.0 | -6.9 |
