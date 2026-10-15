# off_svc_mvd_long_amp vs off_mvd_amp (paired, same 7,815 test frames)

Paired Bayesian block bootstrap (200 replicates, one-minute blocks, same weights for both runs), 95% bias-corrected percentile CI of the difference. Test-frame uncertainty only: no seed or video variance.

| Metric | off_mvd_amp | off_svc_mvd_long_amp | Difference (b - a) | 95% CI | Replicates with b > a |
|---|---|---|---|---|---|
| AP_IVT | 32.08 | 30.22 | **-1.86** | -2.86 to -0.83 | 1% |
| AP_I | 77.84 | 80.88 | **+3.03** | +1.02 to +5.07 | 100% |
| AP_V | 49.93 | 49.62 | **-0.30** | -1.84 to +0.82 | 44% |
| AP_T | 42.74 | 42.44 | **-0.30** | -1.66 to +0.92 | 38% |
| AP_IV | 31.54 | 32.47 | **+0.93** | -0.09 to +1.67 | 98% |
| AP_IT | 36.30 | 34.24 | **-2.06** | -3.06 to -1.07 | 0% |

Triplet classes better in off_svc_mvd_long_amp: 15 of 39.

| Largest gains | test frames | a | b | diff |
|---|---|---|---|---|
| scissors,cut,cystic_artery | 44 | 40.0 | 66.8 | +26.8 |
| bipolar,coagulate,abdominal_wall_cavity | 60 | 62.0 | 74.0 | +12.0 |
| clipper,clip,cystic_artery | 96 | 61.8 | 73.2 | +11.4 |
| clipper,clip,cystic_duct | 161 | 74.2 | 83.2 | +9.1 |
| bipolar,dissect,cystic_artery | 82 | 26.5 | 30.6 | +4.1 |
| hook,dissect,cystic_duct | 884 | 55.1 | 58.8 | +3.7 |
| grasper,retract,liver | 1085 | 69.3 | 71.5 | +2.2 |
| hook,cut,peritoneum | 92 | 8.9 | 9.8 | +0.9 |

| Largest losses | test frames | a | b | diff |
|---|---|---|---|---|
| bipolar,dissect,cystic_duct | 37 | 50.5 | 17.2 | -33.3 |
| bipolar,dissect,gallbladder | 5 | 81.3 | 64.5 | -16.8 |
| bipolar,coagulate,peritoneum | 12 | 57.4 | 46.5 | -10.9 |
| bipolar,coagulate,blood_vessel | 74 | 65.0 | 54.1 | -10.9 |
| grasper,retract,gut | 13 | 18.1 | 8.2 | -10.0 |
| hook,dissect,cystic_artery | 239 | 31.4 | 22.7 | -8.7 |
| grasper,grasp,cystic_duct | 174 | 22.9 | 15.6 | -7.4 |
| grasper,pack,gallbladder | 96 | 45.7 | 38.5 | -7.1 |
