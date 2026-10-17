# scr_s_all5 vs off_mvd_amp (paired, same 7,815 test frames)

Paired Bayesian block bootstrap (200 replicates, one-minute blocks, same weights for both runs), 95% bias-corrected percentile CI of the difference. Test-frame uncertainty only: no seed or video variance.

| Metric | off_mvd_amp | scr_s_all5 | Difference (b - a) | 95% CI | Replicates with b > a |
|---|---|---|---|---|---|
| AP_IVT | 32.08 | 34.92 | **+2.84** | +1.53 to +4.08 | 100% |
| AP_I | 77.84 | 85.47 | **+7.63** | +5.73 to +9.73 | 100% |
| AP_V | 49.93 | 53.96 | **+4.04** | +2.50 to +5.32 | 100% |
| AP_T | 42.74 | 44.57 | **+1.83** | +0.42 to +2.94 | 100% |
| AP_IV | 31.54 | 34.75 | **+3.22** | +2.01 to +4.35 | 100% |
| AP_IT | 36.30 | 37.07 | **+0.77** | -0.74 to +1.76 | 91% |

Triplet classes better in scr_s_all5: 19 of 39.

| Largest gains | test frames | a | b | diff |
|---|---|---|---|---|
| irrigator,irrigate,liver | 3 | 4.8 | 50.8 | +46.0 |
| grasper,grasp,cystic_duct | 174 | 22.9 | 49.5 | +26.6 |
| grasper,grasp,omentum | 6 | 1.9 | 24.5 | +22.6 |
| clipper,clip,cystic_artery | 96 | 61.8 | 77.8 | +16.0 |
| irrigator,aspirate,fluid | 179 | 33.5 | 46.7 | +13.2 |
| scissors,cut,cystic_duct | 78 | 64.1 | 76.8 | +12.7 |
| scissors,cut,cystic_artery | 44 | 40.0 | 49.6 | +9.6 |
| bipolar,coagulate,abdominal_wall_cavity | 60 | 62.0 | 69.7 | +7.8 |

| Largest losses | test frames | a | b | diff |
|---|---|---|---|---|
| bipolar,coagulate,peritoneum | 12 | 57.4 | 30.9 | -26.5 |
| bipolar,dissect,cystic_artery | 82 | 26.5 | 19.5 | -7.0 |
| grasper,retract,liver | 1085 | 69.3 | 62.8 | -6.5 |
| bipolar,dissect,cystic_duct | 37 | 50.5 | 46.1 | -4.3 |
| bipolar,dissect,gallbladder | 5 | 81.3 | 77.3 | -4.0 |
| hook,dissect,cystic_artery | 239 | 31.4 | 28.5 | -2.9 |
| grasper,retract,gallbladder | 1186 | 37.7 | 35.4 | -2.3 |
| grasper,grasp,specimen_bag | 440 | 87.6 | 85.8 | -1.8 |
