# Linear probe on frozen features — CholecT50, 10 held-out videos

| Encoder | mAP_I | mAP_V | mAP_T | mAP_IVT |
|---|---|---|---|---|
| Chance (test prevalence) | 27.4 | 20.1 | 10.7 | 5.3 |
| Random-init video ViT-S | 30.4 | 21.7 | 13.5 | 10.3 |
| ImageNet ViT-S (natural images) | 49.4 | 35.5 | 22.9 | 12.0 |
| SurgVISTA-mini from scratch | 28.8 | 21.3 | 17.0 | 7.2 |
| SurgVISTA-mini from ImageNet init | 32.8 | 25.9 | 15.0 | 8.4 |
| EndoSSL ViT-L (surgical image FM = teacher, 300 M params) | 55.3 | 46.4 | 40.4 | 26.1 |