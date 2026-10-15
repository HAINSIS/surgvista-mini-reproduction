# Data sources for the full SurgVISTA reproduction

Compiled 2026-10-09 from the paper's Extended Data Table 33 (data availability), Tables 1–2 (pretraining
corpus), the official repo `isyangshu/SurgVISTA@00ae4bc` (clone in `work/ref/SurgVISTA`) and the CholecT50
download docs. Paper PDF and text: `work/ref/SurgVISTA_arXiv2506.02692.{pdf,txt}`.
Status: `have` / `todo` (we can download it ourselves) / `ask` (needs a request form, account, or credentials) /
`blocked` (private or not released).

## Paper's pretraining corpus and data-scale settings (Table 2)

| Setting | Adds | Cumulative frames |
|---|---|---|
| A | Cholec80 (40 train videos only) | 86.3k |
| B | + M2CAI16-Workflow (27 train videos) | 153.9k |
| C | + HeiChole (24), PitVis (25), PSI-AVA (8), AutoLaparo (10 train) | 441.6k |
| D | + BernBypass70 + StrasBypass70 (= SurgPub, 274 videos) | 1.20M |
| E | + SurgWeb (3,376 web videos) | 3.55M (the reported model) |

Pretraining compute for E: 8 × NVIDIA H800 for 10 days (paper, Methods). Test splits of downstream datasets
that overlap SurgPub are excluded from pretraining.

## 1. Fine-tuning target: CholecT50 (the one downstream task we reproduce first)

| Item | Link | Access | Status |
|---|---|---|---|
| CholecT50, full, 50 videos / 100,863 frames at 1 fps (FiftyOne export, 100 shards ≈ 58 GB) | https://huggingface.co/datasets/Voxel51/cholect50 | open | have 4/100 shards in `work/data/cholect50` |
| CholecT50 official release from CAMMA (labels + official split files; use it to verify the HF copy) | https://github.com/CAMMA-public/cholect50 → request form https://forms.gle/GbMj8TwNoNpMUJuv9 | ask | todo |
| Official triplet metric `ivtmetrics` (vendored in official repo `downstream/downstream_triplet/ivtmetrics/`) | https://github.com/CAMMA-public/ivtmetrics | open | have (in clone) |

Official split used by SurgVISTA (`generate_pkl_cholect50.py`, `dataset_split = "cholect50-challenge"`):
train 40 videos, **val 5, 18, 36, 65, 74**, **test 92, 96, 103, 110, 111**. The paper reports this as
"45 train / 5 test, 90,489 / 10,374 frames". All 10 val+test videos are in the HF copy.
Paper target (Table 12, ViT-B): SurgVISTA AP_IVT **41.1** (95% CI 38.0–44.1) vs MVD-K400 35.3, VideoMAE-K400 33.5,
ImageNet-MAE 35.3, ImageNet-supervised 29.8.

## 2. Pretraining data, public part (SurgPub)

| Dataset | Setting | Link (paper Table 33) | Access | Status |
|---|---|---|---|---|
| Cholec80 | A | https://camma.unistra.fr/datasets/ | ask (credentialed) | ask — our earlier link gave 403 |
| M2CAI16-Workflow | B | https://camma.unistra.fr/m2cai2016/index.php/program-challenge/ | ask (credentialed) | ask |
| HeiChole | C | https://www.synapse.org/Synapse:syn18824884/wiki/592580 | ask (Synapse account + challenge registration) | ask |
| PitVis | C | https://rdr.ucl.ac.uk/articles/dataset/PitVis_Challenge_Endoscopic_Pituitary_Surgery_videos/26531686 | open | todo |
| PSI-AVA | C | https://github.com/BCV-Uniandes/TAPIR | open | todo |
| AutoLaparo | C | https://autolaparo.github.io/ | ask (request form) | ask |
| BernBypass70 + StrasBypass70 (MultiBypass140, 761k frames) | D | https://github.com/CAMMA-public/MultiBypass140 | open | todo (large) |

Stand-in until Cholec80 arrives: the official repo also has `pretrain_preprosses/generate_CholecT50.py`. The 40
CholecT50 challenge-train videos (about 80k frames, already free on HF) are a Setting-A-sized surgical corpus with
no test leakage (test videos 92–111 come from Cholec120, not Cholec80). Say clearly that this is a stand-in.

## 3. Pretraining data, web part (SurgWeb) — setting E

| Item | Link | Status |
|---|---|---|
| SurgVISTA's own release of the full corpus | https://huggingface.co/datasets/syangcw/SurgVISTADATA | blocked — repo is empty (checked 2026-10-09) |
| GenSurgery (GSViT web videos, 680 h) | https://github.com/SamuelSchmidgall/GSViT | todo — YouTube lists, must be re-scraped |
| SurgeNet (named in the official README) | https://github.com/TimJaspers0801/SurgeNet | check |
| Surg-3M (named in the official README) | https://github.com/visurg-ai/surg-3m | check |

The paper does not list exactly which web videos make up the 3,376, so setting E cannot be rebuilt one-to-one.

## 4. Downstream benchmark (13 datasets, paper Table 3 / Table 33)

| Dataset | Task | Link | Access |
|---|---|---|---|
| CholecT50 | triplet | see section 1 | open (HF) / ask (CAMMA) |
| Cholec80 | phase | https://camma.unistra.fr/datasets/ | ask |
| M2CAI16-Workflow | phase | https://camma.unistra.fr/m2cai2016/index.php/program-challenge/ | ask |
| AutoLaparo | phase | https://autolaparo.github.io/ | ask |
| PmLR50 | phase | https://github.com/RascalGdd/PmNet | ask |
| CATARACTS | phase | https://ieee-dataport.org/open-access/cataracts | open (IEEE DataPort account) |
| Cataract-101 | phase | https://ftp.itec.aau.at/datasets/ovid/cat-101/ | open |
| Cataract-21 | phase | https://ftp.itec.aau.at/datasets/ovid/cat-21/index.html | license-based |
| SurgicalActions160 | action | https://ftp.itec.aau.at/datasets/SurgicalActions160/ | license-based |
| Cholec80-CVS | skill/CVS | https://github.com/ManuelRios18/CHOLEC80-CVS-PUBLIC | open labels, needs Cholec80 videos |
| Endoscapes-CVS | skill/CVS | https://github.com/CAMMA-public/Endoscapes | open |
| ESD57 | phase | — | blocked (private) |
| Prostate21 | triplet | — | blocked (9 of 21 videos private) |

## 5. Weights

| Weights | Link | Status |
|---|---|---|
| EndoSSL ViT-L teacher (authors' converted ckpt) | https://drive.google.com/file/d/1bazW3W5Cglzfeeh_pdJALLktfxpateJ6/view | have |
| EndoSSL originals (Hirsch et al., MICCAI 2023; BSD-3): TF SavedModels, MSN-pretrained on PRIVATE laparoscopy / colonoscopy video; ViT-S/B/L each. Repo cloned to `work/ref/endossl` (2026-10-09) | https://github.com/RoyHirsch/endossl (laparoscopy ViT-L: https://drive.google.com/drive/folders/11TdNyl4HGvpoi6Ro0zZ28L1qxY4IAJPb , ViT-B: https://drive.google.com/drive/folders/1zcLKhE7H50GIDeb53chLrE5SUBtBooAR) | not needed: SurgVISTA uses the authors' PyTorch conversion of the laparoscopy ViT-L (have) |
| SurgVISTA pretrained ViT-B | https://huggingface.co/syangcw/SurgVISTA | blocked — only a README (checked 2026-10-09) |
| VideoMAE ViT-B K400 1600e / SSv2 2400e | https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md (Drive folder https://drive.google.com/file/d/1tEhLyskjb755TJ65ptsrafUG2llSwQE1/view) | todo |
| MVD ViT-B K400 400e | https://github.com/ruiwang2021/mvd/blob/main/MODEL_ZOO.md (https://drive.google.com/file/d/1MFgpUuHTtFMiormgenmxypU9gqRDbbwb/view) | todo |
| UMT ViT-B K710 | https://huggingface.co/OpenGVLab/UMT/tree/main | todo |
| InternVideo VideoMAE ViT-B Hybrid | https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo1/Pretrain/VideoMAE | todo |
| ImageNet ViT-B: supervised / DINO / MAE | https://huggingface.co/timm/vit_base_patch16_224.augreg_in1k , .dino , .mae | todo |
