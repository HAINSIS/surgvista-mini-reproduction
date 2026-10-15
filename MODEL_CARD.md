# Model card: CholecT50 triplet recognition (ViT-B video transformer)

Two fine-tuned models, same architecture and same official fine-tuning recipe:

| File | Start weights | Test AP_IVT (95% CI) | SHA-256 |
|---|---|---|---|
| `mvd_k400_cholect50_finetuned.pt` | MVD-K400 | 32.1 (31.1 to 33.6) | `91338838e0dbc1f8e4886b50210157fe8b43ef6d6d97f830a5870226e28591fc` |
| `surgvista30_cholect50_finetuned.pt` | MVD-K400 + 30 epochs SurgVISTA pretraining | 30.2 (29.3 to 31.7) | `8be65a9f024a7117102a343e344cfb7d94179b59df9c405aaee332c7c4b9413b` |

Each file is 345,276,750 bytes: a PyTorch checkpoint `{"model": state_dict}`, fp32, 162 tensors, 86,304,100
parameters. Check after download: `sha256sum *.pt` (Linux) or `certutil -hashfile <file> SHA256` (Windows).

## Run it on a new video

```bash
# live: frames are read one at a time; each second is predicted from the past 60 s only; latency is measured
python 05_demo/stream_demo.py --ckpt surgvista30_cholect50_finetuned.pt --mp4 your_video.mp4 --show
# offline: per-second CSV, timeline figure and annotated video
python 05_demo/demo_official.py --ckpt surgvista30_cholect50_finetuned.pt --mp4 your_video.mp4 --name surgvista30
```

Needs the repository code and its requirements; a GPU is optional (CPU works, slower). End-to-end latency (frame
arrival to overlay on screen), batch 1, fp32, GTX 1070 laptop GPU: about 153 ms (MVD) and 191 ms (SurgVISTA) per
frame at one frame per second; see `05_demo/output/stream_*_summary.json`.

## Input processing (identical to the official CholecT50 test loader)

1. **Sampling:** 1 frame per second (the first decoded frame at or after each whole second), as CholecT50.
2. **Resize:** each frame to 854×480 (the CholecT50 frame size), then the whole frame to 224×224 (bilinear; the
   official loader squashes, it does not crop). RGB, divided by 255, ImageNet mean/std normalisation.
3. **Clip:** the prediction for second *t* uses 16 frames 4 s apart: *t*, *t*-4, …, *t*-60, i.e. the past
   60 seconds and nothing from the future. In the first minute the missing history is filled with the earliest
   frame. Tensor: 1 × 3 × 16 × 224 × 224.

## Key modules

| Module | What it does |
|---|---|
| Tubelet embedding | Conv3d with kernel and stride 2×16×16: the clip becomes 8×14×14 = 1,568 tokens of width 768 |
| Position embedding | fixed 3-D sin-cos table (time, height, width) added to the tokens |
| Encoder | 12 transformer blocks, 12 heads, joint space-time self-attention (every token attends to every token across all 16 frames), MLP width 3,072 |
| Head | mean of all tokens → LayerNorm → linear layer 768 → 100 → sigmoid: one independent probability per triplet ⟨instrument, verb, target⟩ |
| Component scores | instrument, verb and target scores are the maximum over the triplets that contain them (official ivtmetrics mapping); no separate heads |

## Failure causes seen on held-out test video 111

| Failure | Cause | Example |
|---|---|---|
| A triplet is predicted when no instrument is in view | The official protocol trains only on frames that have at least one triplet, so the model never learned "no instrument" | of 529 seconds with no instrument in view, a triplet with probability ≥ 0.5 is shown in 168 s (MVD) and 379 s (SurgVISTA) |
| Idle instrument read as active | "null verb" (instrument present, doing nothing) looks like the start or end of an action | idle hook read as hook-dissect-gallbladder, minutes 4 to 6 |
| Verb confusion on the same instrument and target | grasp and retract look alike over 60 s at 1 fps | 20:00: predicted grasper-retract-gallbladder; truth grasper-grasp-gallbladder plus grasper-retract-liver |
| Neighbouring targets confused | gallbladder and liver are adjacent and often both in view | same example as above |
| Same instrument shown idle and active at once | the 100 triplets are scored independently (as in the paper); two of the same instrument can be legitimately labelled, so nothing forbids it | 43 s (MVD) and 63 s (SurgVISTA) of 2,146 s |
| Rare triplets | few training examples per class | lowest per-class AP on classes with few test frames (`04_finetune/output/compare_*.md`) |

## What to expect on a new video

- **Procedure:** the 100 classes are laparoscopic cholecystectomy triplets (CholecT50). Other procedures fall
  outside the label set.
- **Frame rate, resolution:** any; frames are sampled at 1 fps and resized as above.
- **Overlays:** on-screen text, borders or picture-in-picture are not in the training data and can change the
  predictions.
- **Context:** the model sees one minute at a time; it does not use the whole surgery.
