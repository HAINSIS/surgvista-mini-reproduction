"""Clip sampling shared by pretrain, finetune and demo.

A clip ending at frame i = the 16 preceding frames (stride = sampling_rate) of the same contiguous
1 fps run, clamped at the run start (first frame repeated), as is standard for online recognition.
All frames live on the GPU as uint8; batches are gathered by index, so no DataLoader is needed.
"""
import numpy as np
import pandas as pd
import torch


def clip_index(df: pd.DataFrame, num_frames=16, stride=1):
    """Returns N x num_frames int array of row indices into df (df sorted by video, frame)."""
    v, f = df.video.values, df.frame.values
    run_start = np.zeros(len(df), int)
    for i in range(1, len(df)):
        same_run = v[i] == v[i - 1] and f[i] - f[i - 1] == 1
        run_start[i] = run_start[i - 1] if same_run else i
    offs = np.arange(-(num_frames - 1) * stride, 1, stride)
    idx = np.arange(len(df))[:, None] + offs[None]
    return np.maximum(idx, run_start[:, None])


def gather(frames_u8: torch.Tensor, idx: torch.Tensor, flip: torch.Tensor | None = None):
    """frames_u8: N x H x W x 3 uint8 (GPU); idx: B x T -> B x 3 x T x H x W float in [0,1]."""
    x = frames_u8[idx].float().div_(255).permute(0, 4, 1, 2, 3)
    if flip is not None and flip.any():
        x[flip] = x[flip].flip(-1)
    return x
