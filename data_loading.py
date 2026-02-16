import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
from deep_learning.utils.data import InriaDataset, Augment
from pathlib import Path

import numpy as np

_STATS = np.load("dataset_root/crevasse_stats.npz")
_P2 = torch.from_numpy(_STATS["p2"].astype("float32")).view(-1, 1, 1)
_P98 = torch.from_numpy(_STATS["p98"].astype("float32")).view(-1, 1, 1)

def transform_fn(sample, eps=1e-7):
    data, mask = sample  # data: (C,H,W)
    data = data.to(torch.float)

    # ---- NEW: force RGB (3 bands) ----
    if data.shape[0] > 3:
        data = data[:3, :, :]   # keep first 3 channels only
    # ----------------------------------

    # Now _P2/_P98 must also be length-3:
    p2 = _P2[: data.shape[0]].to(data.device)
    p98 = _P98[: data.shape[0]].to(data.device)

    # percentile clip
    data = torch.clamp(data, min=p2, max=p98)

    # global min–max to [0, 1]
    denom = (p98 - p2) + eps
    data = (data - p2) / denom

    data = data.to(torch.float32)
    mask = mask.to(torch.float32)
    return data, mask


def get_dataset(dataset, names=['images', 'gt'], augment=False):
    ds_path = 'dataset_root' 
    dataset = InriaDataset(ds_path, names, transform=transform_fn)
    if augment:
        dataset = Augment(dataset)
    return dataset
