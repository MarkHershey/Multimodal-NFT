import json
import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

video_feat_h5 = "data/video_feats.h5"

assert Path(video_feat_h5).is_file()
with h5py.File(video_feat_h5, "r") as f:
    video_idx_to_id = f["ids"][()]

    video_id_to_h5_idx = {str(vid): i for i, vid in enumerate(video_idx_to_id)}

    sample_id = list(video_id_to_h5_idx.keys())[0]
    video_idx = video_id_to_h5_idx[sample_id]

    feat = f["video_features"][video_idx]
    feat = torch.from_numpy(feat)
    feat = feat.unsqueeze(0)
    print(feat.shape)


image_feat_h5 = "data/image_feats.h5"

assert Path(image_feat_h5).is_file()
with h5py.File(image_feat_h5, "r") as f:
    image_id_to_h5_idx = f["ids"][()]

    image_id_to_h5_idx = {str(vid): i for i, vid in enumerate(image_id_to_h5_idx)}

    sample_id = list(image_id_to_h5_idx.keys())[0]
    idx = image_id_to_h5_idx[sample_id]

    feat = f["image_features"][idx]
    feat = torch.from_numpy(feat)
    feat = feat.unsqueeze(0)
    print(feat.shape)
