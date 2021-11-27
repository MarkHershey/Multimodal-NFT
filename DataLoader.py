import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import h5py
import matplotlib.pyplot as plt

# import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor


class NFTDataset(Dataset):
    """Dataset stores the samples and their corresponding labels."""

    def __init__(
        self,
        text_encoded: List[list],
        text_lengths: List[int],
        labels: List[int],
        image_feat_h5: Union[str, Path],
        image_id_to_h5_idx: Dict[int, int],
        video_feat_h5: Union[str, Path],
        video_id_to_h5_idx: Dict[int, int],
    ):
        ...

    def __len__(self):
        return ...

    def __getitem__(self, idx):
        ...


class NFTDataLoader(DataLoader):
    """
    DataLoader wraps an iterable around the Dataset to enable easy access to the samples.

    1. Collect data according to the kwargs during initilization.
    2. Build Dataset.
    3. Pass Dataset to super to complete initialization of DataLoader.

    By doing this, we can use the same DataLoader to load different datasets.
    """

    def __init__(self, **kwargs):
        # 1. Collect data according to the kwargs during initilization.
        self.batch_size = kwargs["batch_size"]

        json_dir = str(kwargs.pop("json_dir"))
        video_features_path = str(kwargs.pop("video_features_path"))

        print(f"loading video feature from {video_features_path}")

        with h5py.File(video_features_path, "r") as f:
            video_idx_to_id = f["ids"][()]

        video_id_to_idx = {str(vid): i for i, vid in enumerate(video_idx_to_id)}

        # 2. Build Dataset.
        self.dataset = NFTDataset()
        # 3. Pass Dataset to super to complete initialization of DataLoader.
        super().__init__(self.dataset, **kwargs)
