import os

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

    def __init__(self, nft_media, pic, labels):
        ...

    def __len__(self):
        return ...

    def __getitem__(self, idx):
        ...


class NFTDataLoader(DataLoader):
    """
    DataLoader wraps an iterable around the Dataset to enable easy access to the samples.

    1. Configure data according to the kwargs.
    2. Construct the Dataset.
    3. Call super to initialize DataLoader.
    """

    def __init__(self, **kwargs):
        json_dir = str(kwargs.pop("json_dir"))
        video_features_path = str(kwargs.pop("video_features_path"))

        print(f"loading video feature from {video_features_path}")

        with h5py.File(video_features_path, "r") as f:
            video_idx_to_id = f["ids"][()]

        video_id_to_idx = {str(vid): i for i, vid in enumerate(video_idx_to_id)}

        self.dataset = NFTDataset()

        super().__init__(self.dataset, **kwargs)
