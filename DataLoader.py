import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor


class NFTDataset(Dataset):
    def __init__(self, nft_media, pic, labels):
        ...

    def __len__(self):
        return ...

    def __getitem__(self, idx):
        ...
