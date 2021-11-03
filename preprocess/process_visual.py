import argparse, os
import h5py
from scipy.misc import imresize
import skvideo.io
from PIL import Image

import torch
from torch import nn
import torchvision
import random
import numpy as np


def build_resnet():
    cnn = torchvision.models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(cnn.children())[:-1])
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def run_batch(cur_batch, model):
    """
    Args:
        cur_batch: treat a video as a batch of images
        model: ResNet model for feature extraction
    Returns:
        ResNet extracted feature.
    """
    # NOTE: Required normalization for all pre-trained model in pytorch
    # ref: https://pytorch.org/docs/stable/torchvision/models.html#classification
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_batch = (image_batch / 255.0 - mean) / std
    # NOTE:  torch.FloatTensor is CPU tensor, torch.cuda.FloatTensor is GPU tensor
    # cuda() returns a copy of this object in CUDA memory.
    # ref: https://pytorch.org/docs/stable/tensors.html?highlight=cuda#torch.Tensor.cuda
    image_batch = torch.FloatTensor(image_batch).cuda()

    # NOTE: torch.no_grad is a Context-manager that disabled gradient calculation
    # ref: https://pytorch.org/docs/stable/generated/torch.no_grad.html#torch.no_grad
    with torch.no_grad():
        image_batch = torch.autograd.Variable(image_batch)

    feats = model(image_batch)
    feats = feats.data.cpu().clone().numpy()

    return feats