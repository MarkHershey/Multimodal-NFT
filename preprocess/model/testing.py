import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import numpy as np
import torch
import torchvision
from PIL import Image
from puts import get_logger

for num_layers in [18, 34, 50, 101, 152]:
    image = torch.rand(size=(1, 3, 224, 224))
    resnet = torchvision.models.resnet.__dict__[f"resnet{num_layers}"](pretrained=True)
    model = torch.nn.Sequential(*list(resnet.children())[:-1])
    model.eval()
    out = model(image)
    print(out.shape)
