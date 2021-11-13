import argparse
import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import skimage.transform
import skvideo.io
import torch
import torchvision
from PIL import Image
from puts import get_logger
from torch import nn

logger = get_logger(stream_only=True)


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


def sample_video_frames(
    video_path: str,
    frame_num: int = 8,
    frame_num_max: int = 60,
    padding_mode: str = "edge",
    variable_sample_length: bool = False,
) -> Tuple[np.ndarray, bool]:
    """
    Read and Sample frames from a video, outputs a numpy array of frames.

    Args:
        video_path: path to the video, it can be a '.gif' or '.mp4' file
        frame_num: (minimum) number of frames to sample
        frame_num_max: maximum number of frames to sample
        padding_mode: 'edge' or 'zero', default is 'edge'
        variable_sample_length: whether to sample a variable length of frames,
                                if False, the length is fixed to frame_num

    Returns:
        video_data:  (T, C, H, W) (frame_num, 3, 224, 224)
        valid: True if video is valid, False otherwise
    """
    output_image_size = (224, 224)
    valid = True
    try:
        # NOTE: Load a video from file entirely into memory.
        # returns a ndarray of dimension (T, M, N, C), where T is the number of frames,
        # M is the height, N is width, and C is depth.
        # ref: http://www.scikit-video.org/stable/modules/generated/skvideo.io.vread.html
        video_data: np.ndarray = skvideo.io.vread(video_path)
        vid_frameCount = video_data.shape[0]
        vid_hight = video_data.shape[1]
        vid_width = video_data.shape[2]
        vid_channel = video_data.shape[3]

    except:
        logger.error(f"file error: {video_path}")
        valid = False
        # return all ZERO features, valid = False
        return np.zeros(shape=(frame_num, 3, 224, 224)), valid

    images = []
    if vid_frameCount < frame_num:
        # Pad video data to the minimum frame_num required
        to_pad_n = frame_num - vid_frameCount
        npad = ((0, to_pad_n), (0, 0), (0, 0), (0, 0))
        if padding_mode == "edge":
            video_data = np.pad(video_data, pad_width=npad, mode="edge")
        elif padding_mode == "zero":
            video_data = np.pad(video_data, npad, mode="constant", constant_values=0)
        else:
            logger.warning(f"Invalid padding_mode '{padding_mode}' fall back to 'edge'")
            video_data = np.pad(video_data, pad_width=npad, mode="edge")

        for i in range(frame_num):
            # resize image to output_image_size
            # ref: https://scikit-image.org/docs/stable/auto_examples/transform/plot_rescale.html
            img = skimage.transform.resize(
                video_data[i],
                output_image_size,
                anti_aliasing=True,
            )
            images.append(img)
    else:
        if variable_sample_length:
            output_n_frames: int = vid_frameCount // 30
            if output_n_frames > frame_num_max:
                output_n_frames = frame_num_max
            elif output_n_frames < frame_num:
                output_n_frames = frame_num
        else:
            output_n_frames = frame_num

        sample_indices = np.linspace(
            start=0,
            stop=vid_frameCount - 1,
            num=output_n_frames,
            dtype=np.int32,
        )
        for i in sample_indices:
            # resize image to output_image_size
            img = skimage.transform.resize(
                video_data[i],
                output_image_size,
                anti_aliasing=True,
            )
            images.append(img)

    video_data = np.array(images)
    video_data = video_data.transpose((0, 3, 1, 2))  # T, C, H, W

    return video_data, valid


if __name__ == "__main__":
    data_root = Path("/home/mark/Data/Cawin-NFT2")
    gif = data_root / "jinglebe-nft-collection_2.gif"
    png = data_root / "doodles-official_1037.png"
    mp4 = data_root / "niftysaxspheres_321.mp4"

    sample_video_frames(str(png), 10)
    sample_video_frames(str(gif), 10)
    sample_video_frames(str(mp4), 10)
