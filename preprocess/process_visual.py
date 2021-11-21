import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import numpy as np
import skimage.transform
import skvideo.io
import torch
import torchvision
from PIL import Image
from puts import get_logger
from torch import nn

logger = get_logger(stream_only=False)


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

    # image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    # logger.debug(image_batch.shape)
    image_batch = (cur_batch / 255.0 - mean) / std
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
    tensor: bool = False,
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
        tensor: whether to return a torch.Tensor instead of a numpy.ndarray, default is False

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

    if tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        video_data = torch.FloatTensor(video_data).to(device)

    return video_data, valid


def get_video_paths(
    json_dir: Union[str, Path],
    media_dir: Union[str, Path],
) -> List[Tuple[str, int]]:
    video_paths: List[Tuple[str, int]] = []
    assert Path(json_dir).is_dir()
    assert Path(media_dir).is_dir()

    for name in os.listdir(json_dir):
        if name.endswith(".json"):
            json_path = Path(json_dir) / name
            with open(json_path, "r") as f:
                data = json.load(f)
                media_filenames = data.get("media_filenames", [])
                video_file = None
                for i in media_filenames:
                    if i.endswith(".mp4"):
                        video_file = i
                        break
                    elif i.endswith(".gif"):
                        if video_file is None:
                            video_file = i
                        else:
                            logger.warning(f"{json_path} has more than one video file")
                        break
                    else:
                        continue

                if video_file is not None:
                    video_path = str((Path(media_dir) / video_file).resolve())
                    video_paths.append((video_path, int(data["id"])))

    return video_paths


def extract_feats_and_generate_h5(
    model,
    video_paths: List[Tuple[str, int]],
    frame_num: int = 16,
    h5_filepath: str = "feats.h5",
    features_dim: int = 512,  # 512 for resnet18
) -> None:
    """
    Args:
        model: loaded pretrained model for feature extraction
        video_paths: list of video ids
        frame_num: expected numbers of frames per video
        h5_filepath: path of output file to be written
    Returns:
        None
    Side effect:
        creating a h5 file containing visual features of given list of videos.
    """

    dataset_size = len(video_paths)
    processed_count = 0
    invalid_count = 0

    with h5py.File(h5_filepath, "w") as fd:
        feature_dataset = fd.create_dataset(
            name="video_features",
            shape=(dataset_size, frame_num, features_dim),
            dtype=np.float32,
        )
        feature_ids_dataset = fd.create_dataset(
            name="ids",
            shape=(dataset_size,),
            dtype=np.int,
        )
        time_start = time.time()
        for i, (video_path, video_id) in enumerate(video_paths):

            ### get video frames as numpy array
            video_data, valid = sample_video_frames(video_path, frame_num=frame_num)
            # logger.debug(video_data.shape)
            if valid:
                feats = run_batch(video_data, model)  # (frame_num=16, features_dim)
                feats = feats.squeeze()  # remove dimension of length 1
                # logger.debug(feats.shape)
            else:
                feats = np.zeros(shape=(frame_num, features_dim))
                logger.warning(f"{video_path} is invalid")
                invalid_count += 1

            feature_dataset[i : i + 1] = feats
            feature_ids_dataset[i : i + 1] = video_id
            processed_count += 1

            mins_left = round(
                (time.time() - time_start)
                / processed_count
                * (dataset_size - processed_count)
                / 60,
            )
            processed_percent = round(processed_count / dataset_size * 100)
            logger.info(
                f"{processed_count}/{dataset_size} ({processed_percent}%) processed. Estimated time left: {mins_left} mins"
            )


def main():
    video_paths = get_video_paths(
        json_dir="/home/mark/Data/MARK_NFT/json",
        media_dir="/home/mark/Data/MARK_NFT/media",
    )
    model = build_resnet()
    extract_feats_and_generate_h5(
        model,
        video_paths,
        frame_num=16,
        h5_filepath="feats.h5",
    )


if __name__ == "__main__":
    # data_root = Path("/home/mark/Data/Cawin-NFT2")
    # gif = data_root / "jinglebe-nft-collection_2.gif"
    # png = data_root / "doodles-official_1037.png"
    # mp4 = data_root / "niftysaxspheres_321.mp4"

    # sample_video_frames(str(png), 10)
    # sample_video_frames(str(gif), 10)
    # sample_video_frames(str(mp4), 10)
    main()
