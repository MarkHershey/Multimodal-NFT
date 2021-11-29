import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import numpy as np
import torch
import torchvision
from PIL import Image
from puts import get_logger

logger = get_logger(stream_only=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_resnet50():
    cnn = torchvision.models.resnet50(pretrained=True)
    # model = torch.nn.Sequential(*list(cnn.children())[:-1])
    model = cnn.to(device)
    model.eval()
    return model


def get_image_paths(
    json_dir: Union[str, Path],
    media_dir: Union[str, Path],
) -> List[Tuple[str, int]]:
    image_paths: List[Tuple[str, int]] = []
    assert Path(json_dir).is_dir()
    assert Path(media_dir).is_dir()

    for name in os.listdir(json_dir):
        if name.endswith(".json"):
            json_path = Path(json_dir) / name
            with open(json_path, "r") as f:
                data = json.load(f)
                media_filenames = data.get("media_filenames", [])
                image_file = None
                for i in media_filenames:
                    if i.endswith(".jpg"):
                        image_file = i
                        break
                    elif i.endswith(".png"):
                        logger.warning(f"{json_path} has unexpected png")

                if image_file is not None:
                    image_path = Path(media_dir).resolve() / image_file
                    if image_path.is_file():
                        image_paths.append((str(image_path), int(data["id"])))
                    else:
                        logger.error(f"{image_file} does not exists")

    return image_paths


def read_image_to_tensor_slow(image_path: str, resize: bool = False) -> torch.Tensor:
    assert Path(image_path).is_file()
    image = Image.open(image_path)
    if resize:
        image = image.resize((224, 224), Image.ANTIALIAS)
    image = np.array(image)
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)
    return image.to(device)  # (1, 3, W, H)


def read_image_to_tensor(image_path: str, resize: bool = False) -> torch.Tensor:
    assert Path(image_path).is_file()
    image = Image.open(image_path)
    if resize:
        image = image.resize((224, 224), Image.ANTIALIAS)
    image = torchvision.transforms.ToTensor()(image)  # (3, W, H)
    image = image.unsqueeze(0)  # (3, W, H) -> (1, 3, W, H)
    return image.to(device)  # (1, 3, W, H)


def extract_feats_and_generate_h5(
    model,
    image_paths: List[Tuple[str, int]],
    h5_filepath: str = "feats.h5",
    features_dim: int = 1000,  # 1000 for resnet50
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

    dataset_size = len(image_paths)
    processed_count = 0
    invalid_count = 0

    if Path(h5_filepath).is_file():
        logger.warning(f"{h5_filepath} already exists")
        return

    with h5py.File(h5_filepath, "w") as fd:
        feature_dataset = fd.create_dataset(
            name="video_features",
            shape=(dataset_size, features_dim),
            dtype=np.float32,
        )
        feature_ids_dataset = fd.create_dataset(
            name="ids",
            shape=(dataset_size,),
            dtype=np.int,
        )
        time_start = time.time()
        for i, (image_path, image_id) in enumerate(image_paths):

            try:
                # read an image
                image = read_image_to_tensor(image_path)  # (1, 3, W, H)
                # extract features
                feats = model(image)  # (1, 1000)
                feats = feats.squeeze()  # (1000, )
                feats = feats.data.cpu().clone().numpy()
            except Exception as e:
                feats = np.zeros(shape=(features_dim,))
                invalid_count += 1
                logger.error(e)

            feature_dataset[i : i + 1] = feats
            feature_ids_dataset[i : i + 1] = image_id
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

    logger.info(f"processed count: {processed_count}")
    logger.info(f"invalid count  : {invalid_count}")
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--media_dir", type=str, required=True)
    parser.add_argument("--h5_filepath", type=str, default="images_feats.h5")
    parser.add_argument("--features_dim", type=int, default=1000)
    args = parser.parse_args()

    model = build_resnet50()
    image_paths = get_image_paths(
        json_dir=args.json_dir,
        media_dir=args.media_dir,
    )
    extract_feats_and_generate_h5(
        model,
        image_paths,
        args.h5_filepath,
        args.features_dim,
    )


if __name__ == "__main__":
    main()