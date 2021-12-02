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

logger = get_logger(stream_only=True)
logger.setLevel(logging.INFO)


def build_resnet(num_layers: int = 50):
    assert num_layers in [18, 34, 50, 101, 152]
    resnet = torchvision.models.resnet.__dict__[f"resnet{num_layers}"](pretrained=True)
    model = torch.nn.Sequential(*list(resnet.children())[:-1])
    model = model.to(device)
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


def read_image_to_tensor(
    image_path: str, resize: bool = False, max_width: int = 1000, max_height: int = 1000
) -> torch.Tensor:
    assert Path(image_path).is_file()
    try:
        image = Image.open(image_path)
    except Exception as e:
        logger.warning(f"Error Image: {image_path}")
        logger.error(e)
        return torch.zeros(1, 3, 224, 224).to(device)

    if resize:
        image = image.resize((224, 224), Image.ANTIALIAS)
    image = torchvision.transforms.ToTensor()(image)  # (3, W, H)

    # Future work: Assert number of channels is 3
    # currently there are some images with 4 channels or 1 channel

    if image.size(1) > max_width or image.size(2) > max_height:
        image = torchvision.transforms.Resize((max_width, max_height))(image)

    image = image.unsqueeze(0)  # (3, W, H) -> (1, 3, W, H)
    return image.to(device)  # (1, 3, W, H)


def extract_feats_and_generate_h5(
    model,
    image_paths: List[Tuple[str, int]],
    h5_filepath: str = "feats.h5",
    features_dim: int = 2048,  # 2048 for resnet50
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
            name="image_features",
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
                feats = model(image)  # (1, 2048, 1, 1) for resnet50
                feats = feats.squeeze()  # (2048, )
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


def _test_image_io():
    image_paths = get_image_paths(
        json_dir="/home/mark/Data/NFT_Dataset/json",
        media_dir="/home/mark/Data/NFT_Dataset/media",
    )
    for i in image_paths:
        image = read_image_to_tensor(i[0])
        print(image.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--media_dir", type=str, required=True)
    parser.add_argument("--h5_filepath", type=str, default="data/image_feats.h5")
    parser.add_argument("--out_dir", type=str, default="data/")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--all", action="store_true", default=False)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths = get_image_paths(
        json_dir=args.json_dir,
        media_dir=args.media_dir,
    )

    if args.all:
        for num_layers, features_dim in [
            (18, 512),
            (34, 512),
            (50, 2048),
            (101, 2048),
            (152, 2048),
        ]:
            logging.info(f"Extracting features using ResNet-{num_layers}...")
            model = build_resnet(num_layers=num_layers)
            out_dir_path = Path(args.out_dir)
            if not out_dir_path.is_dir():
                out_dir_path.mkdir(parents=True)

            h5_filepath = (
                out_dir_path / f"image_feats_resnet{num_layers}_{features_dim}.h5"
            )
            extract_feats_and_generate_h5(model, image_paths, h5_filepath, features_dim)
    else:
        model = build_resnet(num_layers=50)
        extract_feats_and_generate_h5(model, image_paths, args.h5_filepath, 2048)


if __name__ == "__main__":
    main()
