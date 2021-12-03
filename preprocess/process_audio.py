import argparse
import json
import logging
import math
import os
import time
from collections import Counter
from pathlib import Path
from time import sleep
from typing import List, Tuple, Union

import h5py
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from puts import get_logger, timeitprint
from tqdm import tqdm

logger = get_logger(stream_only=True)
logger.setLevel(logging.INFO)


def get_waveform(filepath: str, resample_rate: int = None) -> Tuple[torch.Tensor, int]:
    """
    Loads a waveform from a filepath and resamples it to a specified rate.

    :param filepath: Path to the audio file.
    :param resample_rate: The rate to resample the audio to.

    :return: A tuple of the waveform and the sample rate.
    """
    effects = [["remix", "1"]]
    if resample_rate:
        effects.extend(
            [
                ["lowpass", f"{resample_rate // 2}"],
                ["rate", f"{resample_rate}"],
            ]
        )
    return torchaudio.sox_effects.apply_effects_file(filepath, effects=effects)


def get_waveform_alt(filepath: str, resample_rate: int = 8000):
    waveform, sample_rate = torchaudio.load(filepath)
    resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
    resampled_waveform = resampler(waveform)
    return resampled_waveform


def trim_or_pad_time(data: torch.Tensor, target_time: int) -> torch.Tensor:
    """
    Trims or pads a tensor to a specific time length.

    :param data: The data to trim or pad.
    :param target_time: The time to trim or pad the data to.

    :return: The trimmed or padded data.
    """
    if data.shape[-1] > target_time:
        # NOTE: Ellipsis(...) expands to the number of colon(:) objects
        # needed for the selection tuple to index all dimensions.
        return data[..., :target_time]
    elif data.shape[-1] < target_time:
        # NOTE: FUNCTIONAL.PAD
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        pad_width = target_time - data.shape[-1]
        return torch.nn.functional.pad(data, pad=(0, pad_width), mode="replicate")
    else:
        # no trim nor pad necessary
        return data


def get_MFCC(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Computes the MFCCs of a waveform.

        :param waveform: The waveform to compute the MFCCs of. (B, time)
        :param sample_rate: The sample rate of the waveform.

        :return: The MFCCs of the waveform. (B, n_mfcc, time')
    """

    n_fft = 2048
    win_length = None
    hop_length = 512
    n_mels = 256
    n_mfcc = 256  # Number of mfc coefficients to retain

    # NOTE: MFCC Source
    # https://pytorch.org/audio/stable/transforms.html#torchaudio.transforms.MFCC
    # https://pytorch.org/audio/stable/_modules/torchaudio/transforms.html#MFCC
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "n_mels": n_mels,
            "hop_length": hop_length,
            "mel_scale": "htk",
        },
    )
    mfcc = mfcc_transform(waveform)  # (B, n_mfcc, time')
    return mfcc


def get_audio_paths(
    json_dir: Union[str, Path],
    media_dir: Union[str, Path],
) -> List[Tuple[str, int]]:
    audio_paths: List[Tuple[str, int]] = []
    assert Path(json_dir).is_dir()
    assert Path(media_dir).is_dir()

    for name in os.listdir(json_dir):
        if name.endswith(".json"):
            json_path = Path(json_dir) / name
            with open(json_path, "r") as f:
                data = json.load(f)
                media_filenames = data.get("media_filenames", [])
                audio_file = None
                for i in media_filenames:
                    if i.endswith(".mp3"):
                        audio_file = i
                        break

                if audio_file is not None:
                    audio_path = Path(media_dir).resolve() / audio_file
                    if audio_path.is_file():
                        audio_paths.append((str(audio_path), int(data["id"])))
                    else:
                        logger.error(f"{audio_file} does not exists")

    return audio_paths


def read_audio_to_tensor(
    audio_path: str,
    resample_rate: int = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    output_numpy: bool = False,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Reads an audio file to a tensor.

    :param audio_path: The path to the audio file.
    :param resample_rate: The rate to resample the audio to.
    :param dtype: The dtype of the tensor.
    :param device: The device to store the tensor on.

    :return: The MFCC feature of the audio file in either a torch.Tensor or a numpy array.
    """
    waveform, sample_rate = get_waveform(
        filepath=audio_path,
        resample_rate=resample_rate,
    )
    mfcc = get_MFCC(waveform=waveform, sample_rate=sample_rate)
    mfcc = mfcc.to(dtype=dtype, device=device)

    return mfcc.numpy() if output_numpy else mfcc


def extract_feats_and_generate_h5(
    audio_paths: List[Tuple[str, int]],
    resample_rate: int = 8000,
    h5_filepath: str = "feats.h5",
    features_dim: int = 3600,
) -> None:
    """
    Args:
        audio_paths: list of audio paths and ids
        h5_filepath: path of output file to be written
        features_dim: expected dimension of the features vector
    Returns:
        None
    Side effect:
        creating a h5 file containing visual features of given list of videos.
    """

    dataset_size = len(audio_paths)
    processed_count = 0
    invalid_count = 0

    if Path(h5_filepath).is_file():
        logger.warning(f"{h5_filepath} already exists")
        return

    with h5py.File(h5_filepath, "w") as fd:
        feature_dataset = fd.create_dataset(
            name="audio_features",
            shape=(dataset_size, features_dim),
            dtype=np.float32,
        )
        feature_ids_dataset = fd.create_dataset(
            name="ids",
            shape=(dataset_size,),
            dtype=np.int,
        )
        time_start = time.time()
        for i, (audio_path, audio_id) in enumerate(audio_paths):
            try:
                feats = read_audio_to_tensor(
                    audio_path,
                    resample_rate=resample_rate,
                    output_numpy=True,
                )
                feats = feats.squeeze()  # remove dimension of length 1
            except Exception as e:
                feats = np.zeros(shape=(features_dim,))
                invalid_count += 1
                logger.error(e)

            feature_dataset[i : i + 1] = feats
            feature_ids_dataset[i : i + 1] = audio_id
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


def get_avg_mfcc_time_length(audio_paths, resample_rate: int):
    hop_length = 512
    all_mfcc_lens = []

    for audio_path, _ in tqdm(audio_paths):
        waveform, _ = get_waveform(audio_path, resample_rate=resample_rate)
        waveform_len = waveform.shape[-1]
        mfcc_len = math.ceil(waveform_len / hop_length)
        all_mfcc_lens.append(mfcc_len)

    counter = Counter(all_mfcc_lens)

    most_common_mfcc_len = counter.most_common(1)[0][0]
    top5 = counter.most_common(5)

    max_mfcc_len = max(all_mfcc_lens)
    avg_mfcc_len = sum(all_mfcc_lens) / len(all_mfcc_lens)
    min_mfcc_len = min(all_mfcc_lens)

    print(f"max mfcc len: {max_mfcc_len}")
    print(f"avg mfcc len: {avg_mfcc_len}")
    print(f"min mfcc len: {min_mfcc_len}")
    print("Top 5 most common mfcc len:")
    for i in top5:
        print(i)

    return most_common_mfcc_len


@timeitprint
def test():
    a_mp3 = Path("/home/mark/Data/NFT_Dataset/media/08416.mp3")
    waveform, sample_rate = get_waveform(filepath=a_mp3, resample_rate=8000)
    mfcc = get_MFCC(waveform, sample_rate)
    print(mfcc.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--media_dir", type=str, required=True)
    parser.add_argument("--h5_filepath", type=str, default="data/audio_feats.h5")
    parser.add_argument("--resample_rate", type=int, default=8000)
    parser.add_argument("--features_dim", type=int, default=None)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_paths = get_audio_paths(
        json_dir=args.json_dir,
        media_dir=args.media_dir,
    )

    if args.features_dim is None:
        get_avg_mfcc_time_length(audio_paths, args.resample_rate)
        features_dim = None
        while features_dim is None:
            try:
                features_dim = int(input("Enter features dim: "))
            except ValueError:
                print("Please enter a valid number")
    else:
        features_dim = args.features_dim

    print("Features dimensions: ", features_dim)
    sleep(5)

    extract_feats_and_generate_h5(
        audio_paths=audio_paths,
        resample_rate=args.resample_rate,
        h5_filepath=args.h5_filepath,
        features_dim=features_dim,
    )


if __name__ == "__main__":
    main()
