import json
import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class NFTDataset(Dataset):
    """Dataset stores the samples and their corresponding labels."""

    def __init__(
        self,
        texts_ids: List[int],
        texts_encoded: np.ndarray,
        texts_lengths: np.ndarray,
        labels: Dict[int, int],
        image_feat_h5: Union[str, Path],
        image_id_to_h5_idx: Dict[int, int],
        video_feat_h5: Union[str, Path],
        video_id_to_h5_idx: Dict[int, int],
        audio_feat_h5: Union[str, Path],
        audio_id_to_h5_idx: Dict[int, int],
        visual_in_dim: int,
        motion_in_frames: int,
        motion_in_dim: int,
        audio_mfcc_dim: int,
        audio_time_dim: int,
        text_only: bool = False,
    ):
        self.texts_ids = texts_ids
        self.texts_encoded = torch.LongTensor(texts_encoded)
        self.texts_lengths = torch.LongTensor(texts_lengths)
        self.label_ids = list(labels.keys())
        self.labels = labels
        self.image_feat_h5 = image_feat_h5
        self.image_id_to_h5_idx = image_id_to_h5_idx
        self.video_feat_h5 = video_feat_h5
        self.video_id_to_h5_idx = video_id_to_h5_idx
        self.audio_feat_h5 = audio_feat_h5
        self.audio_id_to_h5_idx = audio_id_to_h5_idx
        self.visual_in_dim = visual_in_dim
        self.motion_in_frames = motion_in_frames
        self.motion_in_dim = motion_in_dim
        self.audio_mfcc_dim = audio_mfcc_dim
        self.audio_time_dim = audio_time_dim
        self.text_only = text_only

    def __len__(self):
        # this is number of samples
        return len(self.label_ids)

    def get_image_feat(self, image_idx):
        with h5py.File(self.image_feat_h5, "r") as f:
            image_feat = f["image_features"][image_idx]
        return torch.from_numpy(image_feat)

    def get_video_feat(self, video_idx):
        with h5py.File(self.video_feat_h5, "r") as f:
            video_feat = f["video_features"][video_idx]
        return torch.from_numpy(video_feat)

    def get_audio_feat(self, audio_idx):
        with h5py.File(self.audio_feat_h5, "r") as f:
            audio_feat = f["audio_features"][audio_idx]
        return torch.from_numpy(audio_feat)

    def __getitem__(self, idx):
        sample_id = self.label_ids[idx]

        # get encoded text
        text_idx = self.texts_ids.index(sample_id)
        text_encoded = self.texts_encoded[text_idx]
        text_length = self.texts_lengths[text_idx]

        # get image features
        if not self.text_only and sample_id in self.image_id_to_h5_idx:
            image_idx = self.image_id_to_h5_idx[sample_id]
            image_feat = self.get_image_feat(image_idx)
        else:
            image_feat = torch.zeros(self.visual_in_dim)

        # get video features
        if not self.text_only and sample_id in self.video_id_to_h5_idx:
            video_idx = self.video_id_to_h5_idx[sample_id]
            video_feat = self.get_video_feat(video_idx)
        else:
            video_feat = torch.zeros(self.motion_in_frames, self.motion_in_dim)

        # get audio features
        if not self.text_only and sample_id in self.audio_id_to_h5_idx:
            audio_idx = self.audio_id_to_h5_idx[sample_id]
            audio_feat = self.get_audio_feat(audio_idx)
        else:
            audio_feat = torch.zeros(self.audio_mfcc_dim, self.audio_time_dim)

        # get label
        label: int = self.labels[sample_id]

        return (
            sample_id,
            text_encoded,
            text_length,
            image_feat,
            video_feat,
            audio_feat,
            label,
        )


class NFTDataLoader(DataLoader):
    """
    DataLoader wraps an iterable around the Dataset to enable easy access to the samples.

    1. Collect data according to the kwargs during initilization.
    2. Build Dataset.
    3. Pass Dataset to super to complete initialization of DataLoader.

    """

    def __init__(self, **kwargs):
        # 1. Collect data according to the kwargs during initilization.
        self.batch_size = kwargs.get("batch_size", 16)
        self.json_dir = str(kwargs.pop("json_dir"))
        self.json_names: List[str] = kwargs.pop("json_names")
        self.text_pickle = str(kwargs.pop("text_pickle"))
        self.image_feat_h5 = str(kwargs.pop("image_feat_h5"))
        self.video_feat_h5 = str(kwargs.pop("video_feat_h5"))
        self.audio_feat_h5 = str(kwargs.pop("audio_feat_h5"))

        self.visual_in_dim: int = kwargs.pop("visual_in_dim")
        self.motion_in_frames: int = kwargs.pop("motion_in_frames")
        self.motion_in_dim: int = kwargs.pop("motion_in_dim")
        self.audio_mfcc_dim: int = kwargs.pop("audio_mfcc_dim")
        self.audio_time_dim: int = kwargs.pop("audio_time_dim")

        self.text_only: bool = kwargs.pop("text_only")

        # get pickle object
        print(f"loading text_pickle from {self.text_pickle}")
        assert Path(self.text_pickle).is_file()
        obj = pickle.load(open(self.text_pickle, "rb"))
        texts_ids: np.ndarray = obj["texts_ids"]
        texts_encoded: np.ndarray = obj["texts_encoded"]
        texts_lengths: np.ndarray = obj["texts_lengths"]
        self.embedding_matrix: np.ndarray = obj[
            "embedding_matrix"
        ]  # could be None if in testing mode

        # get labels
        labels: Dict[int, int] = {}
        print(f"loading labels from {self.json_dir}")
        assert Path(self.json_dir).is_dir()
        for json_name in self.json_names:
            if not json_name.endswith(".json"):
                continue
            json_path = Path(self.json_dir) / json_name
            with json_path.open() as f:
                data = json.load(f)
                data_id: int = data["id"]
                data_label: int = data["price_class"]
                labels[data_id] = data_label
                f.close()
        # self.labels = labels
        self.num_samples = len(labels)

        # get image_id_to_h5_idx
        print(f"loading image feature from {self.image_feat_h5}")
        assert Path(self.image_feat_h5).is_file()
        with h5py.File(self.image_feat_h5, "r") as f:
            image_idx_to_id = f["ids"][()]
        image_id_to_h5_idx = {str(vid): i for i, vid in enumerate(image_idx_to_id)}

        # get video_id_to_h5_idx
        print(f"loading video feature from {self.video_feat_h5}")
        assert Path(self.video_feat_h5).is_file()
        with h5py.File(self.video_feat_h5, "r") as f:
            video_idx_to_id = f["ids"][()]
        video_id_to_h5_idx = {str(vid): i for i, vid in enumerate(video_idx_to_id)}

        # get audio_id_to_h5_idx
        print(f"loading audio feature from {self.audio_feat_h5}")
        assert Path(self.audio_feat_h5).is_file()
        with h5py.File(self.audio_feat_h5, "r") as f:
            audio_idx_to_id = f["ids"][()]
        audio_id_to_h5_idx = {str(vid): i for i, vid in enumerate(audio_idx_to_id)}

        # 2. Build Dataset.
        self.dataset = NFTDataset(
            texts_ids=texts_ids,
            texts_encoded=texts_encoded,
            texts_lengths=texts_lengths,
            labels=labels,
            image_feat_h5=self.image_feat_h5,
            image_id_to_h5_idx=image_id_to_h5_idx,
            video_feat_h5=self.video_feat_h5,
            video_id_to_h5_idx=video_id_to_h5_idx,
            audio_feat_h5=self.audio_feat_h5,
            audio_id_to_h5_idx=audio_id_to_h5_idx,
            visual_in_dim=self.visual_in_dim,
            motion_in_frames=self.motion_in_frames,
            motion_in_dim=self.motion_in_dim,
            audio_mfcc_dim=self.audio_mfcc_dim,
            audio_time_dim=self.audio_time_dim,
            text_only=self.text_only,
        )
        # 3. Pass Dataset to super to complete initialization of DataLoader.
        super().__init__(self.dataset, **kwargs)

        def __len__(self):
            # this is number of batches
            return math.ceil(len(self.dataset) / self.batch_size)


if __name__ == "__main__":
    print("Test DataLoader...\n")
    json_dir = "/home/mark/Data/NFT_Dataset/json"
    all_json_names = []
    for i in os.listdir(json_dir):
        if i.endswith(".json"):
            all_json_names.append(i)

    train_loader_kwargs = {
        "batch_size": 16,
        "json_dir": json_dir,
        "json_names": all_json_names,
        "text_pickle": "data/encoded_text.pickle",
        "image_feat_h5": "data/image_feats_resnet34_512.h5",
        "video_feat_h5": "data/video_feats.h5",
        "audio_feat_h5": "data/audio_feats.h5",
        "visual_in_dim": 512,
        "motion_in_frames": 16,
        "motion_in_dim": 512,
        "audio_mfcc_dim": 256,
        "audio_time_dim": 3600,
        "text_only": False,
        "num_workers": 0,
        "shuffle": True,
    }
    train_loader = NFTDataLoader(**train_loader_kwargs)

    print(f"\ndataloader length: {len(train_loader)} (batches)")
    print(f"dataset length: {len(train_loader.dataset)} (samples)")

    for (
        ids,
        text_encoded,
        text_lengths,
        image_feat,
        video_feat,
        audio_feat,
        label,
    ) in train_loader:
        print()
        print(f"text_encoded: {text_encoded.shape}")
        print(f"text_length: {text_lengths.shape}")
        print("image_feat:", image_feat.shape)
        print("video_feat:", video_feat.shape)
        print("audio_feat:", audio_feat.shape)
        print("label:", label.shape)
        break
