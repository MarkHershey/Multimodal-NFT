import json

import numpy as np

default_configs = dict(
    # General
    exp_name="default",
    mode="train",
    train_ratio=0.8,
    val_flag=False,  # whether split the training data into validation set
    batch_size=16,
    optimizer="Adam",
    learning_rate=0.0001,
    max_epochs=50,
    save_dir="results/",  # directory to save checkpoints
    # Restore training
    restore_flag=False,  # whether to restore training from checkpoint
    restore_path="",  # path to checkpoint
    # Datasets related
    task="classification",  # classification or regression
    json_dir="",  # label json directory
    text_pickle="",  # preprocessed text pickle file path
    image_feat_h5="",  # preprocessed image feature h5 file path
    video_feat_h5="",  # preprocessed video feature h5 file path
    audio_feat_h5="",  # preprocessed audio feature h5 file path
    # Model related
    word_dim=300,  # word embedding dimension
    # misc
    seed=0,  # seed for random number generator
    gpu_ids="0",  # specify single or multiple GPU id to use
    num_workers=0,  # num of workers to use for DataLoader, 0 means no limit
    use_all_gpus=False,  # whether to use all gpus
)


class ExpConfigs:
    def __init__(self, **kwargs):
        self.__dict__.update(default_configs)
        self.__dict__.update(kwargs)

        assert self.mode in ["train", "test", "eval"]
        assert self.task in ["classification", "regression"]
        assert self.val_flag in [True, False]
        assert self.use_all_gpus in [True, False]
        assert self.num_workers >= 0
        assert self.optimizer in ["Adam", "SGD", "RMSprop"]
        assert isinstance(self.max_epochs, int)

    def __repr__(self):
        # return str(self.__dict__)
        return json.dumps(self.__dict__, indent=4)

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.__repr__())

    @staticmethod
    def load(path):
        with open(path, "r") as f:
            return ExpConfigs(**eval(f.read()))
