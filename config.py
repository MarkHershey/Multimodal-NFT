import json
import os
import time
from pathlib import Path
from time import strftime

default_configs = dict(
    # General
    exp_name="default",
    stream_log_only=False,
    mode="train",
    train_ratio=0.8,
    val_flag=True,  # whether split the training data into validation set
    batch_size=64,
    optimizer="Adam",
    learning_rate=0.0001,
    max_epochs=30,
    save_dir="results/",  # directory to save checkpoints
    # Restore training
    restore_flag=False,  # whether to restore training from checkpoint
    restore_path="",  # path to checkpoint
    # Datasets related
    task="classification",  # classification or regression
    json_dir="/home/mark/Data/NFT_Dataset/json",  # label json directory
    text_pickle="data/encoded_text.pickle",  # preprocessed text pickle file path
    image_feat_h5="data/image_feats_resnet34_512.h5",  # preprocessed image feature h5 file path
    video_feat_h5="data/video_feats.h5",  # preprocessed video feature h5 file path
    audio_feat_h5="",  # preprocessed audio feature h5 file path
    # Model related
    word_dim=300,  # word embedding dimension
    text_rnn_dim=256,  # image feature dimension
    visual_in_dim=512,  # image feature dimension
    motion_in_frames=16,  # number of frames sampled per video
    motion_in_dim=512,  # video feature dimension
    motion_mid_dim=128,  # video feature middle representation dimension
    agg_in_dim=256,  # feature aggregation input dimension
    agg_mid_dim=256,  # feature aggregation middle representation dimension
    agg_out_dim=256,  # feature aggregation output dimension
    num_classes=10,  # number of classes for classification task
    # misc
    seed=666,  # seed for random number generator
    gpu_ids="0",  # specify single or multiple GPU id to use
    num_workers=0,  # num of workers to use for DataLoader, 0 means no limit
    use_all_gpus=False,  # whether to use all gpus
    deterministic=False,  # whether to use all gpus
)


class ExpConfigs:
    """ Experiment Configurations"""

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

        exp_dir = Path(self.save_dir) / self.exp_name
        ckpt_dir = exp_dir / "ckpt"
        log_dir = exp_dir / "logs"

        if not ckpt_dir.is_dir():
            ckpt_dir.mkdir(parents=True)
        if not log_dir.is_dir():
            log_dir.mkdir(parents=True)

        self.exp_dir = str(exp_dir)
        self.ckpt_dir = str(ckpt_dir)
        self.log_dir = str(log_dir)

        # write configs to experiment directory as a record
        save_name = exp_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
        self.save(save_name)

    def __repr__(self):
        # return str(self.__dict__)
        return json.dumps(self.__dict__, indent=4)

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.__repr__())

    @staticmethod
    def load(path):
        with open(path, "r") as f:
            return ExpConfigs(**json.loads(f.read()))


if __name__ == "__main__":
    config = ExpConfigs()
    config.save("cfgs/template.json")
