import argparse
import logging
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from puts import get_logger
from termcolor import colored

from config import ExpConfigs
from DataLoader import NFTDataLoader
from model.MMNFT import MMNFT


def train(cfg: ExpConfigs):

    # read json_dir
    all_json_names = []
    for i in os.listdir(cfg.json_dir):
        if i.endswith(".json"):
            all_json_names.append(i)

    # split train/val/test
    random.shuffle(all_json_names)
    split_index = int(len(all_json_names) * cfg.train_ratio)
    train_json_names = all_json_names[:split_index]
    val_json_names = all_json_names[split_index:]

    ###########################################################################
    logger.info("Create train_loader and val_loader.........")
    # load training data
    train_loader_kwargs = {
        "batch_size": cfg.batch_size,
        "json_dir": cfg.json_dir,
        "json_names": train_json_names,
        "text_pickle": cfg.text_pickle,
        "image_feat_h5": cfg.image_feat_h5,
        "video_feat_h5": cfg.video_feat_h5,
        "num_workers": cfg.num_workers,
        "shuffle": True,
    }
    train_loader = NFTDataLoader(**train_loader_kwargs)
    logger.info(f"Number of train instances: {len(train_loader.dataset)}")

    # load validation data
    if cfg.val_flag:
        val_loader_kwargs = {
            "batch_size": cfg.batch_size,
            "json_dir": cfg.json_dir,
            "json_names": val_json_names,
            "text_pickle": cfg.text_pickle,
            "image_feat_h5": cfg.image_feat_h5,
            "video_feat_h5": cfg.video_feat_h5,
            "num_workers": cfg.num_workers,
            "shuffle": True,
        }
        val_loader = NFTDataLoader(**val_loader_kwargs)
        logger.info(f"Number of val instances: {len(val_loader.dataset)}")

    ###########################################################################

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if not cfg.use_all_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids

    ###########################################################################
    pkl_obj = pickle.load(open(cfg.text_pickle, "rb"))

    embedding_matrix = pkl_obj.get("embedding_matrix")

    if embedding_matrix is not None:
        embedding_matrix = torch.from_numpy(embedding_matrix).to(device)
        vocab_size, wordvec_dim = embedding_matrix.size()
    else:
        raise ValueError("embedding_matrix is None")

    logger.info("Create model.........")

    # TODO
    model_kwargs = dict(
        vocab_size=vocab_size,
        wordvec_dim=wordvec_dim,
        glove_matrix=embedding_matrix,
        text_rnn_dim=cfg.text_rnn_dim,
        visual_in_dim=cfg.visual_in_dim,
        motion_in_frames=cfg.motion_in_frames,
        motion_in_dim=cfg.motion_in_dim,
        agg_in_dim=cfg.agg_in_dim,
        agg_out_dim=cfg.agg_out_dim,
        num_classes=cfg.num_classes,
    )

    model = MMNFT(**model_kwargs).to(device)

    # print model info
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("num of params: {}".format(pytorch_total_params))
    logger.info(model)

    # if cfg.train.glove:
    #     logger.info("Load GloVe vectors")
    #     train_loader.glove_matrix = torch.FloatTensor(train_loader.glove_matrix).to(
    #         device
    #     )
    #     with torch.no_grad():
    #         model.textual_input_module.encoder_embed.weight.set_(
    #             train_loader.glove_matrix
    #         )

    if torch.cuda.device_count() > 1:
        # use all visible GPUs
        model = nn.DataParallel(model)
        logger.info(f"Using {torch.cuda.device_count()} GPUs")

    if cfg.deterministic:
        # NOTE: REPRODUCIBILITY
        # Ref: https://pytorch.org/docs/stable/notes/randomness.html
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

    # get optimizer
    optimizer_func = getattr(optim, cfg.optimizer)
    optimizer = optimizer_func(model.parameters(), lr=cfg.learning_rate)

    start_epoch = 0
    best_val = 0 if cfg.task == "classification" else 100.0

    if cfg.restore_flag:
        logger.info("Restore checkpoint and optimizer...")
        # ckpt: checkpoint
        ckpt = torch.load(cfg.restore_path, map_location=lambda storage, loc: storage)
        start_epoch = ckpt["epoch"] + 1
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])

    if cfg.task == "classification":
        criterion = nn.CrossEntropyLoss().to(device)
    elif cfg.task == "regression":
        criterion = nn.MSELoss().to(device)

    ###########################################################################
    logger.info("Start training........")

    for epoch in range(start_epoch, cfg.max_epochs):
        print(
            ">>>>>> epoch {epoch} <<<<<<".format(
                epoch=colored(f"{epoch}", "green", attrs=["bold"])
            )
        )
        model.train()
        # total accuracy, count
        total_acc, count = 0, 0
        # batch mean_square_error sum
        batch_mse_sum = 0.0
        total_loss, avg_loss = 0.0, 0.0
        avg_loss = 0
        train_accuracy = 0
        for i, batch in enumerate(iter(train_loader)):
            progress = epoch + i / len(train_loader)

            text_encoded, text_length, image_feat, video_feat, label = batch
            text_encoded = text_encoded.to(device)
            text_length = text_length.cpu()
            image_feat = image_feat.to(device)
            video_feat = video_feat.to(device)
            label = label.to(device)

            answers = label.cuda().squeeze()
            batch_size = answers.size(0)

            optimizer.zero_grad()

            logits = model(text_encoded, text_length, image_feat, video_feat)
            # logger.debug(f">>> logits size: {logits.size()}")

            if cfg.task == "classification":
                loss = criterion(logits, answers)
                loss.backward()
                total_loss += loss.detach()
                avg_loss = total_loss / (i + 1)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
                optimizer.step()
                aggreeings = batch_accuracy(logits, answers)
            elif cfg.task == "regression":
                answers = answers.unsqueeze(-1)
                loss = criterion(logits, answers.float())
                loss.backward()
                total_loss += loss.detach()
                avg_loss = total_loss / (i + 1)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
                optimizer.step()
                preds = (logits + 0.5).long().clamp(min=1, max=10)
                batch_mse = (preds - answers) ** 2

            if cfg.task == "classification":
                total_acc += aggreeings.sum().item()
                count += answers.size(0)
                train_accuracy = total_acc / count
                sys.stdout.write(
                    "\rProgress = {progress}   ce_loss = {ce_loss}   avg_loss = {avg_loss}    train_acc = {train_acc}    avg_acc = {avg_acc}    exp: {exp_name}".format(
                        progress=colored(
                            "{:.3f}".format(progress), "green", attrs=["bold"]
                        ),
                        ce_loss=colored(
                            "{:.4f}".format(loss.item()), "blue", attrs=["bold"]
                        ),
                        avg_loss=colored(
                            "{:.4f}".format(avg_loss), "red", attrs=["bold"]
                        ),
                        train_acc=colored(
                            "{:.4f}".format(aggreeings.float().mean().cpu().numpy()),
                            "blue",
                            attrs=["bold"],
                        ),
                        avg_acc=colored(
                            "{:.4f}".format(train_accuracy), "red", attrs=["bold"]
                        ),
                        exp_name=cfg.exp_name,
                    )
                )
                sys.stdout.flush()
            elif cfg.task == "regression":
                batch_avg_mse = batch_mse.sum().item() / answers.size(0)
                batch_mse_sum += batch_mse.sum().item()
                count += answers.size(0)
                avg_mse = batch_mse_sum / count
                sys.stdout.write(
                    "\rProgress = {progress}   ce_loss = {ce_loss}   avg_loss = {avg_loss}    train_mse = {train_mse}    avg_mse = {avg_mse}    exp: {exp_name}".format(
                        progress=colored(
                            "{:.3f}".format(progress), "green", attrs=["bold"]
                        ),
                        ce_loss=colored(
                            "{:.4f}".format(loss.item()), "blue", attrs=["bold"]
                        ),
                        avg_loss=colored(
                            "{:.4f}".format(avg_loss), "red", attrs=["bold"]
                        ),
                        train_mse=colored(
                            "{:.4f}".format(batch_avg_mse), "blue", attrs=["bold"]
                        ),
                        avg_mse=colored(
                            "{:.4f}".format(avg_mse), "red", attrs=["bold"]
                        ),
                        exp_name=cfg.exp_name,
                    )
                )
                sys.stdout.flush()

        sys.stdout.write("\n")

        if cfg.task == "classification":
            if (epoch + 1) % 10 == 0:
                optimizer = step_decay(cfg, optimizer)
        elif cfg.task == "regression":
            if (epoch + 1) % 5 == 0:
                optimizer = step_decay(cfg, optimizer)

        sys.stdout.flush()

        logging.info(
            "Epoch = %s   avg_loss = %.3f    avg_acc = %.3f"
            % (epoch, avg_loss, train_accuracy)
        )

        # if cfg.val_flag:
        #     output_dir = os.path.join(cfg.dataset.save_dir, 'preds')
        #     if not os.path.exists(output_dir):
        #         os.makedirs(output_dir)
        #     else:
        #         assert os.path.isdir(output_dir)
        #     valid_acc = validate(cfg, model, val_loader, device, write_preds=False)
        #     if (valid_acc > best_val and cfg.dataset.question_type != 'count') or (valid_acc < best_val and cfg.dataset.question_type == 'count'):
        #         best_val = valid_acc
        #         # Save best model
        #         ckpt_dir = os.path.join(cfg.dataset.save_dir, 'ckpt')
        #         if not os.path.exists(ckpt_dir):
        #             os.makedirs(ckpt_dir)
        #         else:
        #             assert os.path.isdir(ckpt_dir)
        #         save_checkpoint(epoch, model, optimizer, model_kwargs_tosave, os.path.join(ckpt_dir, 'model.pt'))
        #         sys.stdout.write('\n >>>>>> save to %s <<<<<< \n' % (ckpt_dir))
        #         sys.stdout.flush()

        #     logging.info('~~~~~~ Valid Accuracy: %.4f ~~~~~~~' % valid_acc)
        #     sys.stdout.write('~~~~~~ Valid Accuracy: {valid_acc} ~~~~~~~\n'.format(
        #         valid_acc=colored("{:.4f}".format(valid_acc), "red", attrs=['bold'])))
        #     sys.stdout.flush()


def step_decay(cfg, optimizer):
    # compute the new learning rate based on decay rate
    cfg.learning_rate *= 0.5
    logging.info("Reduced learning rate to {}".format(cfg.learning_rate))
    sys.stdout.flush()
    for param_group in optimizer.param_groups:
        param_group["lr"] = cfg.learning_rate

    return optimizer


def todevice(tensor, device):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        assert isinstance(tensor[0], torch.Tensor)
        return [todevice(t, device) for t in tensor]
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device)


def batch_accuracy(predicted, true):
    """ Compute the accuracies for a batch of predictions and answers """
    predicted = predicted.detach().argmax(1)
    agreeing = predicted == true
    return agreeing


def save_checkpoint(epoch, model, optimizer, model_kwargs, filename):
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_kwargs": model_kwargs,
    }
    time.sleep(10)
    torch.save(state, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="", help="config file path")
    args = parser.parse_args()

    # Load config file
    if args.cfg:
        cfg = ExpConfigs.load(args.conf)
    else:
        cfg = ExpConfigs()

    # init logger
    global logger
    logger = get_logger(log_dir=cfg.log_dir, stream_only=cfg.stream_log_only)
    logger.setLevel(logging.INFO)

    # Set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    logger.info(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
