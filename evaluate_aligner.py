import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model_aligner
from utils.tools import to_device_aligner, log_aligner, align_one_sample
from model import AlignerLoss
from dataset_aligner import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, configs, logger=None):
    preprocess_config, model_config, train_config = configs

    if "val_list" in preprocess_config["path"]:
        val_list = preprocess_config["path"]["val_list"]
    else:
        val_list = "val.txt"
    # Get dataset
    dataset = Dataset(
        val_list, preprocess_config, model_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = AlignerLoss(preprocess_config, model_config).to(device)

    # Evaluation
    loss_sums = [0 for _ in range(6)]
    for batchs in loader:
        for batch in batchs:
            batch = to_device_aligner(batch, device)
            with torch.no_grad():
                # Forward
                output = model(*(batch[2:]))

                # Cal Loss
                losses = Loss(batch, output)

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])

    loss_sums = loss_sums[:len(losses)] 
    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.4f}, Content Loss: {:.4f}, Diagonal Loss: {:.4f}".format(
        *([step] + [l for l in loss_means]) 
    )

    if logger is not None:
        fig, tag = align_one_sample(
            batch,
            output,
            model_config,
            preprocess_config,
        )

        log_aligner(logger, step, losses=loss_means)
        log_aligner(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )

    return message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model_aligner(args, configs, device, train=False).to(device)

    message = evaluate(model, args.restore_step, configs)
    print(message)
