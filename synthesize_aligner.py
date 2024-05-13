import re
import argparse
from string import punctuation
import os
import json

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p

from utils.model import get_model_aligner
from utils.tools import to_device_aligner, encode_samples
from dataset_aligner import Dataset
from text import text_to_sequence
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word not in lexicon:
                lexicon[word] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = filter(None, re.split(r"([,;.\-\?\!\s+])", text))
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def encode(model, step, configs, batchs, tag, output_attn=False):
    preprocess_config, model_config, train_config = configs

    rows_losses = []
    rows_tokens = []
    len_batch = 0
    num = 0
    for batch in batchs:
        len_batch = len_batch + len(batch)
        num = num + len(batch[0][0])
        batch = to_device_aligner(batch[0], device)

        with torch.no_grad():
            output = model(*(batch[2:]))
            losses, tokens = encode_samples(
                batch,
                output,
                model_config,
                preprocess_config,
                os.path.join(train_config["path"]["result_path"], f"{int(step/1000)}k"),
                tag,
                output_attn=output_attn
            )
            rows_losses = rows_losses + losses
            rows_tokens = rows_tokens + tokens

    print(len_batch)
    print(num)
    if len(rows_tokens) != 0:
        with open(os.path.join(train_config["path"]["result_path"], f"{int(step/1000)}k", "pred_tokens.txt"), "w") as f:
            f.writelines(rows_tokens)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt",
    )
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

    # Check source texts
    assert args.source is not None and args.text is None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model_aligner(args, configs, device, train=False, output_attn=args.output_attn)

    # Get dataset
    dataset = Dataset(args.source, preprocess_config, model_config, train_config)
    batchs = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=dataset.collate_fn,
    )
    tag = None

    encode(model, args.restore_step, configs, batchs, tag, args.output_attn)
