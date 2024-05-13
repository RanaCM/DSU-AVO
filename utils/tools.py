import os
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt
# from matplotlib.pyplot import MultipleLocator

matplotlib.use("Agg")
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device_aligner(data, device):
    assert len(data) in [13]

    if len(data) == 13: # text, sync, content
        (
            ids,
            raw_texts,
            speakers, 
            texts,
            src_lens,
            max_src_len,
            syncs,
            sync_lens,
            max_sync_len,
            cons,
            con_lens,
            max_con_len,
            durations,  
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        syncs = torch.from_numpy(syncs).float().to(device)
        sync_lens = torch.from_numpy(sync_lens).to(device)
        cons = torch.from_numpy(cons).float().to(device) if len(cons.shape) == 3 else torch.from_numpy(cons).to(device)  
        con_lens = torch.from_numpy(con_lens).to(device)
        durations = torch.from_numpy(durations).to(device) 

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            syncs,
            sync_lens,
            max_sync_len,
            cons,
            con_lens,
            max_con_len,
            durations, 
        )


def log_aligner(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/content_loss", losses[1], step)
        logger.add_scalar("Loss/diagonal_loss", losses[2], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def align_one_sample(targets, predictions, model_config, preprocess_config): 

    basename = targets[0][0]
    src_len = predictions[3][0].item() 
    duration = targets[12][0, :src_len].detach().cpu().numpy()

    sync_len = targets[7][0].item()
    attn = predictions[5][0, :sync_len, :src_len].detach().transpose(0, 1)
    attn_repeat = np.repeat(attn.cpu().numpy(), model_config["aligner"]["repeat_num"], axis=1)
    mel_len = duration.sum() 
    gt_align = np.zeros([src_len, mel_len])
    d = 0
    for i in range(src_len):
        gt_align[i, d:d+duration[i]] = 1
        d = d + duration[i]
    fig = plot_attn(
        [
            (attn_repeat),
            (gt_align),
        ],
        ["Text-Video Attention", "MFA Alignment"], 
    )

    return fig, basename

def encode_samples(targets, predictions, model_config, preprocess_config, path, tag=None, output_attn=True): 

    os.makedirs(os.path.join(path, "attn"), exist_ok=True) 
    mae_loss = nn.L1Loss()
    losses = []
    tokens = []
    basenames = targets[0]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[3][i].item()
        sync_len = targets[7][i].item()
        duration = targets[12][i, :src_len].detach().cpu().numpy()

        if output_attn:

            attn = predictions[5][i, :sync_len, :src_len].detach().transpose(0, 1)
            attn_repeat = np.repeat(attn.cpu().numpy(), model_config["aligner"]["repeat_num"], axis=1)
            mel_len = duration.sum()
            gt_align = np.zeros([src_len, mel_len])
            d = 0
            for j in range(src_len):
                gt_align[j, d:d+duration[j]] = 1
                d = d + duration[j]
            if attn_repeat.shape[1] > gt_align.shape[1]:
                attn_repeat = attn_repeat[:, :gt_align.shape[1]]
            else:
                gt_align = gt_align[:, :attn_repeat.shape[1]]
            fig = plot_attn(
                [
                    (attn_repeat),
                    (gt_align),
                ],
                ["Text-Video Attention", "MFA Alignment"],
            )
            np.save(os.path.join(path, "attn", "{}{}{}.npy".format(basename, "_attn_aligner", f"_{tag}" if tag is not None else "")), attn_repeat) 
            np.save(os.path.join(path, "attn", "{}{}{}.npy".format(basename, "_mfa", f"_{tag}" if tag is not None else "")), gt_align) 
            loss = mae_loss(torch.from_numpy(attn_repeat).to("cuda:0"), torch.from_numpy(gt_align).to("cuda:0")).item()
            losses.append([basename, loss])
            plt.savefig(os.path.join(path, "attn", "{}{}{}.png".format(basename, "_attn", f"_{tag}" if tag is not None else "")), bbox_inches="tight") 
            plt.close()
            
        logits_prediction = predictions[0][i][:sync_len*model_config["aligner"]["pho_to_con"], :]
            
        _, tokens_prediction = torch.max(logits_prediction, dim=1)
        tokens_gt = targets[-4][i]
        min_length = min(len(tokens_gt), len(tokens_prediction))
        tokens_gt = tokens_gt[:min_length]
        tokens_prediction = tokens_prediction[:min_length]
        tokens.append(f'{{\'audio\': \'{os.path.join(os.getcwd(), preprocess_config["path"]["raw_path"].strip("./"), preprocess_config["path"]["sub_dir"], basename)}.wav\', \'hubert\': \'{" ".join([str(x) for x in tokens_prediction.cpu().numpy().tolist()])}\', \'duration\': 0.0}}\n')
            
    return losses, tokens


def plot_attn(data, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]

    for i in range(len(data)):
        attn = data[i]
        axes[i][0].imshow(attn, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, attn.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
