import os
import json

import torch
import numpy as np

from model import ScheduledOptim
from model import Aligner


def get_model_aligner(args, configs, device, train=False, output_attn=False):
    (preprocess_config, model_config, train_config) = configs

    if "output_attn" in model_config: 
        output_attn = model_config["output_attn"]
    model = Aligner(preprocess_config, model_config, output_attn=output_attn).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param
