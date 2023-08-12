from operator import mod
import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad

from transformer.SubLayers import AttentionSafe 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x


class TextVideoAligner(nn.Module):
    """ Text Video Aligner """

    def __init__(self, model_config):
        super(TextVideoAligner, self).__init__()
        self.d_model = model_config["transformer"]["encoder_hidden"] # query: visual emb
        self.n_head = model_config["aligner"]["n_head"] # 1
        self.d_k = self.d_v = model_config["transformer"]["encoder_hidden"] // model_config["aligner"]["n_head"] # key, value: text emb (256)
        self.dropout = model_config["aligner"]["dropout"]
        if "residual_connection" in model_config["aligner"]:
            self.res = model_config["aligner"]["residual_connection"]
        else:
            self.res = True

        self.tva = AttentionSafe(self.d_k, self.d_v)
        self.band_width = model_config["aligner"]["diag_band_width"]
        self.band_mask_factor = model_config["aligner"]["band_mask_factor"]
        self.repeat_num = model_config["aligner"]["repeat_num"]

    def get_diagonal_loss(self, attn, attn_ks, tgt_lens, src_padding_mask=None, tgt_padding_mask=None):
        '''
        attn: (B, L_t, L_s)
        attn_ks: shape: tensor with shape [batch_size], input_lens/output_lens
        diagonal: y=k*x (k=attn_ks, x:output, y:input)
        1 0 0
        0 1 0
        0 0 1
        y>=k*(x-width) and y<=k*(x+width):1
        else:0
        '''

        # attn: (B, max_sync_len, max_text_len)
        # or (B, src, tgt) or (B, x, y)
        # tgt: text
        # src: sync
        # b will not be multiplied by attn_ks
        # attn.sum(-1).sum(-1) == src_len
        width1 = tgt_lens / self.band_mask_factor
        width2 = tgt_lens.new(tgt_lens.size()).fill_(self.band_width).float()
        width = torch.where(width1 < width2, width1, width2).float()
        base = torch.ones(attn.size()).to(attn.device)
        zero = torch.zeros(attn.size()).to(attn.device)
        x = torch.arange(0, attn.size(1)).to(attn.device)[None, :, None].float() * base
        y = torch.arange(0, attn.size(2)).to(attn.device)[None, None, :].float() * base
        cond = (y - attn_ks[:, None, None] * x)
        cond1 = cond + width[:, None, None]
        cond2 = cond - width[:, None, None]
        mask1 = torch.where(cond1 < 0, zero, base)
        mask2 = torch.where(cond2 > 0, zero, base)
        diagonal_mask = mask1 * mask2
        if src_padding_mask is not None:
            attn = attn * (1 - src_padding_mask.float())[:, :, None]
        if tgt_padding_mask is not None:
            attn = attn * (1 - tgt_padding_mask.float())[:, None, :]
        diagonal_attn = attn * diagonal_mask
        diagonal_focus_rate = diagonal_attn.sum(-1).sum(-1) / attn.sum(-1).sum(-1)
        diagonal_loss = -diagonal_focus_rate.mean()

        return diagonal_loss, diagonal_mask

    def forward(self, text_emb, sync_emb, text_lens, sync_lens):
        text_masks = get_mask_from_lengths(text_lens) # (B, max_text_len)
        sync_masks = get_mask_from_lengths(sync_lens) # (B, max_sync_len)
        attn_masks = sync_masks.unsqueeze(1).expand(-1, torch.max(text_lens).item(), -1).transpose(1, 2) + text_masks.unsqueeze(1).expand(-1, torch.max(sync_lens).item(), -1) # (B, max_sync_len, max_text_len)
        _, attn_scores = self.tva(sync_emb, text_emb, text_emb, mask=attn_masks) # context: (B, max_sync_len, n_head*d_v==d_model); attn_scores: (B, max_sync_len, max_text_len)
        attn_scores = attn_scores.masked_fill(torch.isnan(attn_scores), 0.0)

        attn_ks = text_lens.clone().detach() / sync_lens.clone().detach()
        tgt_lens = text_lens
        src_masks = sync_masks
        tgt_masks = text_masks

        diagonal_loss, diagonal_mask = self.get_diagonal_loss(attn_scores, attn_ks, tgt_lens, src_masks, tgt_masks) 
        sync_nonpadding = ~ sync_masks[:, :, None] # (B, max_sync_len, d_model)

        repeated_sync_nonpadding = torch.repeat_interleave(sync_nonpadding, self.repeat_num, dim=1)
        repeated_context = None 

        return repeated_context, attn_scores, repeated_sync_nonpadding, diagonal_loss, diagonal_mask
