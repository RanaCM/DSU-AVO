import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, VisualEncoder, Decoder
from .modules import TextVideoAligner
from utils.tools import get_mask_from_lengths

import numpy as np


class Aligner(nn.Module):
    """ Text-Visual Aligner """

    def __init__(self, preprocess_config, model_config, output_attn=False):
        super(Aligner, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.tva = TextVideoAligner(model_config)

        sync_dim = model_config["visual"]["sync_feat_size"]
        self.sync_linear = nn.Linear(sync_dim, model_config["transformer"]["encoder_hidden"]) 
        self.sync_encoder = VisualEncoder(model_config, feature_type="sync")

        self.text_video_aligner = TextVideoAligner(model_config)
        
        self.class_linear = nn.Linear(model_config["transformer"]["encoder_hidden"], preprocess_config["preprocessing"]["content"]["n_centroids"])

        self.res = model_config["aligner"]["residual_connection"]
        if self.res:
            self.dropout = nn.Dropout(model_config["aligner"]["dropout"])
            
        self.pho_to_con = model_config["aligner"]["pho_to_con"]
                
        self.predictor = Decoder(model_config)


    def forward(
        self,
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
        d_targets=None,
    ):
        
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        con_masks = (
            get_mask_from_lengths(con_lens, max_con_len)
            if con_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        if self.sync_linear is not None:
            syncs = self.sync_linear(syncs) # (B, max_sync_len, d_model)

        sync_masks = get_mask_from_lengths(sync_lens, max_sync_len)
        if self.sync_encoder is not None:
            syncs = self.sync_encoder(syncs, sync_masks)

        _, tv_attn, repeated_sync_nonpadding, diagonal_loss, diagonal_mask = self.text_video_aligner(output, syncs, src_lens, sync_lens)

        output = torch.bmm(tv_attn, output)
        if self.res:
            output = output + self.dropout(syncs)
        output = torch.repeat_interleave(output, self.pho_to_con, dim=1)
                

        if max_con_len is None: 
            con_masks = ~ repeated_sync_nonpadding.squeeze(2)
            
        sync_masks = torch.repeat_interleave(sync_masks, self.pho_to_con, dim=1)
        output, _ = self.predictor(output, sync_masks)
            
        output = self.class_linear(output)

        return (
            output,
            src_masks,
            con_masks,
            src_lens,
            con_lens,
            tv_attn,
            diagonal_loss,
        )
