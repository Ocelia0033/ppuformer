# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn

from .patchtst.model import Model as PatchTSTModel


class _PatchTSTConfig:

    def __init__(
        self,
        enc_in: int,
        seq_len: int,
        label_len: int,
        pred_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        factor: int = 5,
        patch_len: int = 16,
        stride: int = 8,
    ):
        self.task_name = "short_term_forecast"
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.dec_in = enc_in
        self.c_out = enc_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = 1
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.factor = factor
        self.output_attention = 0
        self.embed = "timeF"
        self.freq = "h"
        self.patch_len = patch_len
        self.stride = stride


class PatchTSTWrapper(nn.Module):

    def __init__(
        self,
        num_variates: int = 5,
        seq_len: int = 168,
        label_len: int = 48,
        pred_len: int = 24,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        factor: int = 5,
        patch_len: int = 16,
        stride: int = 8,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.num_variates = num_variates

        cfg = _PatchTSTConfig(
            enc_in=num_variates,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            factor=factor,
            patch_len=patch_len,
            stride=stride,
        )
        self.patchtst = PatchTSTModel(cfg, patch_len=patch_len, stride=stride)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.patchtst(x_enc, None, None, None)
