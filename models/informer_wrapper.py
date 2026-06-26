# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn

from .informer.model import Model as InformerModel


class _InformerConfig:

    def __init__(
        self,
        enc_in, dec_in, c_out,
        seq_len, label_len, pred_len,
        d_model=512, n_heads=8,
        e_layers=2, d_layers=1,
        d_ff=2048, dropout=0.05,
        activation="gelu", factor=5,
        embed="timeF", freq="h",
        distil=1, output_attention=0,
    ):
        self.task_name = "short_term_forecast"
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.factor = factor
        self.embed = embed
        self.freq = freq
        self.distil = distil
        self.output_attention = output_attention


class InformerWrapper(nn.Module):

    def __init__(
        self,
        num_variates: int = 5,
        seq_len: int = 168,
        label_len: int = 48,
        pred_len: int = 24,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 2048,
        factor: int = 5,
        dropout: float = 0.05,
        activation: str = "gelu",
        embed: str = "timeF",
        freq: str = "h",
        distil: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.num_variates = num_variates

        cfg = _InformerConfig(
            enc_in=num_variates, dec_in=num_variates, c_out=num_variates,
            seq_len=seq_len, label_len=label_len, pred_len=pred_len,
            d_model=d_model, n_heads=n_heads,
            e_layers=e_layers, d_layers=d_layers,
            d_ff=d_ff, dropout=dropout, activation=activation,
            factor=factor, embed=embed, freq=freq,
            distil=1 if distil else 0,
        )
        self.informer = InformerModel(cfg)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_mark_dec: torch.Tensor,
    ) -> torch.Tensor:
        B, L, N = x_enc.shape
        x_dec = torch.zeros(
            B, self.label_len + self.pred_len, N,
            device=x_enc.device, dtype=x_enc.dtype,
        )
        out = self.informer(x_enc, None, x_dec, None)
        return out
