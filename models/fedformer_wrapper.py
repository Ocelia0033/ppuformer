# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn

from .fedformer.model import Model as FEDformerModel


class _FEDformerConfig:

    def __init__(
        self,
        enc_in, dec_in, c_out,
        seq_len, label_len, pred_len,
        d_model=512, n_heads=8,
        e_layers=2, d_layers=1,
        d_ff=2048, dropout=0.05,
        activation="gelu", moving_avg=25,
        embed="timeF", freq="h",
        output_attention=0,
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
        self.moving_avg = moving_avg
        self.embed = embed
        self.freq = freq
        self.output_attention = output_attention


class FEDformerWrapper(nn.Module):

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
        dropout: float = 0.05,
        activation: str = "gelu",
        moving_avg: int = 25,
        embed: str = "timeF",
        freq: str = "h",
        version: str = "Fourier",
        mode_select: str = "random",
        modes: int = 32,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.num_variates = num_variates

        cfg = _FEDformerConfig(
            enc_in=num_variates, dec_in=num_variates, c_out=num_variates,
            seq_len=seq_len, label_len=label_len, pred_len=pred_len,
            d_model=d_model, n_heads=n_heads,
            e_layers=e_layers, d_layers=d_layers,
            d_ff=d_ff, dropout=dropout, activation=activation,
            moving_avg=moving_avg, embed=embed, freq=freq,
        )
        self.fedformer = FEDformerModel(
            cfg,
            version=version,
            mode_select=mode_select,
            modes=modes,
        )

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
        out = self.fedformer(x_enc, None, x_dec, None)
        return out
