# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn

from .autoformer import Autoformer


class AutoformerWrapper(nn.Module):

    def __init__(
        self,
        num_variates: int = 17,
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
        factor: int = 1,
        moving_avg: int = 25,
        embed: str = "timeF",
        freq: str = "h",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.num_variates = num_variates

        self.autoformer = Autoformer(
            enc_in=num_variates,
            dec_in=num_variates,
            c_out=num_variates,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            factor=factor,
            moving_avg=moving_avg,
            embed=embed,
            freq=freq,
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_mark_dec: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_enc       : [B, L, N]
        x_mark_enc  : [B, L, n_freq]
        x_mark_dec  : [B, label_len + pred_len, n_freq]
        return      : [B, pred_len, N]
        """
        B, L, N = x_enc.shape

        dec_zeros = torch.zeros(
            B, self.pred_len, N, device=x_enc.device, dtype=x_enc.dtype
        )
        x_dec = torch.cat([x_enc[:, -self.label_len:, :], dec_zeros], dim=1)

        return self.autoformer(x_enc, None, x_dec, None)
