# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn

from .lstm import LSTM


class LSTMWrapper(nn.Module):

    def __init__(
        self,
        num_variates: int = 5,
        seq_len: int = 168,
        label_len: int = 48,
        pred_len: int = 4,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.num_variates = num_variates

        self.lstm = LSTM(
            enc_in=num_variates,
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.lstm(x_enc)
