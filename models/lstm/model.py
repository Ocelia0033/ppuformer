# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        outputs: [B, L, H]
        return:  [B, H]
        """
        scores = self.attn(outputs)                # [B, L, 1]
        weights = F.softmax(scores, dim=1)         # [B, L, 1]
        context = (outputs * weights).sum(dim=1)   # [B, H]
        return context


class LSTM(nn.Module):

    def __init__(
        self,
        enc_in: int,
        seq_len: int,
        pred_len: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=enc_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        num_directions = 2 if bidirectional else 1
        lstm_out_dim = hidden_size * num_directions

        self.attention = TemporalAttention(lstm_out_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_out_dim, pred_len * enc_in)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        x_enc : [B, L, N]
        return: [B, pred_len, N]
        """
        B = x_enc.shape[0]

        outputs, _ = self.lstm(x_enc)               # [B, L, H*num_directions]

        context = self.attention(outputs)           # [B, H*num_directions]
        context = self.dropout(context)

        flat = self.head(context)                   # [B, pred_len * N]
        pred = flat.view(B, self.pred_len, self.enc_in)

        return pred
