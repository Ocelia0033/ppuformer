# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn


class DSCLocalBranch(nn.Module):
    """
    Depthwise Separable Convolution Local Branch (DSC)
    弱局部修正：γ_eff * channel_scale * residual，初始 out = x。
    """

    def __init__(self, num_variates: int, kernels=(3, 5, 7), dropout: float = 0.0,
                 use_ppu: bool = True, gamma_bound: float = 0.03):
        super().__init__()
        self.num_variates = num_variates
        self.kernels = tuple(kernels)
        self.use_ppu = use_ppu
        self.gamma_bound = gamma_bound

        self.dw_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=num_variates,
                out_channels=num_variates,
                kernel_size=k,
                padding=k // 2,
                groups=num_variates,
            )
            for k in self.kernels
        ])
        self.act = nn.GELU()

        self.pw = nn.Conv1d(
            in_channels=num_variates * len(self.kernels),
            out_channels=num_variates,
            kernel_size=1,
        )
        nn.init.normal_(self.pw.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.pw.bias)

        self.res_norm = nn.LayerNorm(num_variates)
        self.dropout = nn.Dropout(dropout)

        if use_ppu:
            self.gamma = nn.Parameter(torch.zeros(1))
            self.channel_gate = nn.Parameter(torch.zeros(num_variates))
        else:
            self.register_buffer("gamma", torch.ones(1))
            self.register_buffer("channel_gate", torch.ones(num_variates))

    def forward(self, x_btn: torch.Tensor) -> torch.Tensor:
        x = x_btn.transpose(1, 2)

        feats = [self.act(conv(x)) for conv in self.dw_convs]
        d_cat = torch.cat(feats, dim=1)

        residual = self.pw(d_cat).transpose(1, 2)
        residual = self.res_norm(residual)
        residual = self.dropout(residual)

        if self.use_ppu:
            gamma_eff = self.gamma_bound * torch.tanh(self.gamma)
        else:
            gamma_eff = self.gamma

        channel_scale = torch.sigmoid(self.channel_gate).view(1, 1, -1)
        out = x_btn + gamma_eff * channel_scale * residual
        return out
