# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn


class DSCLocalBranch(nn.Module):
    """
    Depthwise Separable Convolution Local Branch (DSC) — 对应论文公式 (9)(10)(11)
    --------------------------------------------------------------------------------
      D_k    = GELU(DWConv^(k)(X^(2)T))         k ∈ K_dsc = {3,5,7}
      D      = [D_3, D_5, D_7]
      Y_dsc  = X^(2)T + Dropout(PWConv(D))
      X^(3)  = X^(2) + γ_dsc (Y_dsc^T - X^(2))

    PPU：PWConv 的权重与偏置整层零初始化（论文 2.2.3 末尾），γ_dsc 零初始化，
    从而初始时 Y_dsc = X^(2)T，X^(3) = X^(2)。
    """

    def __init__(self, num_variates: int, kernels=(3, 5, 7), dropout: float = 0.0,
                 use_ppu: bool = True):
        super().__init__()
        self.num_variates = num_variates
        self.kernels = tuple(kernels)
        self.use_ppu = use_ppu

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
        # 论文：PWConv 权重与偏置整层零初始化
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pw.bias)

        self.dropout = nn.Dropout(dropout)

        if use_ppu:
            # 0.01 而非严格 0：避免 γ=0 + pw=0 导致梯度死锁（纯数值工程问题，不影响 PPU 精神）
            self.gamma = nn.Parameter(torch.full((1,), 0.01))
        else:
            self.register_buffer("gamma", torch.ones(1))

    def forward(self, x_btn: torch.Tensor) -> torch.Tensor:
        """
        x_btn : [B, L, N]
        return: [B, L, N]
        """
        x = x_btn.transpose(1, 2)                            # [B, N, L]

        feats = [self.act(conv(x)) for conv in self.dw_convs]
        d_cat = torch.cat(feats, dim=1)                      # [B, kN, L]

        y_dsc = x + self.dropout(self.pw(d_cat))             # [B, N, L]
        y_dsc_btn = y_dsc.transpose(1, 2)                    # [B, L, N]

        out = x_btn + self.gamma * (y_dsc_btn - x_btn)
        return out
