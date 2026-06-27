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
      residual = LayerNorm(PWConv(D)^T)
      X^(3)  = X^(2) + γ_eff * residual
      γ_eff  = γ_bound * tanh(γ_raw)

    PPU：γ_raw 严格零初始化 → 初始 out = X^(2)；
    PWConv 极小随机初始化 → γ 在 step0 即可获得梯度；
    LayerNorm + tanh-bound γ 防止 residual 爆炸。
    """

    def __init__(self, num_variates: int, kernels=(3, 5, 7), dropout: float = 0.0,
                 use_ppu: bool = True):
        super().__init__()
        self.num_variates = num_variates
        self.kernels = tuple(kernels)
        self.use_ppu = use_ppu
        self.gamma_bound = 0.1

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

        residual = self.pw(d_cat).transpose(1, 2)            # [B, L, N]
        residual = self.res_norm(residual)
        residual = self.dropout(residual)

        if self.use_ppu:
            gamma_eff = self.gamma_bound * torch.tanh(self.gamma)
        else:
            gamma_eff = self.gamma

        out = x_btn + gamma_eff * residual
        return out
