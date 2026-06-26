# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn


class PhysBiasMLP(nn.Module):
    """
    Physics Bias MLP — 对应论文公式 (21)
        PhysBias_l(φ) = W_l^(2) GELU(W_l^(1) φ + b_l^(1)) + b_l^(2)

    PPU：use_ppu=True 时，末层 W_l^(2) 与 b_l^(2) 整层零初始化，使 PhysBias_l(φ) = 0
    从而 PGIA 起步时退化为标准反转注意力（论文 2.4 / 2.5）。
    use_ppu=False 时（"无 PPU"对照），末层用小随机初始化，让物理偏置从训练开始即非零。
    """

    def __init__(self, num_variates: int, hidden_dim: int = 32,
                 in_dim: int = 4, use_ppu: bool = True, init_std: float = 1e-3):
        super().__init__()
        self.num_variates = num_variates

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, num_variates * num_variates)

        if use_ppu:
            # 论文：末层权重与偏置整层零初始化（PGIA 的 PPU 实现方式）
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        else:
            # 无 PPU 对照：物理偏置从训练开始就生效
            nn.init.normal_(self.fc2.weight, mean=0.0, std=init_std)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """
        phi: [B, in_dim]
        return bias: [B, N, N]
        """
        h = self.act(self.fc1(phi))                     # [B, hidden_dim]
        out = self.fc2(h)                               # [B, N*N]
        return out.view(-1, self.num_variates, self.num_variates)
