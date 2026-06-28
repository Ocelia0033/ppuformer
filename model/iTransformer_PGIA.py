# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn

from iTransformer import iTransformer

from layers.dsc_local import DSCLocalBranch


# =============================================================================
# =============================================================================

class PhysicsStateGate(nn.Module):
    """
    Physics State Gate (PSG)
    ========================
    根据物理状态向量对 **全部通道** 生成自适应门控权重。
    不同大气状态下各输入变量的信噪比不同，PSG 动态调节各通道的
    贡献权重，使 backbone 在输入端就获得物理感知的特征增强。

      φ = x[:,:, phys_start:phys_end]          物理特征子集
      c = [mean(φ,dim=1); std(φ,dim=1)]        时序统计量
      S = σ(W_2 GELU(W_1 c + b_1) + b_2)      ∈ (0, 1)^N  全通道门
      X^(1) = X^(0) + γ_psg * (S · X^(0) - X^(0))

    PPU：b_2 初始化为 +2 使 S ≈ σ(2) ≈ 0.88（接近 1，门近似全开），
         γ_psg 零初始化，使 X^(1) ≈ X^(0)（起步不扰动 backbone）。
    """

    def __init__(self,
                 num_variates: int = 17,
                 phys_idx_range=(5, 17),
                 hidden_dim: int = 32,
                 use_ppu: bool = True):
        super().__init__()
        self.num_variates = num_variates
        self.phys_start, self.phys_end = phys_idx_range
        n_phys = self.phys_end - self.phys_start

        self.fc1 = nn.Linear(n_phys * 2, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, num_variates)

        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 2.0)

        if use_ppu:
            self.gamma = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("gamma", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = x[:, :, self.phys_start:self.phys_end]          # [B, L, n_phys]
        mu = phi.mean(dim=1)                                   # [B, n_phys]
        sigma = phi.std(dim=1, unbiased=False)                 # [B, n_phys]
        c = torch.cat([mu, sigma], dim=-1)                     # [B, 2*n_phys]

        S = torch.sigmoid(self.fc2(self.act(self.fc1(c))))     # [B, N]
        S = S.unsqueeze(1)                                     # [B, 1, N] 广播到时间维

        return x + self.gamma * (S * x - x)


# =============================================================================
# =============================================================================

# =============================================================================
# =============================================================================

class WeatherAwareSpectralEnhancement(nn.Module):

    def __init__(self,
                 num_variates: int = 17,
                 lookback_len: int = 168,
                 hidden_dim: int = 64,
                 use_ppu: bool = True):
        super().__init__()
        self.num_variates = num_variates
        self.lookback_len = lookback_len
        self.use_ppu = use_ppu
        self.K = lookback_len // 2 + 1

        N = num_variates
        K = self.K

        c_dim = 3 * N

        self.mlp_c = nn.Sequential(
            nn.Linear(c_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, N * K),
        )
        self.mlp_r = nn.Sequential(
            nn.Linear(c_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, N),
        )
        nn.init.zeros_(self.mlp_r[-1].weight)
        nn.init.constant_(self.mlp_r[-1].bias, 3.0)

        self.W_base = nn.Parameter(torch.ones(N, K))

        if use_ppu:
            self.gamma = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("gamma", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, L, N]
        return: [B, L, N]
        """
        B, L, N = x.shape
        K = self.K

        x_bnl = x.transpose(1, 2).contiguous()              # [B, N, L]
        Z = torch.fft.rfft(x_bnl, dim=-1)

        mu_t = x.mean(dim=1)                                 # [B, N]
        std_t = x.std(dim=1, unbiased=False)                 # [B, N]
        x_last = x[:, -1, :]                                 # [B, N]
        c = torch.cat([mu_t, std_t, x_last], dim=-1)         # [B, 3N]

        A = torch.sigmoid(self.mlp_c(c)).view(B, N, K)
        r = torch.sigmoid(self.mlp_r(c))                     # [B, N]

        W = self.W_base.unsqueeze(0) * A                     # [B, N, K]
        Z_filt = Z * W

        x_filt = torch.fft.irfft(Z_filt, n=L, dim=-1)        # [B, N, L]
        x_filt = x_filt.transpose(1, 2)                      # [B, L, N]

        r_btn = r.unsqueeze(1)                               # [B, 1, N]
        x_f = r_btn * x + (1.0 - r_btn) * x_filt             # [B, L, N]

        return x + self.gamma * (x_f - x)


# =============================================================================
# =============================================================================

class iTransformerPGIA(nn.Module):

    def __init__(self,
                 num_variates: int = 17,
                 lookback_len: int = 168,
                 pred_length: int = 24,
                 target_idx: int = 4,
                 cond_idx=(5, 9, 11, 16),
                 dim: int = 128,
                 depth: int = 5,
                 heads: int = 1,
                 dim_head: int = 32,
                 num_tokens_per_variate: int = 1,
                 use_reversible_instance_norm: bool = True,
                 flash_attn: bool = True,
                 attn_dropout: float = 0.,
                 ff_dropout: float = 0.,
                 phys_hidden_dim: int = 32,
                 psg_hidden_dim: int = 32,
                 wase_hidden_dim: int = 64,
                 dsc_kernels=(3, 5, 7),
                 dsc_dropout: float = 0.0,
                 dsc_gamma_bound: float = 0.03,
                 use_psg: bool = True,
                 use_wase: bool = True,
                 use_dsc: bool = True,
                 use_pgia: bool = True,
                 use_ppu: bool = True):
        super().__init__()
        self.num_variates = num_variates
        self.lookback_len = lookback_len
        self.pred_length = pred_length
        self.target_idx = target_idx
        self.cond_idx = list(cond_idx)

        self.use_psg = use_psg
        self.use_wase = use_wase
        self.use_dsc = use_dsc
        self.use_pgia = use_pgia
        self.use_ppu = use_ppu

        self.psg = PhysicsStateGate(
            num_variates=num_variates,
            hidden_dim=psg_hidden_dim,
            use_ppu=use_ppu,
        ) if use_psg else None
        self.wase = WeatherAwareSpectralEnhancement(
            num_variates=num_variates,
            lookback_len=lookback_len,
            hidden_dim=wase_hidden_dim,
            use_ppu=use_ppu,
        ) if use_wase else None
        self.dsc = DSCLocalBranch(
            num_variates=num_variates,
            kernels=dsc_kernels,
            dropout=dsc_dropout,
            use_ppu=use_ppu,
            gamma_bound=dsc_gamma_bound,
        ) if use_dsc else None

        self.backbone = iTransformer(
            num_variates=num_variates,
            lookback_len=lookback_len,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            pred_length=pred_length,
            num_tokens_per_variate=num_tokens_per_variate,
            use_reversible_instance_norm=use_reversible_instance_norm,
            flash_attn=flash_attn,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            use_pgia=use_pgia,
            phys_in_dim=len(self.cond_idx),
            phys_hidden_dim=phys_hidden_dim,
            use_ppu=use_ppu,
        )

    def _build_phi(self, x_raw: torch.Tensor) -> torch.Tensor:
        cond = x_raw[:, :, self.cond_idx]                    # [B, L, 4]
        return cond.mean(dim=1)                              # [B, 4]

    def forward(self, x: torch.Tensor, return_target_only: bool = False):
        phi = self._build_phi(x) if self.use_pgia else None

        h = x
        if self.use_psg:
            h = self.psg(h)
        if self.use_wase:
            h = self.wase(h)
        if self.use_dsc:
            h = self.dsc(h)

        out = self.backbone(h, phi=phi)                      # [B, H, N]

        if return_target_only:
            return out[:, :, self.target_idx]                # [B, H]
        return out
