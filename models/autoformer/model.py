# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn

from ..transformer.embed import DataEmbedding_wo_pos
from .auto_correlation import AutoCorrelation, AutoCorrelationLayer
from .enc_dec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Autoformer(nn.Module):
    """
    Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
    Paper: https://openreview.net/pdf?id=I55UqU-M11y (NeurIPS 2021, Wu et al.)
    """

    def __init__(
        self,
        enc_in: int,
        dec_in: int,
        c_out: int,
        seq_len: int,
        label_len: int,
        pred_len: int,
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

        self.decomp = series_decomp(moving_avg)

        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding_wo_pos(dec_in, d_model, embed, freq, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            mask_flag=False,
                            factor=factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model),
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            mask_flag=True,
                            factor=factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            mask_flag=False,
                            factor=factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True),
        )

    def forecast(self, x_enc, x_mark_enc, x_mark_dec):
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(
            [x_enc.shape[0], self.pred_len, x_enc.shape[2]],
            device=x_enc.device, dtype=x_enc.dtype,
        )
        seasonal_init, trend_init = self.decomp(x_enc)

        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        # 3) Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init
        )

        return trend_part + seasonal_part

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dec_out = self.forecast(x_enc, x_mark_enc, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]
