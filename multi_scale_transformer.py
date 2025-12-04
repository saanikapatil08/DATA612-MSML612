#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Scale Transformer (MST) for Hierarchical Time-Series Forecasting.

Input:
    x:  (B, T, 1)  — batch, time, one feature (e.g., facility CO2e)

Outputs (parallel):
    y_fac: (B, H, 1)  — facility-level forecast
    y_sec: (B, H, 1)  — sector-level forecast
    y_nat: (B, H, 1)  — national-level forecast

Core components:
    - FineScaleEncoder:   high-res stream over original timeline
    - CoarseScaleEncoder: low-res stream via temporal downsampling
    - CrossScaleFusion:   bi-directional cross-attention between streams
    - QueryDecoder:       learnable horizon queries attending over fused memory
    - PredictionHead:     small MLP mapping decoded features → scalar forecast
"""

from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (as in Vaswani et al.)

    Adds position encodings to an input tensor of shape (B, T, d_model).
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)  # not a parameter, but saved with state_dict
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            x + positional_encoding: (B, T, d_model)
        """
        B, T, _ = x.shape
        if T > self.max_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_len {self.max_len} of positional encoding."
            )
        # add encoding (broadcast over batch)
        return x + self.pe[:, :T, :]


# ---------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------
class FineScaleEncoder(nn.Module):
    """
    High-resolution encoder over the original time grid.

    Steps:
        1. Linear projection: 1 → d_model
        2. Sinusoidal positional encoding
        3. TransformerEncoder stack
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int = 5000,
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, T, d_model)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 1)

        Returns:
            features: (B, T, d_model)
        """
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.pos_encoder(x)
        x = self.encoder(x)  # (B, T, d_model)
        return x


class CoarseScaleEncoder(nn.Module):
    """
    Low-resolution encoder over a downsampled time grid.

    Steps:
        1. Average pooling over time with factor f: T → floor(T/f)
        2. Linear projection: 1 → d_model
        3. Sinusoidal positional encoding
        4. TransformerEncoder stack
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        downsample_factor: int,
        max_len: int = 5000,
    ):
        super().__init__()
        if downsample_factor < 1:
            raise ValueError("downsample_factor must be >= 1")

        self.downsample_factor = downsample_factor
        # AvgPool1d expects (B, C, T), so we treat the single feature as C=1
        self.pool = nn.AvgPool1d(
            kernel_size=downsample_factor,
            stride=downsample_factor,
            ceil_mode=False,
        )
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 1)

        Returns:
            features: (B, T_coarse, d_model), where T_coarse = floor(T / f)
        """
        B, T, C = x.shape
        # (B, T, 1) -> (B, 1, T) for pooling
        x_p = x.transpose(1, 2)  # (B, 1, T)
        x_p = self.pool(x_p)  # (B, 1, T_coarse)
        x_down = x_p.transpose(1, 2)  # (B, T_coarse, 1)

        x_down = self.input_proj(x_down)  # (B, T_coarse, d_model)
        x_down = self.pos_encoder(x_down)
        x_down = self.encoder(x_down)  # (B, T_coarse, d_model)
        return x_down


# ---------------------------------------------------------------------
# Cross-scale fusion
# ---------------------------------------------------------------------
class CrossScaleFusion(nn.Module):
    """
    Bi-directional cross-attention between fine and coarse streams.

    - coarse queries fine:  coarse_out = FFN(LN(coarse + Attn(coarse ← fine)))
    - fine   queries coarse: fine_out  = FFN(LN(fine  + Attn(fine  ← coarse)))
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()

        # Coarse queries Fine
        self.mha_c2f = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,  # (B, T, d_model)
        )
        self.norm_c1 = nn.LayerNorm(d_model)
        self.ffn_c = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm_c2 = nn.LayerNorm(d_model)

        # Fine queries Coarse
        self.mha_f2c = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_f1 = nn.LayerNorm(d_model)
        self.ffn_f = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm_f2 = nn.LayerNorm(d_model)

    def forward(
        self,
        fine: torch.Tensor,
        coarse: torch.Tensor,
        fine_key_padding_mask: torch.Tensor = None,
        coarse_key_padding_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fine:   (B, T_fine,   d_model)
            coarse: (B, T_coarse, d_model)
            *_key_padding_mask: optional masks (B, T_*) with True for padded positions.

        Returns:
            fused_fine:   (B, T_fine,   d_model)
            fused_coarse: (B, T_coarse, d_model)
        """

        # --- Coarse queries Fine ---
        # Q = coarse, K = V = fine
        attn_c, _ = self.mha_c2f(
            query=coarse,
            key=fine,
            value=fine,
            key_padding_mask=fine_key_padding_mask,
            need_weights=False,
        )
        coarse_res = self.norm_c1(coarse + attn_c)
        coarse_ffn = self.ffn_c(coarse_res)
        fused_coarse = self.norm_c2(coarse_res + coarse_ffn)

        # --- Fine queries Coarse ---
        # Q = fine, K = V = fused_coarse
        attn_f, _ = self.mha_f2c(
            query=fine,
            key=fused_coarse,
            value=fused_coarse,
            key_padding_mask=coarse_key_padding_mask,
            need_weights=False,
        )
        fine_res = self.norm_f1(fine + attn_f)
        fine_ffn = self.ffn_f(fine_res)
        fused_fine = self.norm_f2(fine_res + fine_ffn)

        return fused_fine, fused_coarse


# ---------------------------------------------------------------------
# Query-based decoder
# ---------------------------------------------------------------------
class QueryDecoder(nn.Module):
    """
    Query-based Transformer decoder.

    - Learnable queries Q ∈ R^(H × d_model)
    - Each batch reuses the same Q, expanded to (B, H, d_model)
    - Decoder attends over memory = concat(fused_fine, fused_coarse)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        forecast_horizon: int,
    ):
        super().__init__()
        self.horizon = int(forecast_horizon)

        # Learnable horizon queries (H, d_model)
        self.query_embed = nn.Parameter(torch.randn(self.horizon, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, H, d_model)
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            memory: (B, T_mem, d_model)
            memory_key_padding_mask: optional (B, T_mem) bool

        Returns:
            decoded: (B, H, d_model)
        """
        B = memory.size(0)
        # Expand learnable queries to batch: (B, H, d_model)
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        decoded = self.decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (B, H, d_model)
        return decoded


# ---------------------------------------------------------------------
# Prediction heads
# ---------------------------------------------------------------------
class PredictionHead(nn.Module):
    """
    Small MLP head for one hierarchy level.

    Maps decoded features D (B, H, d_model) to forecasts (B, H, 1).

    Architecture:
        d_model → d_model/2 → 1 (per step)
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        hidden = max(d_model // 2, 1)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, d_model)

        Returns:
            y: (B, H, 1)
        """
        return self.net(x)


# ---------------------------------------------------------------------
# Top-level MST module
# ---------------------------------------------------------------------
class MultiScaleTransformer(nn.Module):
    """
    Full Multi-Scale Transformer for hierarchical forecasting.

    Steps in forward(x):
        1. Fine-scale encoding
        2. Coarse-scale encoding
        3. Cross-scale fusion (bi-directional)
        4. Concatenate fused streams → memory
        5. Query-based decoding over memory
        6. Three parallel prediction heads (facility / sector / national)
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        downsample_factor: int = 3,
        forecast_horizon: int = 6,
        max_len: int = 5000,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.downsample_factor = downsample_factor
        self.forecast_horizon = forecast_horizon

        # Encoders
        self.fine_encoder = FineScaleEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )
        self.coarse_encoder = CoarseScaleEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            downsample_factor=downsample_factor,
            max_len=max_len,
        )

        # Cross-scale fusion
        self.fusion = CrossScaleFusion(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Query-based decoder
        self.decoder = QueryDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            forecast_horizon=forecast_horizon,
        )

        # Prediction heads (facility / sector / national)
        self.fac_head = PredictionHead(d_model=d_model, dropout=dropout)
        self.sec_head = PredictionHead(d_model=d_model, dropout=dropout)
        self.nat_head = PredictionHead(d_model=d_model, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        fine_key_padding_mask: torch.Tensor = None,
        coarse_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, 1)
            fine_key_padding_mask:   optional (B, T_fine)  bool
            coarse_key_padding_mask: optional (B, T_coarse) bool
            memory_key_padding_mask: optional (B, T_mem)   bool

        Returns:
            y_fac: (B, H, 1)
            y_sec: (B, H, 1)
            y_nat: (B, H, 1)
        """
        # 1) Encode fine and coarse scales
        fine_feat = self.fine_encoder(x)     # (B, T_fine,   d_model)
        coarse_feat = self.coarse_encoder(x) # (B, T_coarse, d_model)

        # 2) Bi-directional cross-scale fusion
        fused_fine, fused_coarse = self.fusion(
            fine=fine_feat,
            coarse=coarse_feat,
            fine_key_padding_mask=fine_key_padding_mask,
            coarse_key_padding_mask=coarse_key_padding_mask,
        )

        # 3) Concatenate fused streams along time to form memory
        memory = torch.cat([fused_fine, fused_coarse], dim=1)  # (B, T_fine + T_coarse, d_model)

        # 4) Query-based decoding
        decoded = self.decoder(
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (B, H, d_model)

        # 5) Parallel prediction heads
        y_fac = self.fac_head(decoded)  # (B, H, 1)
        y_sec = self.sec_head(decoded)  # (B, H, 1)
        y_nat = self.nat_head(decoded)  # (B, H, 1)

        y_fac = self.fac_head(decoded)
        y_sec = self.sec_head(decoded)
        y_nat = self.nat_head(decoded)

        # Optional safety clamp in scaled space
        y_fac = torch.clamp(y_fac, -1e3, 1e3)
        y_sec = torch.clamp(y_sec, -1e3, 1e3)
        y_nat = torch.clamp(y_nat, -1e3, 1e3)

        return y_fac, y_sec, y_nat

