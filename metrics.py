#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics and loss functions for Multi-Scale Transformer (MST).

Expected shapes:
    y_* and y_*_true: (B, H, 1) or (B, H)

This module provides:
    - mse_loss
    - mae
    - rmse
    - consistency_loss (hierarchical consistency term)
    - compute_all_metrics (per-head MAE/RMSE + consistency gap)
"""

from typing import Dict

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Basic losses / metrics
# ---------------------------------------------------------------------
def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error.

    Args:
        pred:   (B, H, 1) or (B, H)
        target: (B, H, 1) or (B, H)

    Returns:
        scalar tensor (MSE)
    """
    # Ensure same shape and float
    pred = pred.float()
    target = target.float()
    return F.mse_loss(pred, target)


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean absolute error.
    """
    pred = pred.float()
    target = target.float()
    return (pred - target).abs().mean()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Root mean squared error.
    """
    return mse_loss(pred, target).sqrt()


# ---------------------------------------------------------------------
# Hierarchical consistency
# ---------------------------------------------------------------------
def consistency_loss(
    y_fac: torch.Tensor,
    y_sec: torch.Tensor,
    y_nat: torch.Tensor,
) -> torch.Tensor:
    """
    Hierarchical consistency penalty.

    As described in your spec:
        L_cons = |mean(ŷ_fac) − mean(ŷ_sec)| + |mean(ŷ_sec) − mean(ŷ_nat)|

    Means are taken over batch and horizon dimensions.

    Args:
        y_fac, y_sec, y_nat: (B, H, 1) or (B, H)

    Returns:
        scalar tensor
    """
    y_fac = y_fac.float()
    y_sec = y_sec.float()
    y_nat = y_nat.float()

    # Flatten over all non-batch dims
    m_fac = y_fac.mean()
    m_sec = y_sec.mean()
    m_nat = y_nat.mean()

    return (m_fac - m_sec).abs() + (m_sec - m_nat).abs()


# ---------------------------------------------------------------------
# Aggregated metrics for logging
# ---------------------------------------------------------------------
def compute_all_metrics(
    y_fac_pred: torch.Tensor,
    y_sec_pred: torch.Tensor,
    y_nat_pred: torch.Tensor,
    y_fac_true: torch.Tensor,
    y_sec_true: torch.Tensor,
    y_nat_true: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-head MAE/RMSE and consistency gap for logging.

    Returns a dict of scalar tensors:
        {
            "fac_mae": ...,
            "sec_mae": ...,
            "nat_mae": ...,
            "fac_rmse": ...,
            "sec_rmse": ...,
            "nat_rmse": ...,
            "consistency_gap": ...,
        }

    The consistency gap here is the same expression as L_cons but
    treated as a diagnostic metric (not weighted).
    """
    # Ensure float
    y_fac_pred = y_fac_pred.float()
    y_sec_pred = y_sec_pred.float()
    y_nat_pred = y_nat_pred.float()
    y_fac_true = y_fac_true.float()
    y_sec_true = y_sec_true.float()
    y_nat_true = y_nat_true.float()

    fac_mae = mae(y_fac_pred, y_fac_true)
    sec_mae = mae(y_sec_pred, y_sec_true)
    nat_mae = mae(y_nat_pred, y_nat_true)

    fac_rmse = rmse(y_fac_pred, y_fac_true)
    sec_rmse = rmse(y_sec_pred, y_sec_true)
    nat_rmse = rmse(y_nat_pred, y_nat_true)

    cons_gap = consistency_loss(y_fac_pred, y_sec_pred, y_nat_pred)

    return {
        "fac_mae": fac_mae,
        "sec_mae": sec_mae,
        "nat_mae": nat_mae,
        "fac_rmse": fac_rmse,
        "sec_rmse": sec_rmse,
        "nat_rmse": nat_rmse,
        "consistency_gap": cons_gap,
    }
