#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch Lightning module for Multi-Scale Transformer (MST).

This wraps:
  - multi_scale_transformer.MultiScaleTransformer
  - metrics (MSE, MAE, RMSE, consistency loss)

Expected batch format from DataLoader:
    batch = {
        "x":     (B, T, 1),
        "y_fac": (B, H, 1),
        "y_sec": (B, H, 1),
        "y_nat": (B, H, 1),
    }

Logged metrics:
  - train/val/test per-head MSE, MAE, RMSE
  - consistency gap
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from multi_scale_transformer import MultiScaleTransformer
from metrics import mse_loss, consistency_loss, compute_all_metrics


class MSTLightningModule(pl.LightningModule):
    """
    Lightning wrapper for the Multi-Scale Transformer.

    Loss:
        L = w_fac * L_fac + w_sec * L_sec + w_nat * L_nat + w_cons * L_cons

    where:
        L_fac = MSE(y_fac_pred, y_fac_true)
        L_sec = MSE(y_sec_pred, y_sec_true)
        L_nat = MSE(y_nat_pred, y_nat_true)
        L_cons = |mean(y_fac_pred) - mean(y_sec_pred)|
                 + |mean(y_sec_pred) - mean(y_nat_pred)|
    """

    def __init__(
        self,
        # MST architecture hyperparams
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        downsample_factor: int = 3,
        forecast_horizon: int = 6,
        max_len: int = 5000,
        # Optimization
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-2,
        step_lr_gamma: float = 0.95,
        step_lr_step_size: int = 10,
        # Loss weights
        w_fac: float = 1.0,
        w_sec: float = 0.5,
        w_nat: float = 0.5,
        w_cons: float = 0.3,
    ):
        super().__init__()
        # This saves all arguments to hparams for checkpointing / logging
        self.save_hyperparameters()

        # Underlying MST model
        self.mst = MultiScaleTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            downsample_factor=downsample_factor,
            forecast_horizon=forecast_horizon,
            max_len=max_len,
        )

    # ----------------- forward ----------------- #
    def forward(self, x: torch.Tensor) -> Any:
        """
        Forward pass through MST.

        Args:
            x: (B, T, 1)

        Returns:
            y_fac, y_sec, y_nat: each (B, H, 1)
        """
        return self.mst(x)

    # ----------------- shared step ----------------- #
    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """
        Shared logic for train/val/test steps.

        Args:
            batch: dict with keys "x", "y_fac", "y_sec", "y_nat"
            stage: "train", "val", or "test"

        Returns:
            total_loss: scalar tensor
        """
        x = batch["x"]          # (B, T, 1)
        y_fac = batch["y_fac"]  # (B, H, 1)
        y_sec = batch["y_sec"]
        y_nat = batch["y_nat"]

        # --------- sanity checks on inputs ----------
        for name, t in [("x", x), ("y_fac", y_fac), ("y_sec", y_sec), ("y_nat", y_nat)]:
            if not torch.isfinite(t).all():
                print(f"[{stage}] Non-finite values detected in {name}")
                print(f"{name} min={t.min().item()}, max={t.max().item()}")
                raise RuntimeError(f"Non-finite values in {name}")

        # Forward pass
        y_fac_pred, y_sec_pred, y_nat_pred = self(x)

        # --------- sanity checks on predictions ----------
        for name, t in [("y_fac_pred", y_fac_pred),
                        ("y_sec_pred", y_sec_pred),
                        ("y_nat_pred", y_nat_pred)]:
            if not torch.isfinite(t).all():
                print(f"[{stage}] Non-finite values detected in {name}")
                print(f"{name} min={t.min().item()}, max={t.max().item()}")
                raise RuntimeError(f"Non-finite values in {name}")

        # Per-head MSE losses
        L_fac = mse_loss(y_fac_pred, y_fac)
        L_sec = mse_loss(y_sec_pred, y_sec)
        L_nat = mse_loss(y_nat_pred, y_nat)

        # Hierarchical consistency
        L_cons = consistency_loss(y_fac_pred, y_sec_pred, y_nat_pred)

        # Total weighted loss
        loss = (
            self.hparams.w_fac * L_fac
            + self.hparams.w_sec * L_sec
            + self.hparams.w_nat * L_nat
            + self.hparams.w_cons * L_cons
        )

        if not torch.isfinite(loss):
            print(f"[{stage}] Loss became non-finite.")
            print(f"  L_fac={L_fac.item()}, L_sec={L_sec.item()}, L_nat={L_nat.item()}, L_cons={L_cons.item()}")
            raise RuntimeError("Non-finite loss encountered")

        # Metrics for logging
        mets = compute_all_metrics(
            y_fac_pred=y_fac_pred,
            y_sec_pred=y_sec_pred,
            y_nat_pred=y_nat_pred,
            y_fac_true=y_fac,
            y_sec_true=y_sec,
            y_nat_true=y_nat,
        )

        # Log everything with stage prefix
        on_step = stage == "train"
        on_epoch = True

        self.log(f"{stage}_loss", loss, on_step=on_step, on_epoch=on_epoch, prog_bar=True)
        self.log(f"{stage}_L_fac", L_fac, on_step=on_step, on_epoch=on_epoch)
        self.log(f"{stage}_L_sec", L_sec, on_step=on_step, on_epoch=on_epoch)
        self.log(f"{stage}_L_nat", L_nat, on_step=on_step, on_epoch=on_epoch)
        self.log(f"{stage}_L_cons", L_cons, on_step=on_step, on_epoch=on_epoch)

        self.log(f"{stage}_fac_mae", mets["fac_mae"], on_step=on_step, on_epoch=on_epoch)
        self.log(f"{stage}_sec_mae", mets["sec_mae"], on_step=on_step, on_epoch=on_epoch)
        self.log(f"{stage}_nat_mae", mets["nat_mae"], on_step=on_step, on_epoch=on_epoch)

        self.log(f"{stage}_fac_rmse", mets["fac_rmse"], on_step=on_step, on_epoch=on_epoch)
        self.log(f"{stage}_sec_rmse", mets["sec_rmse"], on_step=on_step, on_epoch=on_epoch)
        self.log(f"{stage}_nat_rmse", mets["nat_rmse"], on_step=on_step, on_epoch=on_epoch)

        self.log(
            f"{stage}_consistency_gap",
            mets["consistency_gap"],
            on_step=on_step,
            on_epoch=on_epoch,
        )

        return loss

    # ----------------- training / validation / test ----------------- #
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="test")

    # ----------------- optimizers ----------------- #
    def configure_optimizers(self):
        """
        AdamW + StepLR scheduler.
        """
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.step_lr_step_size,
            gamma=self.hparams.step_lr_gamma,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
