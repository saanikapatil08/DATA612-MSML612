#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for the Multi-Scale Transformer (MST).

Runs on GPU automatically if available.
Uses PyTorch Lightning Trainer.
"""

import os
import yaml
import torch
import pytorch_lightning as pl

from lightning_module import MSTLightningModule
from dataloader import create_dataloaders

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


# ---------------------------------------------------------
# Load YAML config
# ---------------------------------------------------------
def load_config(path: str = "configs/mst.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------
# Trainer entrypoint
# ---------------------------------------------------------
def main():

    # 1. Load config
    cfg = load_config("configs/mst.yaml")

    # 2. Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        context_length=cfg["data"]["context_length"],
        forecast_horizon=cfg["data"]["forecast_horizon"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 0),
    )

    # 3. Instantiate model
    model = MSTLightningModule(
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_encoder_layers=cfg["model"]["num_encoder_layers"],
        num_decoder_layers=cfg["model"]["num_decoder_layers"],
        dim_feedforward=cfg["model"]["dim_feedforward"],
        dropout=cfg["model"]["dropout"],
        downsample_factor=cfg["model"]["downsample_factor"],
        # 👇 IMPORTANT: use data.forecast_horizon so it matches dataloader
        forecast_horizon=cfg["data"]["forecast_horizon"],
        learning_rate=cfg["model"]["learning_rate"],
        weight_decay=cfg["model"].get("weight_decay", 1e-2),
        step_lr_gamma=cfg["model"]["step_lr_gamma"],
        step_lr_step_size=cfg["model"].get("step_lr_step_size", 10),
        w_fac=cfg["loss"]["w_fac"],
        w_sec=cfg["loss"]["w_sec"],
        w_nat=cfg["loss"]["w_nat"],
        w_cons=cfg["loss"]["w_cons"],
    )

    # 4. Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=None,                # let PL put it under logger.log_dir / "checkpoints"
        filename="mst-{epoch:02d}-{train_loss:.4e}",
        save_top_k=1,
        save_last=True,
        monitor="train_loss",        # IMPORTANT: we know val_loss doesn't exist right now
        mode="min",
        )

    early_stop_cb = EarlyStopping(
        monitor="train_loss",   # was "val_loss"
        mode="min",
        patience=5,
        check_on_train_epoch_end=True,  # optional but nice
    )

    logger = TensorBoardLogger("logs", name="mst")

    # ---------------------------------------------------------
    # 5. GPU settings (auto-detect)
    # ---------------------------------------------------------
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
        print("🔥 Using GPU:", torch.cuda.get_device_name(0))
    else:
        accelerator = "cpu"
        devices = 1
        print("⚠️ CUDA not available — training on CPU.")

    # 6. Trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=cfg["trainer"]["max_epochs"],
        log_every_n_steps=cfg["trainer"].get("log_every_n_steps", 20),
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        gradient_clip_val=cfg["trainer"].get("gradient_clip_val", 1.0),
    )

    # 7. Fit
    trainer.fit(model, train_loader, val_loader)

    # 8. Test best checkpoint
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
