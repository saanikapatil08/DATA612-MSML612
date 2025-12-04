#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference & plotting for Multi-Scale Transformer (MST).

Capabilities:
    1) Load a trained MSTLightningModule checkpoint.
    2) Run inference on a custom 1D time-series.
    3) Run inference for a specific facility from facility_series.pkl.
    4) Plot facility, sector, and national forecasts vs actuals.

IMPORTANT:
    This script assumes you used a scaling factor of SCALE = 1e4
    in your dataloader (dataloader.py) when building x_fac / y_fac / y_nat.
    If you change SCALE there, change it here as well.
"""

import os
from pathlib import Path
from typing import List, Union, Optional, Dict

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

from lightning_module import MSTLightningModule

# ----------------------------------------------------------------------
# Paths / config
# ----------------------------------------------------------------------
CONFIG_PATH = "configs/mst.yaml"

# Folder where Lightning saved checkpoints (from train_mst.py / ModelCheckpoint)
CHECKPOINT_DIR = Path("logs/mst")   # matches TensorBoardLogger("logs", name="mst")
CHECKPOINT_PATH: Optional[str] = None

# Processed EPA series (from data_processing.py)
OUT_EPA_DIR = Path("out_epa/output")
FACILITY_SERIES_PATH = OUT_EPA_DIR / "facility_series.pkl"
SECTOR_SERIES_PATH   = OUT_EPA_DIR / "sector_series.pkl"
NATIONAL_SERIES_PATH = OUT_EPA_DIR / "national_series.pkl"

# Must match the scale in dataloader.py
SCALE = 1e4


# ----------------------------------------------------------------------
# Helper: load config & checkpoint
# ----------------------------------------------------------------------
def load_config(path: str = CONFIG_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_mean_facility_series(facility_series_dict: dict):
    """
    Build a mean facility series over ALL facilities.

    Returns:
        pd.Series indexed by year, values = mean CO2e across facilities.
    """
    import pandas as pd

    if not facility_series_dict:
        raise ValueError("facility_series_dict is empty, cannot build mean series.")

    # Align all facility series by year and take column-wise mean
    df = pd.concat(
        [s.sort_index() for s in facility_series_dict.values()],
        axis=1
    )
    mean_fac = df.mean(axis=1)
    mean_fac.name = "mean_facility"
    return mean_fac


def build_mean_sector_series(sector_series_dict: dict):
    """
    Build a mean sector series over ALL sectors.

    Returns:
        pd.Series indexed by year, values = mean CO2e across sectors.
    """
    import pandas as pd

    if not sector_series_dict:
        return None  # no sector data available

    df = pd.concat(
        [s.sort_index() for s in sector_series_dict.values()],
        axis=1
    )
    mean_sec = df.mean(axis=1)
    mean_sec.name = "mean_sector"
    return mean_sec


def plot_mean_hierarchy_last_window(
    model: MSTLightningModule,
    context_length: int,
    forecast_horizon: int,
    facility_series_dict: dict,
    sector_series_dict: Optional[dict] = None,
    national_series: Optional["pd.Series"] = None,
):
    """
    Plot forecasts vs actuals for:
        - Mean facility over all facilities
        - Mean sector over all sectors (if available)
        - National series

    Uses the last (T + H) window of the mean facility series as context + target.
    """
    import pandas as pd  # local import

    if national_series is None:
        national_series = load_national_series()
    if not isinstance(national_series, pd.Series):
        raise TypeError("national_series must be a pandas Series.")

    # Build mean series
    mean_fac = build_mean_facility_series(facility_series_dict)          # pd.Series
    mean_sec = build_mean_sector_series(sector_series_dict or {})        # pd.Series or None

    mean_fac = mean_fac.sort_index()
    years = mean_fac.index.to_numpy()
    values = mean_fac.to_numpy(dtype=float)

    T = context_length
    H = forecast_horizon

    if len(values) < T + H:
        raise ValueError(
            f"Mean facility series has only {len(values)} points, "
            f"but need at least T+H = {T + H}."
        )

    # Last T as context, last H as true future
    ctx_values = values[-(T + H):-H]    # (T,)
    y_fac_true = values[-H:]            # (H,)
    years_future = years[-H:]

    # Run model forecast with mean facility context
    preds = predict_from_series(
        model,
        ctx_values,
        context_length=T,
        forecast_horizon=H,
    )
    y_fac_pred = preds["y_fac"]
    y_sec_pred = preds["y_sec"]
    y_nat_pred = preds["y_nat"]

    # National actual
    nat_series = national_series.sort_index()
    y_nat_true = nat_series.reindex(years_future).to_numpy(dtype=float)

    # Mean sector actual
    if mean_sec is not None:
        mean_sec = mean_sec.sort_index()
        y_sec_true = mean_sec.reindex(years_future).to_numpy(dtype=float)
    else:
        y_sec_true = None

    steps = np.arange(1, H + 1)

    plt.figure(figsize=(12, 8))

    # 1) Mean Facility
    plt.subplot(3, 1, 1)
    plt.plot(steps, y_fac_true, marker="o", label="Actual mean facility")
    plt.plot(steps, y_fac_pred, marker="x", linestyle="--", label="Pred mean facility")
    plt.title("Mean Facility (over all facilities): forecast vs actual")
    plt.ylabel("CO₂e")
    plt.legend()
    plt.grid(True)

    # 2) National
    plt.subplot(3, 1, 2)
    plt.plot(steps, y_nat_true, marker="o", label="Actual national")
    plt.plot(steps, y_nat_pred, marker="x", linestyle="--", label="Pred national")
    plt.title("National: forecast vs actual")
    plt.ylabel("CO₂e")
    plt.legend()
    plt.grid(True)

    # 3) Mean Sector
    plt.subplot(3, 1, 3)
    if y_sec_true is not None:
        plt.plot(steps, y_sec_true, marker="o", label="Actual mean sector")
    else:
        plt.plot([], [], label="No sector actual (missing sector series)")
    plt.plot(steps, y_sec_pred, marker="x", linestyle="--", label="Pred mean sector")
    plt.title("Mean Sector (over all sectors): forecast vs actual")
    plt.xlabel("Forecast step")
    plt.ylabel("CO₂e")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def find_checkpoint() -> str:
    """
    Find a checkpoint to use for inference.

    Priority:
        1) CHECKPOINT_PATH if explicitly specified
        2) Latest *.ckpt file under CHECKPOINT_DIR
    """
    if CHECKPOINT_PATH is not None:
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(f"Explicit checkpoint not found: {CHECKPOINT_PATH}")
        return CHECKPOINT_PATH

    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(
            f"No checkpoint directory found at {CHECKPOINT_DIR}. "
            f"Run training first."
        )

    ckpts = list(CHECKPOINT_DIR.rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found under {CHECKPOINT_DIR}.")
    # Pick the most recently modified
    ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"Using latest checkpoint: {ckpts[0]}")
    return str(ckpts[0])


# ----------------------------------------------------------------------
# Helper: load trained model (on proper device)
# ----------------------------------------------------------------------
def load_trained_model(config: dict, checkpoint_path: str) -> MSTLightningModule:
    """
    Load MSTLightningModule from checkpoint, put it on GPU if available.
    """
    # Load model from checkpoint with saved hyperparameters
    model = MSTLightningModule.load_from_checkpoint(checkpoint_path)

    # Put model in eval mode
    model.eval()

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Model loaded on device:", device)

    return model


# ----------------------------------------------------------------------
# Inference on a single time-series
# ----------------------------------------------------------------------
def predict_from_series(
    model: MSTLightningModule,
    series: Union[List[float], np.ndarray, torch.Tensor],
    context_length: int,
    forecast_horizon: int,
) -> Dict[str, np.ndarray]:
    """
    Run MST forecast on a single 1D series.

    Args:
        model: trained MSTLightningModule
        series: 1D time-series (Python list, NumPy array, or 1D torch tensor)
        context_length: T (must be <= len(series))
        forecast_horizon: H

    Returns:
        dict with:
            {
              "y_fac": np.ndarray (H,),
              "y_sec": np.ndarray (H,),
              "y_nat": np.ndarray (H,),
            }
        in ORIGINAL units (after un-scaling).
    """
    model.eval()
    device = next(model.parameters()).device

    # Convert series to 1D numpy
    if isinstance(series, torch.Tensor):
        series_np = series.detach().cpu().numpy().astype(float)
    else:
        series_np = np.asarray(series, dtype=float)

    if series_np.ndim != 1:
        raise ValueError(f"series must be 1D, got shape {series_np.shape}")

    if len(series_np) < context_length:
        raise ValueError(
            f"series length {len(series_np)} is smaller than context_length {context_length}"
        )

    # Take last T points as context
    ctx = series_np[-context_length:]  # (T,)

    # Scale context to match training
    ctx_scaled = ctx / SCALE

    # Reshape to (1, T, 1) → (B, T, C)
    x = torch.from_numpy(ctx_scaled.reshape(1, context_length, 1)).float().to(device)

    with torch.no_grad():
        y_fac_scaled, y_sec_scaled, y_nat_scaled = model(x)  # each (1, H, 1)

    # Convert outputs to 1D numpy (still scaled)
    y_fac_np = y_fac_scaled.squeeze(0).squeeze(-1).cpu().numpy()
    y_sec_np = y_sec_scaled.squeeze(0).squeeze(-1).cpu().numpy()
    y_nat_np = y_nat_scaled.squeeze(0).squeeze(-1).cpu().numpy()

    # Un-scale outputs back to original units
    y_fac_np = y_fac_np * SCALE
    y_sec_np = y_sec_np * SCALE
    y_nat_np = y_nat_np * SCALE

    return {
        "y_fac": y_fac_np,
        "y_sec": y_sec_np,
        "y_nat": y_nat_np,
    }


# ----------------------------------------------------------------------
# Series loaders
# ----------------------------------------------------------------------
def load_facility_series_dict(path: Path = FACILITY_SERIES_PATH) -> dict:
    import pickle

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run data_preprocessing.py first to create facility_series.pkl."
        )
    with open(path, "rb") as f:
        series_dict = pickle.load(f)
    return series_dict


def load_sector_series_dict(path: Path = SECTOR_SERIES_PATH) -> dict:
    import pickle

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run data_preprocessing.py first to create sector_series.pkl."
        )
    with open(path, "rb") as f:
        series_dict = pickle.load(f)
    return series_dict


def load_national_series(path: Path = NATIONAL_SERIES_PATH):
    import pickle

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run data_preprocessing.py first to create national_series.pkl."
        )
    with open(path, "rb") as f:
        d = pickle.load(f)
    if "national" not in d:
        raise KeyError("national_series.pkl must contain key 'national'.")
    return d["national"]  # pd.Series


# ----------------------------------------------------------------------
# Inference for a specific facility
# ----------------------------------------------------------------------
def predict_for_facility(
    model: MSTLightningModule,
    fac_id: str,
    context_length: int,
    forecast_horizon: int,
    series_dict: Optional[dict] = None,
) -> Dict[str, np.ndarray]:
    """
    Predict horizon for a given fac_id using facility_series.pkl.

    Args:
        model: trained MSTLightningModule
        fac_id: facility ID as string (must exist in facility_series.pkl)
        context_length: T
        forecast_horizon: H
        series_dict: optional {fac_id: pd.Series} (if already loaded)

    Returns:
        same dict as predict_from_series()
    """
    if series_dict is None:
        series_dict = load_facility_series_dict()

    if fac_id not in series_dict:
        raise KeyError(f"Facility ID {fac_id} not found in facility_series.pkl")

    s = series_dict[fac_id]  # pd.Series indexed by year
    values = s.sort_index().to_numpy(dtype=float)  # 1D array
    return predict_from_series(model, values, context_length, forecast_horizon)


# ----------------------------------------------------------------------
# Plotting: facility, sector, national
# ----------------------------------------------------------------------
def plot_hierarchy_last_window(
    model: MSTLightningModule,
    fac_id: str,
    context_length: int,
    forecast_horizon: int,
    facility_series_dict: Optional[dict] = None,
    sector_series_dict: Optional[dict] = None,
    national_series: Optional["pd.Series"] = None,
    sector_name: Optional[str] = None,
):
    """
    Plot facility, sector, and national forecasts vs actuals using
    the last (T + H) window of the chosen facility.

    - Facility:
        context = last T facility points
        true future = last H facility points
        predicted = model's y_fac

    - National:
        true future = national series reindexed to facility's last H years
        predicted   = model's y_nat

    - Sector:
        true future = chosen sector series reindexed to facility's last H years
                      (if sector_name is provided / available)
        predicted   = model's y_sec

    NOTE:
        There is no facility→sector mapping here. sector_name is just a
        chosen sector key from sector_series.pkl. During training y_sec
        was set to fallback = y_nat, so sector predictions are mostly a
        second head on a similar target.
    """
    import pandas as pd  # local import to avoid hard dependency at top

    if facility_series_dict is None:
        facility_series_dict = load_facility_series_dict()
    if fac_id not in facility_series_dict:
        raise KeyError(f"Facility ID {fac_id} not found in facility_series.pkl")

    if national_series is None:
        national_series = load_national_series()
    if not isinstance(national_series, pd.Series):
        raise TypeError("national_series must be a pandas Series.")

    if sector_series_dict is None:
        try:
            sector_series_dict = load_sector_series_dict()
        except FileNotFoundError:
            sector_series_dict = {}

    # Choose a sector if not provided and dict not empty
    if sector_name is None and sector_series_dict:
        sector_name = list(sector_series_dict.keys())[0]

    # Facility series
    s_fac = facility_series_dict[fac_id].sort_index()
    years = s_fac.index.to_numpy()
    values = s_fac.to_numpy(dtype=float)

    T = context_length
    H = forecast_horizon

    if len(values) < T + H:
        raise ValueError(
            f"Facility {fac_id} has only {len(values)} points, "
            f"but need at least T+H = {T + H}."
        )

    # Use last T as context, last H as true future
    ctx_values = values[-(T + H):-H]    # (T,)
    y_fac_true = values[-H:]            # (H,)
    years_future = years[-H:]

    # Run model forecast using context only
    preds = predict_from_series(
        model,
        ctx_values,
        context_length=T,
        forecast_horizon=H,
    )
    y_fac_pred = preds["y_fac"]
    y_sec_pred = preds["y_sec"]
    y_nat_pred = preds["y_nat"]

    # National actual
    nat_series = national_series.sort_index()
    y_nat_true = nat_series.reindex(years_future).to_numpy(dtype=float)

    # Sector actual (if available)
    if sector_name is not None and sector_name in sector_series_dict:
        s_sec = sector_series_dict[sector_name].sort_index()
        y_sec_true = s_sec.reindex(years_future).to_numpy(dtype=float)
    else:
        y_sec_true = None

    steps = np.arange(1, H + 1)

    plt.figure(figsize=(12, 8))

    # 1) Facility
    plt.subplot(3, 1, 1)
    plt.plot(steps, y_fac_true, marker="o", label="Actual facility")
    plt.plot(steps, y_fac_pred, marker="x", linestyle="--", label="Pred facility")
    plt.title(f"Facility {fac_id}: forecast vs actual")
    plt.ylabel("CO₂e")
    plt.legend()
    plt.grid(True)

    # 2) National
    plt.subplot(3, 1, 2)
    plt.plot(steps, y_nat_true, marker="o", label="Actual national")
    plt.plot(steps, y_nat_pred, marker="x", linestyle="--", label="Pred national")
    plt.title("National: forecast vs actual")
    plt.ylabel("CO₂e")
    plt.legend()
    plt.grid(True)

    # 3) Sector
    plt.subplot(3, 1, 3)
    if y_sec_true is not None:
        plt.plot(steps, y_sec_true, marker="o", label=f"Actual sector {sector_name}")
    else:
        plt.plot([], [], label="No sector actual (missing sector series)")
    plt.plot(steps, y_sec_pred, marker="x", linestyle="--", label="Pred sector")
    plt.title(f"Sector: forecast vs actual (sector={sector_name})")
    plt.xlabel("Forecast step")
    plt.ylabel("CO₂e")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Example usage in main()
# ----------------------------------------------------------------------
def main():
    # 1) Load config
    cfg = load_config(CONFIG_PATH)
    T = cfg["data"]["context_length"]
    H = cfg["data"]["forecast_horizon"]

    # 2) Resolve checkpoint & load model
    ckpt_path = find_checkpoint()
    model = load_trained_model(cfg, ckpt_path)

    # 3) Load series dicts
    facility_dict = load_facility_series_dict()
    nat_series = load_national_series()
    try:
        sector_dict = load_sector_series_dict()
    except FileNotFoundError:
        sector_dict = {}

    print("\n=== Plotting hierarchy for MEAN over all facilities and sectors ===")
    plot_mean_hierarchy_last_window(
        model,
        context_length=T,
        forecast_horizon=H,
        facility_series_dict=facility_dict,
        sector_series_dict=sector_dict,
        national_series=nat_series,
    )


if __name__ == "__main__":
    main()
