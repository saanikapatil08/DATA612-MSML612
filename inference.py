#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference & plotting for Multi-Scale Transformer (MST) on CAMPD plant–day data.

Capabilities:
    1) Load a trained MSTLightningModule checkpoint.
    2) Run inference on a custom 1D time-series.
    3) Run inference for a specific plant from plant_series.pkl.
    4) Plot plant, sector, and national forecasts vs actuals (last window).

IMPORTANT:
    This script assumes you used a scaling factor of SCALE = 1e4
    in your dataloader (dataloader.py) when building x (plant context).
    If you change SCALE there, change it here as well.

    Typical daily config we expect:
        context_length = 60 days
        forecast_horizon = 14 days
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
CHECKPOINT_DIR = Path("logs/mst")   # <- adjust if your logs directory differs
CHECKPOINT_PATH: Optional[str] = None

# Processed EPA series (from data_preprocessing.py)
OUT_EPA_DIR = Path("out_epa/output")
PLANT_SERIES_PATH    = OUT_EPA_DIR / "plant_series.pkl"
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
                in ORIGINAL units (e.g., daily plant co2e_total).
        context_length: T days (must be <= len(series))
        forecast_horizon: H days (must match model.forecast_horizon)

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

    # Take last T points as context (T days)
    ctx = series_np[-context_length:]  # (T,)

    # Scale context to match training
    ctx_scaled = ctx / SCALE

    # Reshape to (1, T, 1) → (B, T, C)
    x = torch.from_numpy(ctx_scaled.reshape(1, context_length, 1)).float().to(device)

    with torch.no_grad():
        # MSTLightningModule.forward → mst(x) → returns (y_fac, y_sec, y_nat)
        # All are in the same scaled space as inputs.
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
def load_plant_series_dict(path: Path = PLANT_SERIES_PATH) -> dict:
    """
    Load {plant_id: pd.Series(date -> daily co2e_total)} from plant_series.pkl.
    """
    import pickle

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run data_preprocessing.py first to create plant_series.pkl."
        )
    with open(path, "rb") as f:
        series_dict = pickle.load(f)
    return series_dict


def load_sector_series_dict(path: Path = SECTOR_SERIES_PATH) -> dict:
    """
    Load {sector_name: pd.Series(date -> daily sector_co2)} from sector_series.pkl.
    """
    import pickle

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run data_preprocessing.py first to create sector_series.pkl."
        )
    with open(path, "rb") as f:
        series_dict = pickle.load(f)
    return series_dict


def load_national_series(path: Path = NATIONAL_SERIES_PATH):
    """
    Load national series from national_series.pkl, expecting key 'national'.
    Returns a pandas Series indexed by date (daily).
    """
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
# Inference for a specific plant
# ----------------------------------------------------------------------
def predict_for_plant(
    model: MSTLightningModule,
    plant_id: str,
    context_length: int,
    forecast_horizon: int,
    series_dict: Optional[dict] = None,
) -> Dict[str, np.ndarray]:
    """
    Predict horizon for a given plant_id using plant_series.pkl.

    Args:
        model: trained MSTLightningModule
        plant_id: plant ID as string (must exist in plant_series.pkl)
        context_length: T days
        forecast_horizon: H days
        series_dict: optional {plant_id: pd.Series} (if already loaded)

    Returns:
        same dict as predict_from_series()
    """
    if series_dict is None:
        series_dict = load_plant_series_dict()

    plant_id = plant_id.strip()
    if plant_id not in series_dict:
        raise KeyError(f"Plant ID {plant_id} not found in plant_series.pkl")

    s = series_dict[plant_id]  # pd.Series indexed by daily date
    values = s.sort_index().to_numpy(dtype=float)  # 1D array
    return predict_from_series(model, values, context_length, forecast_horizon)


# ----------------------------------------------------------------------
# Helpers to inspect plant coverage
# ----------------------------------------------------------------------
def get_plants_with_min_history(
    plant_series_dict: dict,
    min_points: int,
) -> Dict[str, int]:
    """
    Return {plant_id: length} for plants with at least min_points observations.
    Each observation is one day.
    """
    eligible = {}
    for pid, s in plant_series_dict.items():
        n = len(s)
        if n >= min_points:
            eligible[pid] = n
    return dict(sorted(eligible.items(), key=lambda kv: kv[1], reverse=True))


def print_plants_with_min_history(
    plant_series_dict: dict,
    min_points: int,
    max_print: int = 50,
):
    """
    Pretty-print plants with at least min_points observations (days).
    """
    eligible = get_plants_with_min_history(plant_series_dict, min_points)
    if not eligible:
        print(f"No plants found with at least {min_points} days of history.")
        return

    print(
        f"\nPlants with at least {min_points} days of history "
        f"(showing up to {max_print}):"
    )
    print("plant_id, length_days")
    for i, (pid, n) in enumerate(eligible.items()):
        if i >= max_print:
            print(f"... (+{len(eligible) - max_print} more)")
            break
        print(f"{pid}, {n}")


# ----------------------------------------------------------------------
# Plotting: plant, sector, national
# ----------------------------------------------------------------------
def plot_hierarchy_last_window(
    model: MSTLightningModule,
    plant_id: str,
    context_length: int,
    forecast_horizon: int,
    plant_series_dict: Optional[dict] = None,
    sector_series_dict: Optional[dict] = None,
    national_series: Optional["pd.Series"] = None,
    sector_name: Optional[str] = None,
):
    """
    Plot plant, sector, and national forecasts vs actuals using
    the last (T + H) daily window of the chosen plant.

    - Plant:
        context = last T daily plant points
        true future = last H daily plant points
        predicted = model's y_fac

    - National:
        true future = national series reindexed to plant's last H dates
        predicted   = model's y_nat

    - Sector:
        true future = chosen sector series reindexed to plant's last H dates
                      (if sector_name is provided / available)
        predicted   = model's y_sec

    NOTE:
        There is no plant→sector mapping here. sector_name is just a
        chosen sector key from sector_series.pkl. During training y_sec
        was set to fallback = y_nat, so sector predictions are mostly a
        second head on a similar target.
    """
    import pandas as pd  # local import to avoid hard dependency at top

    if plant_series_dict is None:
        plant_series_dict = load_plant_series_dict()
    plant_id = plant_id.strip()
    if plant_id not in plant_series_dict:
        raise KeyError(f"Plant ID {plant_id} not found in plant_series.pkl")

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

    # Plant series
    s_plant = plant_series_dict[plant_id].sort_index()
    dates = s_plant.index.to_numpy()
    values = s_plant.to_numpy(dtype=float)

    T = context_length
    H = forecast_horizon

    if len(values) < T + H:
        raise ValueError(
            f"Plant {plant_id} has only {len(values)} daily points, "
            f"but need at least T+H = {T + H} days."
        )

    # Use last T as context, last H as true future
    ctx_values = values[-(T + H):-H]    # (T,)
    y_fac_true = values[-H:]            # (H,)
    dates_future = dates[-H:]

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
    y_nat_true = nat_series.reindex(dates_future).to_numpy(dtype=float)

    # Sector actual (if available)
    if sector_name is not None and sector_name in sector_series_dict:
        s_sec = sector_series_dict[sector_name].sort_index()
        y_sec_true = s_sec.reindex(dates_future).to_numpy(dtype=float)
    else:
        y_sec_true = None

    # For x-axis in plots we just use steps (1..H) = days ahead;
    # you could also use dates_future directly if you prefer calendar dates.
    steps = np.arange(1, H + 1)

    plt.figure(figsize=(12, 8))

    # 1) Plant
    plt.subplot(3, 1, 1)
    plt.plot(steps, y_fac_true, marker="o", label="Actual plant (daily)")
    plt.plot(steps, y_fac_pred, marker="x", linestyle="--", label="Pred plant")
    plt.title(f"Plant {plant_id}: {H}-day forecast vs actual")
    plt.ylabel("CO₂e (daily)")
    plt.legend()
    plt.grid(True)

    # 2) National
    plt.subplot(3, 1, 2)
    plt.plot(steps, y_nat_true, marker="o", label="Actual national (daily)")
    plt.plot(steps, y_nat_pred, marker="x", linestyle="--", label="Pred national")
    plt.title("National: daily forecast vs actual")
    plt.ylabel("CO₂e (daily)")
    plt.legend()
    plt.grid(True)

    # 3) Sector
    plt.subplot(3, 1, 3)
    if y_sec_true is not None:
        plt.plot(steps, y_sec_true, marker="o", label=f"Actual sector {sector_name} (daily)")
    else:
        plt.plot([], [], label="No sector actual (missing sector series)")
    plt.plot(steps, y_sec_pred, marker="x", linestyle="--", label="Pred sector")
    plt.title(f"Sector: daily forecast vs actual (sector={sector_name})")
    plt.xlabel("Days ahead")
    plt.ylabel("CO₂e (daily)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


import argparse


# ----------------------------------------------------------------------
# Example usage in main()
import argparse


# ----------------------------------------------------------------------
# Example usage in main()
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MST inference & plotting on CAMPD plant–day data",
    )
    parser.add_argument(
        "--plant-id",
        type=str,
        default=None,
        help="Plant ID to plot. If not provided, picks the first plant with enough history.",
    )
    parser.add_argument(
        "--sector-name",
        type=str,
        default=None,
        help="Sector name to plot (must be a key in sector_series.pkl). "
             "If not provided, picks the first available sector.",
    )
    parser.add_argument(
        "--list-plants",
        action="store_true",
        help="Only list plant IDs with enough history (T+H days) and exit.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=50,
        help="Max number of plants to print when using --list-plants.",
    )
    args = parser.parse_args()

    # 1) Load config
    cfg = load_config(CONFIG_PATH)
    T = int(cfg["data"]["context_length"])
    H_cfg = int(cfg["data"]["forecast_horizon"])

    # 2) Resolve checkpoint & load model
    ckpt_path = find_checkpoint()
    model = load_trained_model(cfg, ckpt_path)

    # 2a) Get horizon from the trained model (if available)
    H_model = H_cfg
    if hasattr(model, "hparams") and hasattr(model.hparams, "forecast_horizon"):
        try:
            H_model = int(model.hparams.forecast_horizon)
        except Exception:
            H_model = H_cfg

    if H_model != H_cfg:
        print(
            f"[WARN] Config forecast_horizon={H_cfg}, "
            f"but checkpoint was trained with forecast_horizon={H_model}. "
            f"Using H={H_model} for inference."
        )
    H = H_model
    min_points = T + H

    # 3) Load series dicts
    plant_dict = load_plant_series_dict()
    nat_series = load_national_series()
    try:
        sector_dict = load_sector_series_dict()
    except FileNotFoundError:
        sector_dict = {}

    # 3a) Optionally just list plants with enough history
    if args.list_plants:
        print_plants_with_min_history(
            plant_dict,
            min_points=min_points,
            max_print=args.max_print,
        )
        return

    # 4) Choose plant_id
    eligible_plants = get_plants_with_min_history(plant_dict, min_points=min_points)

    if not eligible_plants:
        raise RuntimeError(
            f"No plants have at least T+H = {min_points} days of history. "
            f"Check your data, context_length, and forecast_horizon."
        )

    if args.plant_id is not None:
        plant_id = args.plant_id.strip()
        if plant_id not in eligible_plants:
            raise KeyError(
                f"Requested plant_id='{plant_id}' does not have at least {min_points} days "
                f"or is not present in plant_series.pkl."
            )
        example_plant_id = plant_id
    else:
        # pick the longest series by default
        example_plant_id = next(iter(eligible_plants.keys()))

    # 5) Choose sector_name (optional)
    if sector_dict:
        if args.sector_name is not None:
            if args.sector_name not in sector_dict:
                raise KeyError(
                    f"Requested sector_name={args.sector_name} not found in sector_series.pkl."
                )
            example_sector_name = args.sector_name
        else:
            example_sector_name = list(sector_dict.keys())[0]
    else:
        example_sector_name = None

    print(
        f"\n=== Plotting hierarchy for plant_id='{example_plant_id}' "
        f"sector={example_sector_name} (T={T} days, H={H} days) ==="
    )

    plot_hierarchy_last_window(
        model,
        plant_id=example_plant_id,
        context_length=T,
        forecast_horizon=H,     # <- now H comes from model
        plant_series_dict=plant_dict,
        sector_series_dict=sector_dict,
        national_series=nat_series,
        sector_name=example_sector_name,
    )


if __name__ == "__main__":
    main()
