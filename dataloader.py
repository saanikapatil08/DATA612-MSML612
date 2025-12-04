#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loaders for Multi-Scale Transformer (MST) hierarchical time-series.

Consumes the outputs from data_preprocessing.py:

- out_epa/output/plant_series.pkl     # {plant_id: pd.Series(date -> co2e_total)}
- out_epa/output/sector_series.pkl    # {sector_name: pd.Series(date -> sector_co2)}
- out_epa/output/national_series.pkl  # {"national": pd.Series(date -> national_co2)}
- out_epa/output/splits_dates.json    # {"train": ["2015-01-01","2022-12-31"], ...}

We build sliding plant windows:

Inputs:
    x:      (T, 1)  plant context window (scaled)

Targets (still linear/original magnitudes):
    y_fac:  (H, 1)  plant emissions
    y_sec:  (H, 1)  sector-level target (currently fallback = national)
    y_nat:  (H, 1)  national emissions

Batch item:
    {
      "x":     Tensor(B, T, 1),
      "y_fac": Tensor(B, H, 1),
      "y_sec": Tensor(B, H, 1),
      "y_nat": Tensor(B, H, 1),
    }
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ROOT_DIR   = Path("out_epa")
OUTPUT_DIR = ROOT_DIR / "output"

PLANT_SERIES_PATH   = OUTPUT_DIR / "plant_series.pkl"
SECTOR_SERIES_PATH  = OUTPUT_DIR / "sector_series.pkl"
NATIONAL_SERIES_PATH = OUTPUT_DIR / "national_series.pkl"
SPLITS_PATH         = OUTPUT_DIR / "splits_dates.json"


# ------------------ loading utilities ------------------ #

def _load_pickle(path: Path):
    import pickle
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}. Run data_preprocessing.py first.")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info("Loaded %s", path)
    return obj


def load_series_dicts(root_dir: Path = OUTPUT_DIR):
    plant_series   = _load_pickle(root_dir / "plant_series.pkl")
    sector_series  = _load_pickle(root_dir / "sector_series.pkl")   # not used yet
    national_dict  = _load_pickle(root_dir / "national_series.pkl")

    if "national" not in national_dict:
        raise KeyError("national_series.pkl must contain key 'national'.")

    nat_series = national_dict["national"]
    return plant_series, sector_series, nat_series


def load_splits(path: Path = SPLITS_PATH) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Load splits_dates.json and return:
        {
          "train": (start_ts, end_ts),
          "val":   (start_ts, end_ts),
          "test":  (start_ts, end_ts),
        }
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing splits_dates.json at {path}. "
            f"Run data_preprocessing.py to copy it into output."
        )
    with open(path, "r") as f:
        d = json.load(f)

    out: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for split, val in d.items():
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            raise ValueError(f"Expected [start, end] for split '{split}', got: {val}")
        start_str, end_str = val
        start_ts = pd.to_datetime(start_str).normalize()
        end_ts   = pd.to_datetime(end_str).normalize()
        out[split] = (start_ts, end_ts)

    logger.info("Loaded splits_dates.json: %s", out)
    return out


# ------------------ sliding window builder ------------------ #

def _build_plant_windows(
    plant_id: str,
    s_plant: pd.Series,
    nat_series: pd.Series,
    context_length: int,
    forecast_horizon: int,
    split_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]],
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build sliding windows for a single plant.

    Returns a list of (x, y_fac, y_sec, y_nat) numpy arrays with shapes:
        x:      (T, 1)
        y_fac:  (H, 1)
        y_sec:  (H, 1)  # sector-level target (currently = national)
        y_nat:  (H, 1)
    """

    # Series is indexed by date (Timestamp)
    s_plant = s_plant.sort_index()
    dates = pd.to_datetime(s_plant.index).to_numpy()  # numpy datetime64
    values = s_plant.to_numpy(dtype=float)
    n = len(dates)

    if n < context_length + forecast_horizon:
        return []

    nat_series = nat_series.sort_index()

    T = context_length
    H = forecast_horizon
    samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for start_idx in range(0, n - (T + H) + 1):
        ctx_dates    = dates[start_idx: start_idx + T]
        horizon_dates = dates[start_idx + T: start_idx + T + H]

        # Filter by horizon last date if we have a split_range
        if split_range is not None:
            last_h_date = pd.to_datetime(horizon_dates[-1]).normalize()
            start_ts, end_ts = split_range
            if not (start_ts <= last_h_date <= end_ts):
                continue

        # Plant context + target
        x_plant = values[start_idx: start_idx + T].reshape(T, 1)
        y_fac   = values[start_idx + T: start_idx + T + H].reshape(H, 1)

        # National target aligned by date
        horizon_idx = pd.to_datetime(horizon_dates)
        y_nat_vals = nat_series.reindex(horizon_idx).to_numpy(dtype=float)
        if np.any(np.isnan(y_nat_vals)):
            # Skip if we don't have national data for this horizon
            continue
        y_nat = y_nat_vals.reshape(H, 1)

        # ---- SCALE INPUTS TO A REASONABLE RANGE ----
        # Adjust if needed based on your magnitudes.
        scale_x = 1e4
        x_plant = x_plant / scale_x

        # Sector target: fallback = national for now
        y_sec = y_nat.copy()

        samples.append((x_plant, y_fac, y_sec, y_nat))

    return samples


# ------------------ Dataset ------------------ #

class HierarchicalWindowedDataset(Dataset):
    """
    Dataset of windowed hierarchical samples for MST.

    Each item:
        {
          "x":     (T, 1),
          "y_fac": (H, 1),
          "y_sec": (H, 1),
          "y_nat": (H, 1),
        }
    """

    def __init__(
        self,
        split: str,
        context_length: int,
        forecast_horizon: int,
        root_dir: Path = OUTPUT_DIR,
        min_plant_length: int = 1,
    ):
        assert split in ("train", "val", "test"), f"Invalid split: {split}"

        plant_series, _sector_series, nat_series = load_series_dicts(root_dir)
        splits = load_splits(root_dir / "splits_dates.json")

        split_range = splits.get(split)
        if split_range is None:
            raise KeyError(f"Split '{split}' not found in splits_dates.json")

        min_len = min(len(s) for s in plant_series.values())
        max_len = max(len(s) for s in plant_series.values())
        print(f"[{split}] plant series lengths: min={min_len}, max={max_len}")
        print(f"[{split}] required length per sample: T+H = {context_length + forecast_horizon}")

        self.samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for plant_id, s_plant in plant_series.items():
            if len(s_plant) < min_plant_length:
                continue

            win = _build_plant_windows(
                plant_id=str(plant_id),
                s_plant=s_plant,
                nat_series=nat_series,
                context_length=context_length,
                forecast_horizon=forecast_horizon,
                split_range=split_range,
            )
            self.samples.extend(win)

        if not self.samples:
            logger.warning(
                "No samples built for split '%s'. Check context_length, "
                "forecast_horizon, splits_dates.json, and data coverage.", split
            )
        else:
            logger.info("Built %d samples for split '%s'.", len(self.samples), split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x_np, y_fac_np, y_sec_np, y_nat_np = self.samples[idx]
        return {
            "x":     torch.from_numpy(x_np).float(),
            "y_fac": torch.from_numpy(y_fac_np).float(),
            "y_sec": torch.from_numpy(y_sec_np).float(),
            "y_nat": torch.from_numpy(y_nat_np).float(),
        }


# ------------------ Public factory ------------------ #

def create_dataloaders(
    context_length: int,
    forecast_horizon: int,
    batch_size: int = 64,
    num_workers: int = 1,
    root_dir: Path = OUTPUT_DIR,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_ds = HierarchicalWindowedDataset(
        split="train",
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        root_dir=root_dir,
    )
    val_ds = HierarchicalWindowedDataset(
        split="val",
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        root_dir=root_dir,
    )
    test_ds = HierarchicalWindowedDataset(
        split="test",
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        root_dir=root_dir,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, val_loader, test_loader
