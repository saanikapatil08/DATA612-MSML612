#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loaders for Multi-Scale Transformer (MST) hierarchical time-series.

Consumes the outputs from data_processing.py:

- out_epa/output/facility_series.pkl   # {fac_id: pd.Series(year -> co2e_total)}
- out_epa/output/sector_series.pkl     # {sector_name: pd.Series(year -> sector_co2e)}
- out_epa/output/national_series.pkl   # {"national": pd.Series(year -> national_co2e)}
- out_epa/output/splits_years.json     # {"train": [...], "val": [...], "test": [...]}

We build sliding facility windows:

Inputs:
    x:      (T, 1)  facility context window

Targets:
    y_fac:  (H, 1)  facility emissions
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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ROOT_DIR = Path("out_epa")
OUTPUT_DIR = ROOT_DIR / "output"

FACILITY_SERIES_PATH = OUTPUT_DIR / "facility_series.pkl"
SECTOR_SERIES_PATH   = OUTPUT_DIR / "sector_series.pkl"
NATIONAL_SERIES_PATH = OUTPUT_DIR / "national_series.pkl"
SPLITS_PATH          = OUTPUT_DIR / "splits_years.json"


# ------------------ loading utilities ------------------ #

def _load_pickle(path: Path):
    import pickle
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}. Run data_processing.py first.")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info("Loaded %s", path)
    return obj


def load_series_dicts(root_dir: Path = OUTPUT_DIR):
    facility_series = _load_pickle(root_dir / "facility_series.pkl")
    sector_series   = _load_pickle(root_dir / "sector_series.pkl")
    national_dict   = _load_pickle(root_dir / "national_series.pkl")

    if "national" not in national_dict:
        raise KeyError("national_series.pkl must contain key 'national'.")

    nat_series = national_dict["national"]
    return facility_series, sector_series, nat_series


def load_splits(path: Path = SPLITS_PATH) -> Dict[str, List[int]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing splits_years.json at {path}. "
            f"Run datadownload.py (make_splits) and data_processing.py."
        )
    with open(path, "r") as f:
        d = json.load(f)
    splits = {k: [int(y) for y in v] for k, v in d.items()}
    logger.info("Loaded splits_years.json: %s", splits)
    return splits


# ------------------ sliding window builder ------------------ #

def _build_facility_windows(
    fac_id: str,
    s_fac: pd.Series,
    nat_series: pd.Series,
    context_length: int,
    forecast_horizon: int,
    split_years: List[int]  # can also be None
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build sliding windows for a single facility.

    Returns a list of (x, y_fac, y_sec, y_nat) numpy arrays with shapes:
        x:      (T, 1)
        y_fac:  (H, 1)
        y_sec:  (H, 1)  # sector-level target (currently = national)
        y_nat:  (H, 1)
    """

    # Sort by year and extract arrays
    s_fac = s_fac.sort_index()
    years = s_fac.index.to_numpy()
    values = s_fac.to_numpy(dtype=float)
    n = len(years)

    if n < context_length + forecast_horizon:
        return []

    # Downstream code may pass None (e.g. for train) to disable year filtering
    split_years_set = set(int(y) for y in split_years) if split_years else None

    nat_series = nat_series.sort_index()

    T = context_length
    H = forecast_horizon
    samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for start_idx in range(0, n - (T + H) + 1):
        ctx_years   = years[start_idx : start_idx + T]
        horizon_yrs = years[start_idx + T : start_idx + T + H]

        # Filter by horizon last year only if we HAVE a split_years_set
        if split_years_set is not None and int(horizon_yrs[-1]) not in split_years_set:
            continue

        # Facility context + target
        x_fac = values[start_idx : start_idx + T].reshape(T, 1)
        y_fac = values[start_idx + T : start_idx + T + H].reshape(H, 1)

        # National target aligned by year
        y_nat_vals = nat_series.reindex(horizon_yrs).to_numpy(dtype=float)
        if np.any(np.isnan(y_nat_vals)):
            continue
        y_nat = y_nat_vals.reshape(H, 1)

        # ---- SCALE EVERYTHING DOWN TO A REASONABLE RANGE ----
        # You can adjust this depending on your data magnitudes.
        # but y_fac / y_nat / y_sec are left in original units so we can
        # apply log1p in the LightningModule.
        scale_x = 1e4  # adjust if needed
        x_fac = x_fac / scale_x
        # y_fac, y_nat, y_sec remain unscaled

        # Sector target: fallback = national for now
        y_sec = y_nat.copy()

        samples.append((x_fac, y_fac, y_sec, y_nat))

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
        min_facility_length: int = 8,
    ):
        assert split in ("train", "val", "test"), f"Invalid split: {split}"

        facility_series, _sector_series, nat_series = load_series_dicts(root_dir)
        splits = load_splits(root_dir / "splits_years.json")

        if split == "train":
            split_years = None      # do NOT filter train windows by year
        else:
            split_years = splits.get(split, [])

        min_len = min(len(s) for s in facility_series.values())
        max_len = max(len(s) for s in facility_series.values())
        print(f"[{split}] facility series lengths: min={min_len}, max={max_len}")
        print(f"[{split}] required length per sample: T+H = {context_length + forecast_horizon}")

        self.samples = []
        for fac_id, s_fac in facility_series.items():
            if len(s_fac) < min_facility_length:
                continue

            win = _build_facility_windows(
                fac_id=str(fac_id),
                s_fac=s_fac,
                nat_series=nat_series,
                context_length=context_length,
                forecast_horizon=forecast_horizon,
                split_years=split_years,
            )
            self.samples.extend(win)

        if not self.samples:
            logger.warning(
                "No samples built for split '%s'. Check context_length, "
                "forecast_horizon, and splits_years.json.", split
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
