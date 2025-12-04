#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data processing for MST — build hierarchical time-series dicts
from the finalized CAMPD monthly pipeline outputs.

Assumes these files already exist (from your builder pipeline):

1) out_epa/training_plant_period.csv
   - primary_key: [plant_id, date]
   - columns include:
       plant_id, date (month-start), co2e_total, ...
       (plus engineered features, sector_name, etc.)

2) out_epa/sector_period_totals.csv
   - primary_key: [sector_name, date]
   - columns: date, sector_name, sector_co2

3) out_epa/splits_dates.json
   - {
       "train": ["2015-01-01","2022-12-31"],
       "val":   ["2023-01-01","2023-12-31"],
       "test":  ["2024-01-01","2024-12-31"]
     }

Outputs written to out_epa/output/:

- plant_series.pkl    # {plant_id: pd.Series(date -> co2e_total)}
- sector_series.pkl   # {sector_name: pd.Series(date -> sector_co2)}
- national_series.pkl # {"national": pd.Series(date -> national_co2)}
- splits_dates.json   # copy of splits for convenience

These are what your MST dataloader will consume.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict

import pandas as pd

# ---------------------------------------------------------------------
# Paths & logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path("out_epa")
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_PLANT_PATH = BASE_DIR / "training_plant_period.csv"
SECTOR_TOTALS_PATH = BASE_DIR / "sector_period_totals.csv"
SPLITS_PATH        = BASE_DIR / "splits_dates.json"


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------
def load_training_plant(path: Path = TRAINING_PLANT_PATH) -> pd.DataFrame:
    """Load training_plant_period.csv and standardize types."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Make sure your EPA pipeline "
            f"has produced training_plant_period.csv."
        )

    df = pd.read_csv(path, parse_dates=["date"])
    logger.info("Loaded training_plant_period.csv with %d rows, %d columns.",
                len(df), df.shape[1])

    required_id_cols = {"plant_id", "date"}
    missing_ids = required_id_cols - set(df.columns)
    if missing_ids:
        raise KeyError(f"training_plant_period.csv missing columns: {missing_ids}")

    # Target column: prefer co2e_total, fall back to co2_total if needed
    cols_low = {c.lower(): c for c in df.columns}
    co2e_col = cols_low.get("co2e_total") or cols_low.get("co2_total")
    if co2e_col is None:
        raise KeyError(
            "training_plant_period.csv must contain 'co2e_total' or 'co2_total' column "
            "for plant-level emissions."
        )

    if co2e_col != "co2e_total":
        df = df.rename(columns={co2e_col: "co2e_total"})

    # Normalize date to month-start and types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["plant_id", "date", "co2e_total"])
    df["date"] = df["date"].dt.normalize()

    df["co2e_total"] = pd.to_numeric(df["co2e_total"], errors="coerce")
    df = df.dropna(subset=["co2e_total"])

    return df


def load_sector_totals(path: Path = SECTOR_TOTALS_PATH) -> pd.DataFrame:
    """Load sector_period_totals.csv and aggregate by (date, sector_name) if needed."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Your EPA pipeline should have written sector_period_totals.csv."
        )

    df = pd.read_csv(path, parse_dates=["date"])
    logger.info("Loaded sector_period_totals.csv with %d rows, %d columns.",
                len(df), df.shape[1])

    cols_low = {c.lower(): c for c in df.columns}
    date_col    = cols_low.get("date")
    sector_col  = cols_low.get("sector_name") or cols_low.get("sector")
    sector_co2_col = (
        cols_low.get("sector_co2")
        or cols_low.get("sector_co2e")
        or cols_low.get("co2e")
        or cols_low.get("emissions")
    )

    if date_col is None or sector_col is None or sector_co2_col is None:
        raise KeyError(
            "sector_period_totals.csv must have columns for date, sector_name, and sector_co2 "
            f"(found columns: {df.columns.tolist()})"
        )

    df = df.rename(columns={
        date_col: "date",
        sector_col: "sector_name",
        sector_co2_col: "sector_co2",
    })

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sector_co2"] = pd.to_numeric(df["sector_co2"], errors="coerce")
    df = df.dropna(subset=["date", "sector_name", "sector_co2"])
    df["date"] = df["date"].dt.normalize()

    # Aggregate in case there are multiple rows per (date, sector_name)
    df_tidy = (
        df.groupby(["date", "sector_name"], as_index=False)["sector_co2"]
          .sum()
          .sort_values(["sector_name", "date"])
    )

    logger.info("Tidied sector totals to %d rows (unique date×sector).", len(df_tidy))
    return df_tidy


def load_splits_raw(path: Path = SPLITS_PATH) -> Dict[str, list]:
    """
    Load splits_dates.json as raw dict; we just copy it into output for the dataloader.
    """
    if not path.exists():
        logger.warning("splits_dates.json not found at %s — proceeding without splits.", path)
        return {}
    with open(path, "r") as f:
        splits = json.load(f)
    logger.info("Loaded splits_dates.json: %s", splits)
    return splits


def save_pickle(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Wrote %s", path)


# ---------------------------------------------------------------------
# Series builders
# ---------------------------------------------------------------------
def build_plant_series(
    training_plant: pd.DataFrame,
    min_points: int = 1,
) -> Dict[str, pd.Series]:
    """
    Build {plant_id: Series(date -> co2e_total)} from training_plant_period.

    Uses only 'plant_id', 'date', and 'co2e_total'.
    """
    required = {"plant_id", "date", "co2e_total"}
    missing = required - set(training_plant.columns)
    if missing:
        raise KeyError(f"training_plant_period.csv missing columns: {missing}")

    df = (
        training_plant
        .copy()
        .loc[:, ["plant_id", "date", "co2e_total"]]
        .dropna(subset=["date", "co2e_total"])
    )

    plant_series: Dict[str, pd.Series] = {}

    for plant_id, g in df.groupby("plant_id"):
        s = (
            g[["date", "co2e_total"]]
            .drop_duplicates(subset=["date"])
            .set_index("date")["co2e_total"]
            .sort_index()
        )
        if len(s) >= min_points:
            plant_series[str(plant_id)] = s

    logger.info("Built plant series for %d plants (min_points=%d).",
                len(plant_series), min_points)
    return plant_series


def build_sector_series(
    sector_tidy: pd.DataFrame,
    min_points: int = 6,
) -> Dict[str, pd.Series]:
    """
    Build {sector_name: Series(date -> sector_co2)}.
    """
    required = {"date", "sector_name", "sector_co2"}
    missing = required - set(sector_tidy.columns)
    if missing:
        raise KeyError(f"sector_period_totals missing columns: {missing}")

    sector_series: Dict[str, pd.Series] = {}

    for sector, g in sector_tidy.groupby("sector_name"):
        s = (
            g[["date", "sector_co2"]]
            .drop_duplicates(subset=["date"])
            .set_index("date")["sector_co2"]
            .sort_index()
        )
        if len(s) >= min_points:
            sector_series[str(sector)] = s

    logger.info("Built sector series for %d sectors (min_points=%d).",
                len(sector_series), min_points)
    return sector_series


def build_national_series(sector_tidy: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Build {"national": Series(date -> national_co2)},
    where national_co2 is sum over sectors for each date.
    """
    nat = (
        sector_tidy
        .groupby("date", as_index=True)["sector_co2"]
        .sum()
        .sort_index()
    )
    logger.info("Built national series with %d months.", len(nat))
    return {"national": nat}


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    logger.info("=== MST data processing (plant-month structure) ===")

    # 1) Load tables
    training_plant = load_training_plant(TRAINING_PLANT_PATH)
    sector_tidy    = load_sector_totals(SECTOR_TOTALS_PATH)
    splits_raw     = load_splits_raw(SPLITS_PATH)

    # 2) Build hierarchical time series
    plant_series   = build_plant_series(training_plant, min_points=1)
    sector_series  = build_sector_series(sector_tidy, min_points=6)
    national_series = build_national_series(sector_tidy)

    # 3) Save pickles for the MST dataloader
    save_pickle(plant_series,   OUTPUT_DIR / "plant_series.pkl")
    save_pickle(sector_series,  OUTPUT_DIR / "sector_series.pkl")
    save_pickle(national_series, OUTPUT_DIR / "national_series.pkl")

    # 4) Copy splits_dates.json for convenience
    if splits_raw:
        with open(OUTPUT_DIR / "splits_dates.json", "w") as f:
            json.dump(splits_raw, f, indent=2)
        logger.info("Copied splits_dates.json → %s", OUTPUT_DIR / "splits_dates.json")

    logger.info("Done. Hierarchical series written under %s", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
