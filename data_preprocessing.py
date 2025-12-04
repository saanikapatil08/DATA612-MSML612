#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data processing for MST — build hierarchical time-series dicts
from the finalized EPA pipeline outputs.

Assumes these files already exist (from your builder pipeline):

1) out_epa/training_facility_year.csv
   - primary_key: [fac_id, year]
   - columns include:
       fac_id, year, co2e_total, ...
       (optionally sector_name, sector_co2e, sector lags, etc.)

2) out_epa/sector_year_totals.csv
   - primary_key: [year, sector_name]
   - columns: year, sector_name, sector_co2e

3) out_epa/splits_years.json
   - {"train": [...], "val": [...], "test": [...]}

Outputs written to out_epa/output/:

- facility_series.pkl   # {fac_id: pd.Series(year -> co2e_total)}
- sector_series.pkl     # {sector_name: pd.Series(year -> sector_co2e)}
- national_series.pkl   # {"national": pd.Series(year -> national_co2e)}
- splits_years.json     # copy of splits for convenience

These are what your MST dataloader will consume.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Paths & logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUT_DIR = Path("out_epa")
OUTPUT_DIR = OUT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_FACILITY_PATH = OUT_DIR / "training_facility_year.csv"
SECTOR_TOTALS_PATH     = OUT_DIR / "sector_year_totals.csv"
SPLITS_PATH            = OUT_DIR / "splits_years.json"


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------
def load_training_facility(path: Path = TRAINING_FACILITY_PATH) -> pd.DataFrame:
    """Load training_facility_year.csv and standardize types."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Make sure your EPA builder pipeline "
            f"has produced training_facility_year.csv."
        )

    df = pd.read_csv(path)
    logger.info("Loaded training_facility_year.csv with %d rows, %d columns.",
                len(df), df.shape[1])

    if "year" not in df.columns or "fac_id" not in df.columns:
        raise KeyError("training_facility_year.csv must have 'fac_id' and 'year' columns.")

    if "co2e_total" not in df.columns:
        raise KeyError("training_facility_year.csv must contain 'co2e_total' column "
                       "for facility-level emissions.")

    # Normalize types
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", "fac_id", "co2e_total"])
    df["year"] = df["year"].astype(int)
    df["co2e_total"] = pd.to_numeric(df["co2e_total"], errors="coerce")

    return df


def load_sector_totals(path: Path = SECTOR_TOTALS_PATH) -> pd.DataFrame:
    """Load sector_year_totals.csv and aggregate by (year, sector_name) if needed."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Your EPA builder should have written sector_year_totals.csv."
        )

    df = pd.read_csv(path)
    logger.info("Loaded sector_year_totals.csv with %d rows, %d columns.",
                len(df), df.shape[1])

    cols_low = {c.lower(): c for c in df.columns}
    # Map flexible column names to canonical ones
    year_col   = cols_low.get("year")
    sector_col = cols_low.get("sector_name") or cols_low.get("sector")
    co2e_col   = cols_low.get("sector_co2e") or cols_low.get("co2e") or cols_low.get("emissions")

    if year_col is None or sector_col is None or co2e_col is None:
        raise KeyError(
            "sector_year_totals.csv must have columns for year, sector_name, and sector_co2e "
            f"(found columns: {df.columns.tolist()})"
        )

    df = df.rename(columns={
        year_col: "year",
        sector_col: "sector_name",
        co2e_col: "sector_co2e",
    })

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["sector_co2e"] = pd.to_numeric(df["sector_co2e"], errors="coerce")
    df = df.dropna(subset=["year", "sector_name", "sector_co2e"])
    df["year"] = df["year"].astype(int)

    # Aggregate in case there are multiple rows per (year, sector_name)
    df_tidy = (df
               .groupby(["year", "sector_name"], as_index=False)["sector_co2e"]
               .sum()
               .sort_values(["sector_name", "year"]))

    logger.info("Tidied sector totals to %d rows (unique year×sector).", len(df_tidy))
    return df_tidy


def load_splits(path: Path = SPLITS_PATH) -> Dict[str, list]:
    if not path.exists():
        logger.warning("splits_years.json not found at %s — proceeding without splits.", path)
        return {}
    with open(path, "r") as f:
        splits = json.load(f)
    # normalize to int lists
    out = {k: [int(y) for y in v] for k, v in splits.items()}
    logger.info("Loaded splits_years.json: %s", out)
    return out


def save_pickle(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Wrote %s", path)


# ---------------------------------------------------------------------
# Series builders
# ---------------------------------------------------------------------
def build_facility_series(
    training_facility: pd.DataFrame,
    min_points: int = 4,
) -> Dict[str, pd.Series]:
    """
    Build {fac_id: Series(year -> co2e_total)} from training_facility_year.

    Uses only 'fac_id', 'year', and 'co2e_total'.
    """
    required = {"fac_id", "year", "co2e_total"}
    missing = required - set(training_facility.columns)
    if missing:
        raise KeyError(f"training_facility_year.csv missing columns: {missing}")

    df = (training_facility
          .copy()
          .loc[:, ["fac_id", "year", "co2e_total"]]
          .dropna(subset=["year", "co2e_total"]))

    facility_series: Dict[str, pd.Series] = {}

    for fac_id, g in df.groupby("fac_id"):
        s = (g[["year", "co2e_total"]]
             .drop_duplicates(subset=["year"])
             .set_index("year")["co2e_total"]
             .sort_index())
        if len(s) >= min_points:
            facility_series[str(fac_id)] = s

    logger.info("Built facility series for %d facilities (min_points=%d).",
                len(facility_series), min_points)
    return facility_series


def build_sector_series(
    sector_tidy: pd.DataFrame,
    min_points: int = 3,
) -> Dict[str, pd.Series]:
    """
    Build {sector_name: Series(year -> sector_co2e)}.
    """
    required = {"year", "sector_name", "sector_co2e"}
    missing = required - set(sector_tidy.columns)
    if missing:
        raise KeyError(f"sector_year_totals missing columns: {missing}")

    sector_series: Dict[str, pd.Series] = {}

    for sector, g in sector_tidy.groupby("sector_name"):
        s = (g[["year", "sector_co2e"]]
             .drop_duplicates(subset=["year"])
             .set_index("year")["sector_co2e"]
             .sort_index())
        if len(s) >= min_points:
            sector_series[str(sector)] = s

    logger.info("Built sector series for %d sectors (min_points=%d).",
                len(sector_series), min_points)
    return sector_series


def build_national_series(sector_tidy: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Build {"national": Series(year -> national_co2e)},
    where national_co2e is sum over sectors for each year.
    """
    nat = (sector_tidy
           .groupby("year", as_index=True)["sector_co2e"]
           .sum()
           .sort_index())
    logger.info("Built national series with %d years.", len(nat))
    return {"national": nat}


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    logger.info("=== MST data processing (new structure) ===")

    # 1) Load tables
    training_facility = load_training_facility(TRAINING_FACILITY_PATH)
    sector_tidy       = load_sector_totals(SECTOR_TOTALS_PATH)
    splits            = load_splits(SPLITS_PATH)

    # 2) Build hierarchical time series
    facility_series = build_facility_series(training_facility, min_points=4)
    sector_series   = build_sector_series(sector_tidy, min_points=3)
    national_series = build_national_series(sector_tidy)

    # 3) Save pickles for the MST dataloader
    save_pickle(facility_series, OUTPUT_DIR / "facility_series.pkl")
    save_pickle(sector_series,   OUTPUT_DIR / "sector_series.pkl")
    save_pickle(national_series, OUTPUT_DIR / "national_series.pkl")

    # 4) Copy splits_years.json for convenience
    if splits:
        with open(OUTPUT_DIR / "splits_years.json", "w") as f:
            json.dump(splits, f, indent=2)
        logger.info("Copied splits_years.json → %s", OUTPUT_DIR / "splits_years.json")

    logger.info("Done. Hierarchical series written under %s", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
