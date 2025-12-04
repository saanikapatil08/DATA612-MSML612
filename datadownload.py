#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build EPA GHGRP dataset for MST training — schema-robust version.

Key fixes:
- Accepts ID-style schemas like: ['co2e_emission','facility_id','gas_id','sub_part_id','year'].
- Normalizes columns to internal names to avoid KeyError: None.
- Uses CO2e directly when provided (e.g., 'co2e_emission').
- Adds MST-friendly features: sector features, gas masks, years_of_history, integer IDs.

Outputs: out_epa/ (raw downloads + combined tables)
"""

import os, json, warnings
from io import StringIO
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import requests

warnings.simplefilter("ignore", category=FutureWarning)

# ===================== Config =====================
OUT_DIR = "out_epa"
RAW_DIR = os.path.join(OUT_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# Pick one: "AR4" | "AR5" | "AR6" (only used when quantity is NOT already CO2e)
GWP_VERSION = "AR5"

# Year splits (by reporting year). If None → auto: last 2 test, prev 2 val.
SPLIT_CONFIG = {"test_years": None, "val_years": None}

# Feature knobs
LAGS = [1, 3, 5]
ROLL_WINDOWS = [3]

# API tables (still supported if you want fresh pulls)
BASE = "https://data.epa.gov/efservice"
T_FACILITY   = "PUB_DIM_FACILITY"   # contains facility metadata which are in ghgrp program
T_EMIS_SUBP  = "PUB_FACTS_SUBP_GHG_EMISSION" # emissions by subpart like year, gas
T_SECTOR     = "PUB_FACTS_SECTOR_GHG_EMISSION" # emissions by sector
T_FRS_SITE   = "FRS_FACILITY_SITES" # meta data for all frs sites
T_FRS_PROG   = "FRS_PROGRAM_LINKS" # data telling which frs sites are in which programs

# GWP tables (100y horizon)
GWP_TABLES = {
    "AR4": {"CO2": 1, "CH4": 25, "N2O": 298, "SF6": 22800, "NF3": 17200},
    "AR5": {"CO2": 1, "CH4": 28, "N2O": 265, "SF6": 23500, "NF3": 16100},
    "AR6": {"CO2": 1, "CH4": 27.2, "N2O": 273, "SF6": 25400, "NF3": 19900},
}
FAMILY_DEFAULTS = {"HFC": 1240, "PFC": 7390}

def gwp_lookup(gas_upper: str) -> float:
    tbl = GWP_TABLES[GWP_VERSION]
    if gas_upper in tbl: return tbl[gas_upper]
    if gas_upper.startswith("HFC"): return FAMILY_DEFAULTS["HFC"]
    if gas_upper.startswith("PFC"): return FAMILY_DEFAULTS["PFC"]
    return 0.0

# ===================== Helpers =====================
def ef_csv(table: str, where: Optional[List[Tuple[str, str]]] = None, rows=(0, 9_999_999)) -> pd.DataFrame:
    url = f"{BASE}/{table}"
    if where:
        for col, val in where:
            url += f"/{col}/{val}"
    url += f"/ROWS/{rows[0]}:{rows[1]}/CSV"
    r = requests.get(url, timeout=240)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text), low_memory=False)

def safe_save(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df):,} rows)")

def pick_optional(colset, *cands):
    up = {c.lower(): c for c in colset}
    for c in cands:
        if c in colset: return c
        if c.lower() in up: return up[c.lower()]
    return None

def pick_required(context: str, colset, *cands) -> str:
    col = pick_optional(colset, *cands)
    if col is None:
        sample_cols = sorted(list(colset))[:25]
        raise KeyError(f"[{context}] Required column not found. "
                       f"Tried {list(cands)}; available columns (sample): {sample_cols} ...")
    return col

# --- Normalizers: standardize column names upfront (robust to lowercase/underscore schemas) ---
def normalize_fac_cols(df: pd.DataFrame) -> pd.DataFrame:
    c = {col: col for col in df.columns}
    cols_low = {col.lower(): col for col in df.columns}
    # Map to internal canonical names
    mapping = {}
    # facility id
    for k in ["facility_id", "facility_id_number", "facilityid"]:
        if k in cols_low: mapping[cols_low[k]] = "fac_id"
    # common attrs
    for k, std in [
        ("facility_name","fac_name"), ("name","fac_name"),
        ("state","state"), ("facility_state","state"),
        ("county","county"), ("facility_county","county"),
        ("city","city"), ("facility_city","city"),
        ("zip","zip"), ("facility_zip","zip"),
        ("latitude","lat"), ("facility_latitude","lat"), ("lat_dd","lat"),
        ("longitude","lon"), ("facility_longitude","lon"), ("lon_dd","lon"),
        ("registry_id","frs_registry_id"), ("frs_id","frs_registry_id"),
    ]:
        if k in cols_low: mapping[cols_low[k]] = std
    
    print("Facility column mapping:", mapping)
    return df.rename(columns=mapping)

def normalize_emis_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols_low = {col.lower(): col for col in df.columns}
    mapping = {}
    # facility id
    for k in ["facility_id", "facility_id_number", "facilityid"]:
        if k in cols_low: mapping[cols_low[k]] = "fac_id"
    # year
    for k in ["reporting_year", "ghg_reporting_year", "year"]:
        if k in cols_low: mapping[cols_low[k]] = "year"
    # gas (name or id)
    for k in ["gas_name", "ghg_name", "gas"]:
        if k in cols_low: mapping[cols_low[k]] = "gas_name"
    for k in ["gas_id", "gas"]:
        if k in cols_low and "gas_name" not in mapping.values():
            mapping[cols_low[k]] = "gas_id"
    # quantity (CO2e or mass)
    for k in ["co2e_emissions", "emissions_co2e", "co2e_emission", "carbon_dioxide_eq", "co2_eq"]:
        if k in cols_low: mapping[cols_low[k]] = "qty_co2e"
    for k in ["ghg_quantity", "quantity", "emissions", "mass", "value"]:
        if k in cols_low and "qty_co2e" not in mapping.values():
            mapping[cols_low[k]] = "qty_raw"
    # units, subpart, sector
    for k in ["unit_name","unit","units"]:
        if k in cols_low: mapping[cols_low[k]] = "unit"
    for k in ["subpart_name","subpart","sub_part_id","subpart_description"]:
        if k in cols_low: mapping[cols_low[k]] = "subpart"
    for k in ["sector_name","sector","ghg_sector"]:
        if k in cols_low: mapping[cols_low[k]] = "sector_name"

    print("Emissions column mapping:", mapping)
    return df.rename(columns=mapping)

# ===================== Downloads (optional live pulls) =====================
def download_facility() -> pd.DataFrame:
    print("Downloading facility master…")
    df = ef_csv(T_FACILITY)
    safe_save(df, os.path.join(RAW_DIR, "pub_dim_facility.csv"))
    return df

def download_sector() -> pd.DataFrame:
    print("Downloading sector/national rollups…")
    df = ef_csv(T_SECTOR)
    safe_save(df, os.path.join(RAW_DIR, "pub_facts_sector_ghg_emission.csv"))
    return df

def discover_year_column(sample: pd.DataFrame) -> str:
    return pick_required("discover_year", set(sample.columns), "REPORTING_YEAR", "YEAR", "GHG_REPORTING_YEAR", "year")

def download_emissions_sharded(year_col: str) -> pd.DataFrame:
    print("Sampling emissions to get year range…")
    sample = ef_csv(T_EMIS_SUBP, rows=(0, 200))
    if not year_col:
        year_col = discover_year_column(sample)
    y = pd.to_numeric(sample[year_col], errors="coerce").dropna()
    if len(y) == 0:
        raise RuntimeError("Could not detect reporting years from emissions sample.")
    min_year, max_year = int(y.min()), int(y.max())
    print(f"Detected '{year_col}' range {min_year}–{max_year}. Downloading by year…")
    shards = []
    for yr in range(min_year, max_year + 1):
        try:
            dfy = ef_csv(T_EMIS_SUBP, where=[(year_col, str(yr))])
            shards.append(dfy)
            safe_save(dfy, os.path.join(RAW_DIR, f"pub_facts_subp_ghg_emission_{yr}.csv"))
        except Exception as e:
            print(f"  Year {yr} skipped: {e}")
    if shards:
        return pd.concat(shards, ignore_index=True)
    raise RuntimeError("Emissions shards empty. Check API availability.")

# ===================== Transforms =====================
def build_long_wide(fac: pd.DataFrame, emis: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Normalize columns for robust downstream logic
    fac_n = normalize_fac_cols(fac.copy())
    emis_n = normalize_emis_cols(emis.copy())

    # Required minimal set after normalization
    for req in ["fac_id", "year"]:
        if req not in emis_n.columns:
            raise KeyError(f"[emissions] Missing required column after normalization: '{req}'. "
                           f"Have: {sorted(emis_n.columns.tolist())[:25]} ...")

    # Identify gas key: prefer name; else id
    gas_key = "gas_name" if "gas_name" in emis_n.columns else ("gas_id" if "gas_id" in emis_n.columns else None)
    if gas_key is None:
        raise KeyError("[emissions] Missing gas identifier ('gas_name' or 'gas_id').")

    # Determine quantity columns
    has_co2e = "qty_co2e" in emis_n.columns
    has_raw  = "qty_raw" in emis_n.columns

    # Compute row-level CO2e
    emis_n["gas_key"] = emis_n[gas_key].astype(str)
    if gas_key == "gas_name":
        emis_n["gas_key"] = emis_n["gas_key"].str.strip().str.upper()

    if has_co2e:
        emis_n["co2e_row"] = pd.to_numeric(emis_n["qty_co2e"], errors="coerce").fillna(0.0)
    elif has_raw:
        # Apply GWP only if we have gas_name; if only IDs, leave as 0 unless you later map IDs to names
        if gas_key == "gas_name":
            emis_n["co2e_row"] = pd.to_numeric(emis_n["qty_raw"], errors="coerce").fillna(0.0) * \
                                 emis_n["gas_key"].map(lambda g: gwp_lookup(g))
        else:
            # Without a mapping for gas_id -> name/GWP, we cannot convert reliably.
            # Keep raw quantity as CO2e proxy = 0. (You can replace with your own mapping later.)
            emis_n["co2e_row"] = 0.0
    else:
        raise KeyError("[emissions] No usable quantity column: expected 'co2e' (qty_co2e) or raw mass (qty_raw).")

    # Build LONG: facility-year-gas_key
    long_keep = ["fac_id", "year", "gas_key", "co2e_row"]
    if "qty_raw" in emis_n.columns:    long_keep.append("qty_raw")
    if "qty_co2e" in emis_n.columns:   long_keep.append("qty_co2e")
    if "subpart" in emis_n.columns:    long_keep.append("subpart")
    if "sector_name" in emis_n.columns:long_keep.append("sector_name")

    long = emis_n[long_keep].copy()
    # Aggregate (sum) per facility-year-gas
    agg = {"co2e_row": "sum"}
    if "qty_raw" in long.columns:  agg["qty_raw"]  = "sum"
    if "qty_co2e" in long.columns: agg["qty_co2e"] = "sum"
    long_g = long.groupby(["fac_id", "year", "gas_key"], as_index=False).agg(agg)

    print(f"Built LONG format: {len(long_g):,} rows (facility-year-gas)")
    print("Sample LONG format data:")
    print(long_g.head())

    # Attach facility attributes
    fac_keep = [c for c in ["fac_id","fac_name","state","county","city","zip","lat","lon","frs_registry_id"] if c in fac_n.columns]
    if fac_keep:
        long_g = long_g.merge(fac_n[fac_keep].drop_duplicates(), on="fac_id", how="left")

    # Save LONG
    safe_save(long_g, os.path.join(OUT_DIR, "facility_year_gas_long.csv"))

    # Presence masks (per gas)
    presence = long_g.copy()
    presence["present"] = True
    mask_wide = presence.pivot_table(index=["fac_id","year"], columns="gas_key", values="present", aggfunc="max").fillna(False)
    mask_wide.columns = [f"mask_{str(c).lower()}" for c in mask_wide.columns]

    print("Sample gas presence mask data:")
    print(mask_wide.reset_index().head())   

    # Wide pivots
    if "qty_raw" in long_g.columns:
        qty_wide = long_g.pivot_table(index=["fac_id","year"], columns="gas_key", values="qty_raw", aggfunc="sum").fillna(0.0)
        qty_wide.columns = [f"qty_{str(c).lower()}" for c in qty_wide.columns]
    else:
        qty_wide = pd.DataFrame(index=mask_wide.index)

    co2e_wide = long_g.pivot_table(index=["fac_id","year"], columns="gas_key", values="co2e_row", aggfunc="sum").fillna(0.0)
    co2e_wide.columns = [f"co2e_{str(c).lower()}" for c in co2e_wide.columns]

    wide = pd.concat([qty_wide, co2e_wide, mask_wide], axis=1).reset_index()

    print(f"Built WIDE format: {len(wide):,} rows (facility-year)")
    print("Sample WIDE format data:")
    print(wide.head())
    # Total CO2e
    co2e_cols = [c for c in wide.columns if c.startswith("co2e_")]
    wide["co2e_total"] = wide[co2e_cols].sum(axis=1) if co2e_cols else 0.0

    # Attach facility attrs
    if "fac_name" in fac_n.columns:
        wide = wide.merge(fac_n[["fac_id","fac_name"]].drop_duplicates(), on="fac_id", how="left")
    for c in ["state","county","city","zip","lat","lon","frs_registry_id"]:
        if c in fac_n.columns:
            wide = wide.merge(fac_n[["fac_id",c]].drop_duplicates(), on="fac_id", how="left")
    wide = wide.loc[:,~wide.columns.duplicated()]
    safe_save(wide, os.path.join(OUT_DIR, "facility_year_wide.csv"))

    # Best-effort facility→sector map from emissions if present
    facility_sector_map = pd.DataFrame(columns=["fac_id","sector_name"])
    if "sector_name" in emis_n.columns:
        sec_mode = (emis_n[["fac_id","sector_name"]].dropna()
                    .groupby("fac_id")["sector_name"].agg(lambda s: s.value_counts().idxmax()).reset_index())
        facility_sector_map = sec_mode
        safe_save(facility_sector_map, os.path.join(OUT_DIR, "facility_sector_map.csv"))

    return long_g, wide, facility_sector_map

def tidy_sector(sector: pd.DataFrame) -> pd.DataFrame:
    # Normalize minimal columns
    s = sector.copy()
    cols_low = {c.lower(): c for c in s.columns}
    col_year = cols_low.get("reporting_year") or cols_low.get("year")
    col_sector = cols_low.get("sector_name") or cols_low.get("sector") or cols_low.get("ghg_sector") or cols_low.get("sector_id")
    col_qty = cols_low.get("emissions") or cols_low.get("co2e_emission") or cols_low.get("emissions_co2e") or cols_low.get("quantity")
    if not (col_year and col_sector and col_qty):
        # If sector table lacks these (rare), output empty tidy
        tidy = pd.DataFrame(columns=["year","sector_name","sector_co2e"])
    else:
        tidy = s[[col_year, col_sector, col_qty]].copy()
        tidy.columns = ["year", "sector_name", "sector_co2e"]
    safe_save(tidy, os.path.join(OUT_DIR, "sector_year_totals.csv"))
    return tidy

def add_engineered_features(wide: pd.DataFrame) -> pd.DataFrame:
    df = wide.copy()
    # Ensure types
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"])
    df = df.sort_values(["fac_id","year"])

    # Lags & rolling on co2e_total
    for k in LAGS:
        df[f"co2e_total_lag{k}"] = df.groupby("fac_id")["co2e_total"].shift(k)
    for w in ROLL_WINDOWS:
        df[f"roll{w}_mean_co2e"] = df.groupby("fac_id")["co2e_total"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        df[f"roll{w}_std_co2e"]  = df.groupby("fac_id")["co2e_total"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).std())

    # History depth
    def prior_count(s):
        v = s.notna().astype(int)
        return v.shift(1).fillna(0).cumsum()
    df["years_of_history"] = df.groupby("fac_id")["co2e_total"].transform(prior_count)

    # Stable integer encodings
    df["facility_idx"], _ = pd.factorize(df["fac_id"], sort=True)
    if "state" in df.columns:
        df["state_idx"], _ = pd.factorize(df["state"], sort=True)

    return df

def join_sector_features(train_view: pd.DataFrame,
                         facility_sector_map: pd.DataFrame,
                         sector_tidy: pd.DataFrame) -> pd.DataFrame:
    df = train_view.copy()
    if facility_sector_map is None or facility_sector_map.empty or sector_tidy is None or sector_tidy.empty:
        print("Sector features skipped (no mapping or sector totals).")
        return df
    df = df.merge(facility_sector_map, on="fac_id", how="left")
    df = df.merge(sector_tidy, on=["year","sector_name"], how="left")
    # Sector lags/rolling
    df = df.sort_values(["sector_name","year"])
    for k in [1,3]:
        df[f"sector_co2e_lag{k}"] = df.groupby("sector_name")["sector_co2e"].shift(k)
    df["sector_co2e_roll3_mean"] = df.groupby("sector_name")["sector_co2e"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    # Sector idx
    df["sector_idx"], _ = pd.factorize(df["sector_name"], sort=True)
    return df

def make_splits(train_view: pd.DataFrame):
    # collect years as plain Python ints
    years = pd.to_numeric(train_view["year"], errors="coerce").dropna().astype(int).drop_duplicates().tolist()
    years = sorted(int(y) for y in years)  # ensure built-in int, not numpy.int64

    if not years:
        raise RuntimeError("No years found for splitting.")

    test_years = SPLIT_CONFIG["test_years"]
    val_years  = SPLIT_CONFIG["val_years"]

    if test_years is None or val_years is None:
        if len(years) >= 5:
            test_years = [int(y) for y in years[-2:]]
            val_years  = [int(y) for y in years[-4:-2]]
            print(f"Auto-selected test years: {test_years}, val years: {val_years}")
        elif len(years) >= 3:
            test_years = [int(years[-1])]
            val_years  = [int(years[-2])]
            print(f"Auto-selected test year: {test_years}, val year: {val_years}")
        else:
            test_years = [int(years[-1])]
            val_years  = []
            print(f"Auto-selected test year: {test_years}, no val year (insufficient data).")

    # ensure all are plain ints
    train_years = [int(y) for y in years if y not in set(test_years + val_years)]
    splits = {
        "train": train_years,
        "val":   [int(y) for y in val_years],
        "test":  [int(y) for y in test_years],
    }

    with open(os.path.join(OUT_DIR, "splits_years.json"), "w") as f:
        json.dump(splits, f, indent=2)

    print("Wrote splits_years.json:", splits)
    return splits

# ===================== Main =====================
def main():
    # You can replace these with your **already-loaded** DataFrames if you’re not pulling via API.
    # fac = ef_csv(T_FACILITY)           # or pd.read_csv("path/to/your/facility.csv")
    fac = pd.read_csv(os.path.join(RAW_DIR, "pub_dim_facility.csv"))
    safe_save(fac, os.path.join(RAW_DIR, "pub_dim_facility.csv"))

    # sector = ef_csv(T_SECTOR)          # or pd.read_csv("path/to/your/sector.csv")
    sector = pd.read_csv(os.path.join(RAW_DIR, "pub_facts_sector_ghg_emission.csv"))
    safe_save(sector, os.path.join(RAW_DIR, "pub_facts_sector_ghg_emission.csv"))

    # Emissions: if you already have a file like ghgp_data_2010.csv with ['co2e_emission','facility_id','gas_id','sub_part_id','year'],
    # load it HERE instead of the API shards:
    # emis = pd.read_csv("ghgp_data_2010.csv")
    # Otherwise, pull from API (sharded):
    sample = ef_csv(T_EMIS_SUBP, rows=(0, 200))
    # to find year column robustly
    year_col = discover_year_column(sample)
    emis = download_emissions_sharded(year_col)

    # Save emissions raw (combined)
    safe_save(emis, os.path.join(RAW_DIR, "pub_facts_subp_ghg_emission_all.csv"))

    # Build long/wide and facility→sector map
    long_g, wide, facility_sector_map = build_long_wide(fac, emis)

    # Sector tidy
    sector_tidy = tidy_sector(sector)

    # Training view + engineered features
    train_view = add_engineered_features(wide)

    # Sector features into training view
    train_view = join_sector_features(train_view, facility_sector_map, sector_tidy)

    # Final save
    safe_save(train_view, os.path.join(OUT_DIR, "training_facility_year.csv"))

    # Splits
    make_splits(train_view)

    print("\nAll done. Outputs in:", os.path.abspath(OUT_DIR))

if __name__ == "__main__":   # <-- fixed guard (previous error showed _name_/_main_)
    main()