import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

PRIMARY = "#4C72B0"      # Blue
SECONDARY = "#55A868"    # Green
ACCENT = "#C44E52"       # Red
GRID = "#DDDDDD"

plt.rcParams.update({
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.alpha": 0.3,
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 12
})

COLOR_CYCLE = [
    "#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974",
    "#64B5CD", "#937860", "#8C8C8C", "#DA8BC3", "#B0CF5B"
]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=COLOR_CYCLE)

output_dir = "eda_output"
os.makedirs(output_dir, exist_ok=True)

# ============================================
# Load dataset
# ============================================

df = pd.read_csv("training_facility_year.csv")
target_col = "co2e_total"

gas_cols = [c for c in df.columns if c.startswith("co2e_") and c.endswith(".0")]

print("\nDataset Shape:", df.shape)
print("\nFirst Rows:")
print(df.head())

# =====================================================
# 1. MISSING VALUE SUMMARY BEFORE CLEANING
# =====================================================

print("\n==============================")
print(" MISSING VALUE SUMMARY (BEFORE)")
print("==============================")

total_missing_before = df.isna().sum().sum()
print(f"\nTotal missing values BEFORE cleaning: {total_missing_before:,}")

missing_per_col_before = df.isna().sum()
missing_per_col_before = (
    missing_per_col_before[missing_per_col_before > 0]
    .sort_values(ascending=False)
)

print("\nColumns with missing values BEFORE cleaning:")
print(missing_per_col_before)

# =====================================================
# 2. FILL ALL MISSING VALUES
#    - Numeric: 0
#    - Categorical: "UNKNOWN"
# =====================================================

print("\nApplying BEST missing value strategy: filling numeric with 0 and categoricals with 'UNKNOWN'...")

numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(exclude=[np.number]).columns

# Fill numeric values with 0
df[numeric_cols] = df[numeric_cols].fillna(0)

# Fill ALL categorical values with 'UNKNOWN'
for col in categorical_cols:
    df[col] = df[col].fillna("UNKNOWN")

# =====================================================
# 3. MISSING VALUE SUMMARY AFTER CLEANING
# =====================================================

print("\n=============================")
print(" MISSING VALUE SUMMARY (AFTER)")
print("=============================")

total_missing_after = df.isna().sum().sum()
print(f"\nTotal missing values AFTER cleaning: {total_missing_after:,}")

missing_per_col_after = df.isna().sum()
missing_per_col_after = missing_per_col_after[missing_per_col_after > 0]

if missing_per_col_after.empty:
    print("\nAll missing values have been successfully handled (no NaNs remain).")
else:
    print("\nColumns still containing missing values AFTER cleaning:")
    print(missing_per_col_after)

# =====================================================
# 4. FULL NUMERIC SUMMARY STATISTICS
# =====================================================

print("\n=====================================")
print(" FULL NUMERIC SUMMARY STATISTICS")
print("=====================================")

summary_df = pd.DataFrame({
    "count": df[numeric_cols].count(),
    "missing": df[numeric_cols].isna().sum(),
    "mean": df[numeric_cols].mean(),
    "median": df[numeric_cols].median(),
    "mode": df[numeric_cols].mode().iloc[0],   # mode per column
    "std": df[numeric_cols].std(),
    "min": df[numeric_cols].min(),
    "p25": df[numeric_cols].quantile(0.25),
    "p50": df[numeric_cols].quantile(0.50),
    "p75": df[numeric_cols].quantile(0.75),
    "max": df[numeric_cols].max()
})

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)

print("\nNumeric summary statistics (first 20 columns):")
print(summary_df.head(20))

# =====================================================
# 5. PLOTS
# =====================================================

# 5A. Distribution of Total Emissions
plt.figure(figsize=(7, 5))
plt.hist(df[target_col], bins=50, color=PRIMARY, edgecolor="black")
plt.title("Distribution of Total CO2e (Linear)")
plt.xlabel("Total CO2e")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{output_dir}/co2e_distribution_linear.png")
plt.show()

plt.figure(figsize=(7, 5))
plt.hist(df[target_col], bins=50, log=True, color=PRIMARY, edgecolor="black")
plt.title("Distribution of Total CO2e (Log Scale)")
plt.xlabel("Total CO2e")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{output_dir}/co2e_distribution_log.png")
plt.show()

# 5B. Yearly Emission Trend
yearly_totals = df.groupby("year")[target_col].sum().reset_index()

plt.figure(figsize=(8, 5))
plt.plot(yearly_totals["year"], yearly_totals[target_col], marker="o")
plt.title("Total CO2e by Year")
plt.xlabel("Year")
plt.ylabel("Total CO2e")
plt.tight_layout()
plt.savefig(f"{output_dir}/co2e_trend_by_year.png")
plt.show()

# 5C. Top 10 Facilities
fac_totals = (
    df.groupby("fac_id")[target_col]
    .sum()
    .reset_index()
    .sort_values(target_col, ascending=False)
)

top10_fac = fac_totals.head(10)

plt.figure(figsize=(9, 5))
plt.barh(top10_fac["fac_id"].astype(str), top10_fac[target_col],
         color=SECONDARY, edgecolor="black")
plt.gca().invert_yaxis()
plt.title("Top 10 Highest-Emitting Facilities")
plt.xlabel("Total CO2e")
plt.tight_layout()
plt.savefig(f"{output_dir}/top10_facilities.png")
plt.show()

# 5D. Top 5 Facility Emission Trajectories
top5_ids = top10_fac["fac_id"].head(5).tolist()
df_top5 = df[df["fac_id"].isin(top5_ids)]

plt.figure(figsize=(10, 6))
for fac_id, sub in df_top5.groupby("fac_id"):
    sub = sub.sort_values("year")
    plt.plot(sub["year"], sub[target_col], marker="o", label=f"Facility {fac_id}")

plt.title("Emission Trajectories for Top 5 Facilities")
plt.xlabel("Year")
plt.ylabel("Total CO2e")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/top5_facility_trends.png")
plt.show()

# 5E. Sector-Level EDA
if "sector_name" in df.columns:

    df_sec = df[df["sector_name"] != "UNKNOWN"]

    # Top sectors
    sector_totals = (
        df_sec.groupby("sector_name")[target_col]
        .sum()
        .reset_index()
        .sort_values(target_col, ascending=False)
    )

    top_sectors = sector_totals.head(10)

    plt.figure(figsize=(10, 5))
    plt.barh(top_sectors["sector_name"], top_sectors[target_col],
             color=SECONDARY, edgecolor="black")
    plt.gca().invert_yaxis()
    plt.title("Top 10 Sectors by CO2e")
    plt.xlabel("Total CO2e")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top10_sectors.png")
    plt.show()

    # Multi-colored trends for top 5 sectors
    top5_sector_names = top_sectors["sector_name"].head(5).tolist()
    df_top_sec = df_sec[df_sec["sector_name"].isin(top5_sector_names)]

    sector_year_totals = (
        df_top_sec.groupby(["year", "sector_name"])[target_col]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))
    for sector, sub in sector_year_totals.groupby("sector_name"):
        sub = sub.sort_values("year")
        plt.plot(sub["year"], sub[target_col], marker="o", label=sector)

    plt.title("Emission Trends for Top 5 Sectors")
    plt.xlabel("Year")
    plt.ylabel("Total CO2e")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sector_trends_multicolor.png")
    plt.show()

# 5F. Gas-Level EDA
if gas_cols:
    gas_totals = df[gas_cols].sum().sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(gas_totals.index, gas_totals.values,
            color=PRIMARY, edgecolor="black")
    plt.xticks(rotation=45, ha="right")
    plt.title("Total Emissions by Gas Category")
    plt.ylabel("Total CO2e")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gas_totals.png")
    plt.show()

    gas_share = (gas_totals / gas_totals.sum()) * 100

    plt.figure(figsize=(10, 5))
    plt.bar(gas_share.index, gas_share.values,
            color=SECONDARY, edgecolor="black")
    plt.xticks(rotation=45, ha="right")
    plt.title("Gas Category Percent Share")
    plt.ylabel("Percent of Total")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gas_share.png")
    plt.show()

# 5G. Correlation Heatmap
corr_features = [
    "co2e_total", "co2e_total_lag1", "co2e_total_lag3",
    "co2e_total_lag5", "roll3_mean_co2e",
    "roll3_std_co2e", "years_of_history", "sector_co2e"
]

corr_features = [c for c in corr_features if c in df.columns]

if len(corr_features) >= 2:
    corr_matrix = df[corr_features].corr(method="pearson", min_periods=50)

    plt.figure(figsize=(9, 7))
    im = plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(len(corr_features)), corr_features, rotation=45, ha="right")
    plt.yticks(range(len(corr_features)), corr_features)
    plt.title("Correlation Heatmap (Key Features)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.show()

# =====================================================
# 6. facility_sector_map.csv plot (optional)
# =====================================================

try:
    df_sector_map = pd.read_csv("facility_sector_map.csv")

    if "sector_name" in df_sector_map.columns:
        sector_counts = df_sector_map["sector_name"].value_counts()

        plt.figure(figsize=(14, 6))
        plt.bar(sector_counts.index, sector_counts.values,
                color=PRIMARY, edgecolor="black")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Count")
        plt.title("Facility Count by Sector (facility_sector_map)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sector_map_distribution.png")
        plt.show()

except FileNotFoundError:
    print("facility_sector_map.csv not found.")
