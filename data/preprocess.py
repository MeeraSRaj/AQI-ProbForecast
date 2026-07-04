"""
preprocess.py
-------------
Cleans and prepares the CPCB India Air Quality dataset for time series forecasting.

Dataset:  "Air Quality Data in India (2015–2020)" by rohanrao on Kaggle
File:     city_hour.csv
Download: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

Usage:
    python data/preprocess.py --city Delhi --output data/processed/delhi_aqi.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


# ─── Config ───────────────────────────────────────────────────────────────────

RAW_FILE    = Path("data/raw/city_hour.csv")
TARGET_COL  = "PM2.5"          # primary pollutant to forecast
FREQ        = "1h"             # target temporal resolution
MAX_CONSEC_GAP = 3             # max consecutive hours to forward-fill; longer = drop

# AQI category thresholds for PM2.5 (µg/m³) — India CPCB standard
AQI_BREAKS  = [0, 30, 60, 90, 120, 250, np.inf]
AQI_LABELS  = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


# ─── Step 1: Load ─────────────────────────────────────────────────────────────

def load_raw(city: str) -> pd.DataFrame:
    """Load and filter to a single city."""
    df = pd.read_csv(RAW_FILE, parse_dates=["Datetime"])
    df.columns = df.columns.str.strip()

    available = df["City"].unique()
    if city not in available:
        raise ValueError(
            f"City '{city}' not found.\nAvailable: {sorted(available)}"
        )

    df = df[df["City"] == city].copy()
    df = df.sort_values("Datetime").reset_index(drop=True)
    print(f"[load]  {len(df):,} rows for {city} "
          f"({df['Datetime'].min().date()} → {df['Datetime'].max().date()})")
    return df


# ─── Step 2: Reindex to strict hourly ─────────────────────────────────────────

def reindex_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force a complete hourly DatetimeIndex.
    Duplicate timestamps (if any) are averaged; missing hours get NaN.
    """
    df = df.set_index("Datetime")
    df = df[~df.index.duplicated(keep="first")]          # drop exact duplicates
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=FREQ)
    df = df.reindex(full_idx)
    df.index.name = "Datetime"
    print(f"[reindex]  {len(df):,} hourly slots "
          f"({df[TARGET_COL].isna().sum():,} missing in target)")
    return df


# ─── Step 3: Impute PM2.5 ─────────────────────────────────────────────────────

def impute_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill gaps ≤ MAX_CONSEC_GAP hours.
    Longer runs stay NaN and will be dropped before modelling.
    """
    s = df[TARGET_COL].copy()
    # mark run lengths of consecutive NaNs
    is_null = s.isna()
    run_id  = (is_null != is_null.shift()).cumsum()
    run_len = is_null.groupby(run_id).transform("sum")

    # only fill short gaps
    fillable = is_null & (run_len <= MAX_CONSEC_GAP)
    s[fillable] = s.ffill()[fillable]
    df[TARGET_COL] = s

    remaining = df[TARGET_COL].isna().sum()
    print(f"[impute]  {remaining:,} rows still missing after gap-fill "
          f"(will be dropped)")
    return df


# ─── Step 4: Feature engineering ──────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temporal features useful for both classical and deep-learning models.
    Cyclical encoding (sin/cos) prevents the model treating hour 23 as
    far from hour 0.
    """
    df = df.copy()
    idx = df.index

    # Raw temporal
    df["hour"]       = idx.hour
    df["dayofweek"]  = idx.dayofweek      # 0 = Monday
    df["month"]      = idx.month
    df["dayofyear"]  = idx.dayofyear

    # Cyclical encoding
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"]      / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"]      / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"]     / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"]     / 12)

    # Lag features (look-back in hours)
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        df[f"pm25_lag{lag}"] = df[TARGET_COL].shift(lag)

    # Rolling statistics (over past 24h)
    df["pm25_roll24_mean"] = df[TARGET_COL].shift(1).rolling(24).mean()
    df["pm25_roll24_std"]  = df[TARGET_COL].shift(1).rolling(24).std()

    # AQI category (for visualisation / threshold plots)
    df["aqi_category"] = pd.cut(
        df[TARGET_COL],
        bins=AQI_BREAKS,
        labels=AQI_LABELS,
        right=False
    )

    print(f"[features]  {len(df.columns)} columns total")
    return df


# ─── Step 5: Drop rows that cannot be used ────────────────────────────────────

def drop_unusable(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where the target or any lag feature is still NaN."""
    before = len(df)
    # Must have: target + all lag columns
    lag_cols = [c for c in df.columns if c.startswith("pm25_lag")]
    required = [TARGET_COL] + lag_cols
    df = df.dropna(subset=required)
    print(f"[drop]  {before - len(df):,} rows removed → {len(df):,} usable rows")
    return df


# ─── Step 6: Train / val / test split ─────────────────────────────────────────

def split_and_save(df: pd.DataFrame, output_path: Path) -> None:
    """
    Chronological 70 / 15 / 15 split.
    Saves the full processed file + three split CSVs.
    """
    n  = len(df)
    t1 = int(n * 0.70)
    t2 = int(n * 0.85)

    train = df.iloc[:t1]
    val   = df.iloc[t1:t2]
    test  = df.iloc[t2:]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)

    stem = output_path.stem
    train.to_csv(output_path.parent / f"{stem}_train.csv")
    val.to_csv(output_path.parent   / f"{stem}_val.csv")
    test.to_csv(output_path.parent  / f"{stem}_test.csv")

    print(f"\n[split]")
    print(f"  Train : {len(train):,} rows  "
          f"({train.index.min().date()} → {train.index.max().date()})")
    print(f"  Val   : {len(val):,} rows  "
          f"({val.index.min().date()} → {val.index.max().date()})")
    print(f"  Test  : {len(test):,} rows  "
          f"({test.index.min().date()} → {test.index.max().date()})")
    print(f"\n[save]  {output_path}")


# ─── Step 7: Quick sanity report ──────────────────────────────────────────────

def sanity_report(df: pd.DataFrame) -> None:
    pm = df[TARGET_COL]
    print(f"\n{'─'*45}")
    print(f"  PM2.5 summary  (µg/m³)")
    print(f"{'─'*45}")
    print(f"  Mean   : {pm.mean():.1f}")
    print(f"  Median : {pm.median():.1f}")
    print(f"  Std    : {pm.std():.1f}")
    print(f"  Min    : {pm.min():.1f}")
    print(f"  Max    : {pm.max():.1f}")
    print(f"  95th % : {pm.quantile(0.95):.1f}")
    print(f"\n  AQI category distribution:")
    counts = df["aqi_category"].value_counts()
    total  = len(df)
    for cat in AQI_LABELS:
        if cat in counts:
            pct = 100 * counts[cat] / total
            bar = "█" * int(pct / 2)
            print(f"    {cat:<15} {bar:<25} {pct:.1f}%")
    print(f"{'─'*45}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess India AQI data")
    parser.add_argument("--city",   default="Delhi",
                        help="City name (must match dataset exactly)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: data/processed/<city>_aqi.csv)")
    args = parser.parse_args()

    output = Path(args.output) if args.output else \
             Path(f"data/processed/{args.city.lower()}_aqi.csv")

    print(f"\n{'='*45}")
    print(f"  AQI Preprocessing — {args.city}")
    print(f"{'='*45}\n")

    df = load_raw(args.city)
    df = reindex_hourly(df)
    df = impute_target(df)
    df = add_features(df)
    df = drop_unusable(df)

    sanity_report(df)
    split_and_save(df, output)
    print("Done.\n")


if __name__ == "__main__":
    main()
