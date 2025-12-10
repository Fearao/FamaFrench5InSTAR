"""Utility script to refresh correlation and stationarity stats for factors."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
DOCS_DIR = Path(__file__).parent.parent / "docs"

FIVE_FACTOR_FILE = DATA_DIR / "factor_returns_f5.parquet"
MOM_CONFIGS = [
    "mom_w4_s1",
    "mom_w8_s1",
    "mom_w16_s2",
    "mom_w26_s4",
    "mom_w52_s4",
    "mom_w78_s4",
]


def load_factors() -> pd.DataFrame:
    df = pd.read_parquet(FIVE_FACTOR_FILE)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def build_mom_eq() -> pd.DataFrame:
    combined = None
    cols = []
    for cfg in MOM_CONFIGS:
        path = DATA_DIR / f"factor_returns_{cfg}.parquet"
        if not path.exists():
            continue
        cur = pd.read_parquet(path)
        cur["trade_date"] = pd.to_datetime(cur["trade_date"])
        col = cfg.upper()
        cur = cur.rename(columns={cfg: col})
        cols.append(col)
        combined = cur if combined is None else combined.merge(cur, on="trade_date", how="inner")
    if combined is None:
        raise FileNotFoundError("No momentum factor files found. Run momentum_comparison.py first.")
    combined["MOM_EQ"] = combined[cols].mean(axis=1, skipna=True)
    return combined[["trade_date", "MOM_EQ"]]


def main() -> None:
    five = load_factors()
    mom_eq = build_mom_eq()
    merged = five.merge(mom_eq, on="trade_date", how="inner").dropna(subset=["MOM_EQ"])

    cols = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM_EQ"]
    corr = merged[cols].corr().round(6)
    corr_path = DOCS_DIR / "factor_correlations.csv"
    corr.to_csv(corr_path)

    rows = []
    for col in cols:
        series = merged[col].dropna()
        stat, pvalue, _, _, crit, _ = adfuller(series, autolag="AIC")
        rows.append(
            {
                "factor": col,
                "adf_stat": stat,
                "pvalue": pvalue,
                "crit_1pct": crit["1%"],
                "crit_5pct": crit["5%"],
                "crit_10pct": crit["10%"],
            }
        )
    stationarity_path = DOCS_DIR / "factor_stationarity.csv"
    pd.DataFrame(rows).to_csv(stationarity_path, index=False)

    print(f"Saved correlation matrix to {corr_path}")
    print(f"Saved stationarity stats to {stationarity_path}")
    print(
        f"Observations: {len(merged)} weeks, from {merged['trade_date'].min().date()} "
        f"to {merged['trade_date'].max().date()}"
    )


if __name__ == "__main__":
    main()
