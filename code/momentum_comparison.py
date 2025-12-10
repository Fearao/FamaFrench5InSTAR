"""Compare different momentum lookback windows via regressions."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
DOCS_DIR = Path(__file__).parent.parent / "docs"
FIVE_FACTOR_PATH = DATA_DIR / "factor_returns_f5.parquet"
WEEKLY_PANEL_PATH = DATA_DIR / "weekly_panel.parquet"


@dataclass
class MomentumConfig:
    label: str
    window: int
    skip: int
    min_periods: int | None = None


MOMENTUM_CONFIGS: List[MomentumConfig] = [
    MomentumConfig(label="mom_w4_s1", window=4, skip=1, min_periods=3),
    MomentumConfig(label="mom_w8_s1", window=8, skip=1, min_periods=5),
    MomentumConfig(label="mom_w16_s2", window=16, skip=2, min_periods=8),
    MomentumConfig(label="mom_w26_s4", window=26, skip=4),
    MomentumConfig(label="mom_w52_s4", window=52, skip=4),
    MomentumConfig(label="mom_w78_s4", window=78, skip=4),
]


def _load_weekly_panel() -> pd.DataFrame:
    cols = ["stock_id", "trade_date", "weekly_return", "market_cap"]
    df = pd.read_parquet(WEEKLY_PANEL_PATH, columns=cols)
    df = df.dropna(subset=["weekly_return", "market_cap"])
    df = df.sort_values(["stock_id", "trade_date"])
    return df


def _value_weighted_return(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    weights = df["market_cap"].clip(lower=0)
    denom = weights.sum()
    if denom <= 0:
        return np.nan
    return float((df["weekly_return"] * weights).sum() / denom)


def _cross_sectional_quintiles(values: pd.Series) -> pd.Series | None:
    labels = [f"Q{i}" for i in range(1, 6)]
    try:
        buckets = pd.qcut(values, q=5, labels=labels, duplicates="drop")
    except ValueError:
        return None
    unique = set(buckets.dropna().unique())
    if "Q1" not in unique or "Q5" not in unique:
        return None
    return buckets


def build_momentum_factor(panel: pd.DataFrame, cfg: MomentumConfig) -> pd.DataFrame:
    df = panel.copy()
    df["log_ret"] = np.log1p(df["weekly_return"].clip(lower=-0.95))
    grouped = df.groupby("stock_id", group_keys=False)
    min_periods = cfg.min_periods or max(4, int(cfg.window * 0.8))
    rolling = (
        grouped["log_ret"]
        .rolling(window=cfg.window, min_periods=min_periods)
        .sum()
        .shift(cfg.skip)
        .reset_index(level=0, drop=True)
    )
    df[f"signal_{cfg.label}"] = np.expm1(rolling)
    df = df.dropna(subset=[f"signal_{cfg.label}"])

    rows = []
    for trade_date, group in df.groupby("trade_date", sort=True):
        working = group[group["market_cap"] > 0]
        buckets = _cross_sectional_quintiles(working[f"signal_{cfg.label}"])
        if buckets is None:
            rows.append({"trade_date": trade_date, cfg.label: np.nan})
            continue
        working = working.assign(bucket=buckets.values)
        high = working[working["bucket"] == "Q5"]
        low = working[working["bucket"] == "Q1"]
        mom_value = _value_weighted_return(high) - _value_weighted_return(low)
        rows.append({"trade_date": trade_date, cfg.label: mom_value})

    factor = pd.DataFrame(rows).sort_values("trade_date")
    out_path = DATA_DIR / f"factor_returns_{cfg.label}.parquet"
    factor.to_parquet(out_path, index=False)
    print(f"Saved {out_path} with {len(factor)} rows")
    return factor


def _load_portfolio_returns() -> pd.DataFrame:
    weekly = pd.read_parquet(WEEKLY_PANEL_PATH, columns=["trade_date", "weekly_return"])
    weekly = weekly.dropna(subset=["weekly_return"])
    portfolio = (
        weekly.groupby("trade_date")["weekly_return"].mean().reset_index().rename(
            columns={"weekly_return": "portfolio_return"}
        )
    )
    return portfolio


def run_regression(momentum: pd.DataFrame, momentum_col: str, label: str) -> dict:
    factors = pd.read_parquet(FIVE_FACTOR_PATH)
    merged_factors = factors.merge(momentum, on="trade_date", how="inner")

    portfolio = _load_portfolio_returns()
    frame = portfolio.merge(merged_factors, on="trade_date", how="inner").sort_values("trade_date")
    frame["excess_portfolio"] = frame["portfolio_return"] - frame["RF"]
    frame = frame.dropna(subset=[momentum_col])

    cols = ["MKT_RF", "SMB", "HML", "RMW", "CMA", momentum_col]
    X = sm.add_constant(frame[cols])
    model = sm.OLS(frame["excess_portfolio"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})

    summary = pd.DataFrame(
        {
            "term": model.params.index,
            "coef": model.params.values,
            "tvalue": model.tvalues,
            "pvalue": model.pvalues,
        }
    )
    meta = {
        "label": label,
        "n_obs": int(model.nobs),
        "r_squared": float(model.rsquared),
        "alpha": float(model.params.get("const", np.nan)),
        "alpha_t": float(model.tvalues.get("const", np.nan)),
        "mom_coef": float(model.params.get(momentum_col, np.nan)),
        "mom_t": float(model.tvalues.get(momentum_col, np.nan)),
    }

    DOCS_DIR.mkdir(exist_ok=True)
    summary_path = DOCS_DIR / f"{label}_coeffs.csv"
    meta_path = DOCS_DIR / f"{label}_meta.json"
    summary.to_csv(summary_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved regression outputs for {label}")
    return meta


def main() -> None:
    panel = _load_weekly_panel()
    momentum_tables = []
    regression_meta = []

    for cfg in MOMENTUM_CONFIGS:
        factor = build_momentum_factor(panel, cfg)
        meta = run_regression(factor, cfg.label, cfg.label)
        momentum_tables.append(factor.rename(columns={cfg.label: cfg.label.upper()}))
        meta["window"] = cfg.window
        meta["skip"] = cfg.skip
        regression_meta.append(meta)

    # Equal-weight aggregate
    agg = momentum_tables[0][["trade_date"]]
    for tbl in momentum_tables:
        agg = agg.merge(tbl, on="trade_date", how="inner")
    mom_cols = [cfg.label.upper() for cfg in MOMENTUM_CONFIGS]
    agg["MOM_EQ"] = agg[mom_cols].mean(axis=1)
    agg_result = agg[["trade_date", "MOM_EQ"]]
    meta_eq = run_regression(agg_result, "MOM_EQ", "MOM_EQ")
    meta_eq["window"] = "avg"
    meta_eq["skip"] = "avg"
    regression_meta.append(meta_eq)

    # Document summary
    summary_path = DOCS_DIR / "momentum_comparison.md"
    lines = ["# Momentum Window Comparison", "", "| Config | Window | Skip | R² | α | MOM Coef | MOM t |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for meta in regression_meta:
        lines.append(
            f"| {meta['label']} | {meta['window']} | {meta['skip']} | "
            f"{meta['r_squared']:.3f} | {meta['alpha']:.4e} | {meta['mom_coef']:.4f} | {meta['mom_t']:.2f} |"
        )
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
