"""构建 5×5 尺寸-账面/盈利/投资组合并导出月度超额收益。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MONTH_FORMAT = "%Y-%m"
QUINTILES: Tuple[int, ...] = (1, 2, 3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-month",
        type=str,
        default="2010-01",
        help="样本起始月份 (YYYY-MM)",
    )
    parser.add_argument(
        "--end-month",
        type=str,
        default="2025-12",
        help="样本结束月份 (YYYY-MM)",
    )
    return parser.parse_args()


def build_month_list(start: str, end: str) -> List[str]:
    start_period = pd.Period(start, freq="M")
    end_period = pd.Period(end, freq="M")
    if start_period > end_period:
        raise ValueError("start-month 晚于 end-month")
    return pd.period_range(start_period, end_period, freq="M").strftime(MONTH_FORMAT).tolist()


def normalize_stkcd(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    values = values.where(values.notna(), None)
    values = values.str.replace(".SH", "", regex=False).str.replace(".SZ", "", regex=False)
    return values.str.zfill(6)


def normalize_month(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.to_period("M").dt.strftime(MONTH_FORMAT)


def load_monthly_frame(filename: str, keep_cols: Sequence[str], months: Iterable[str]) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / filename, columns=list(keep_cols))
    if "Stkcd" in df.columns:
        df["Stkcd"] = normalize_stkcd(df["Stkcd"])
    if "month" in df.columns:
        df["month"] = normalize_month(df["month"])
        df = df[df["month"].isin(months)]
    return df.reset_index(drop=True)


def assign_quintiles(df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
    def _per_month(series: pd.Series) -> pd.Series:
        result = pd.Series(data=np.nan, index=series.index, dtype="float64")
        valid = series.dropna()
        if valid.empty:
            return result
        ranks = valid.rank(method="first")
        buckets = np.floor((ranks - 1) / len(valid) * len(QUINTILES)).astype(int) + 1
        buckets = buckets.clip(1, len(QUINTILES))
        result.loc[valid.index] = buckets
        return result

    df[target_col] = (
        df.groupby("month", group_keys=False)[source_col].apply(_per_month).astype("Int64")
    )
    return df


def reindex_full_grid(
    df: pd.DataFrame,
    month_list: Sequence[str],
    metric_col: str,
) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [month_list, QUINTILES, QUINTILES],
        names=["month", "size_quintile", metric_col],
    )
    reindexed = df.set_index(["month", "size_quintile", metric_col]).reindex(idx).reset_index()
    reindexed["n_stocks"] = reindexed["n_stocks"].fillna(0).astype("Int64")
    return reindexed


def compute_portfolios(
    panel: pd.DataFrame,
    rf: pd.DataFrame,
    month_list: Sequence[str],
    metric_column: str,
    metric_quintile_col: str,
) -> pd.DataFrame:
    needed_cols = [
        "month",
        "Stkcd",
        "monthly_return",
        "size_quintile",
        metric_quintile_col,
    ]
    df = panel.dropna(subset=["size_quintile", metric_column]).copy()
    df = assign_quintiles(df, metric_column, metric_quintile_col)
    df = df.dropna(subset=[metric_quintile_col])
    df[metric_quintile_col] = df[metric_quintile_col].astype("Int64")
    df = df[needed_cols]

    grouped = (
        df.groupby(["month", "size_quintile", metric_quintile_col], as_index=False)
        .agg(monthly_return=("monthly_return", "mean"), n_stocks=("Stkcd", "nunique"))
    )
    merged = grouped.merge(rf, on="month", how="left")
    merged["excess_return"] = merged["monthly_return"] - merged["rf_rate"]
    merged = merged.drop(columns=["monthly_return", "rf_rate"])
    merged = reindex_full_grid(merged, month_list, metric_quintile_col)
    merged = merged.sort_values(["month", "size_quintile", metric_quintile_col]).reset_index(drop=True)
    return merged


def print_summary(df: pd.DataFrame, metric_col: str, label: str, output_path: Path) -> None:
    print(f"[{label}] 数据来自 data/processed/monthly_*.parquet, 写入 {output_path.relative_to(ROOT)}")
    combos = df.groupby(["size_quintile", metric_col])
    for (size_q, metric_q), group in combos:
        active = group[group["n_stocks"] > 0]
        if active.empty:
            print(f"  size={size_q}, {metric_col}={metric_q}: 无有效样本")
            continue
        months_with_data = active["month"]
        time_range = f"{months_with_data.min()}~{months_with_data.max()}"
        avg_n = active["n_stocks"].mean()
        print(f"  size={size_q}, {metric_col}={metric_q}: {time_range}, 平均成份数={avg_n:.1f}")


def main() -> None:
    args = parse_args()
    month_list = build_month_list(args.start_month, args.end_month)

    returns = load_monthly_frame(
        "monthly_returns.parquet",
        ["Stkcd", "month", "monthly_return"],
        month_list,
    )
    size = load_monthly_frame(
        "monthly_size.parquet",
        ["Stkcd", "month", "market_cap"],
        month_list,
    )
    financials = load_monthly_frame(
        "financials.parquet",
        ["Stkcd", "month", "book_value", "profitability_metric", "investment_metric"],
        month_list,
    )
    rf = load_monthly_frame("risk_free_monthly.parquet", ["month", "rf_rate"], month_list)
    rf = (
        rf.sort_values("month")
        .drop_duplicates("month", keep="last")
        .set_index("month")
        .reindex(month_list)
        .ffill()
        .bfill()
        .reset_index()
    )

    panel = (
        returns.merge(size, on=["Stkcd", "month"], how="inner")
        .merge(financials, on=["Stkcd", "month"], how="inner")
    )
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel.loc[panel["market_cap"] <= 0, "market_cap"] = np.nan
    panel = panel.dropna(subset=["monthly_return", "market_cap"])
    panel = assign_quintiles(panel, "market_cap", "size_quintile")
    panel = panel.dropna(subset=["size_quintile"]).copy()
    panel["size_quintile"] = panel["size_quintile"].astype("Int64")
    panel["bm_ratio"] = panel["book_value"] / panel["market_cap"]
    panel = panel.replace([np.inf, -np.inf], np.nan)

    specs = [
        ("bm_ratio", "bm_quintile", PROCESSED_DIR / "portfolio_returns_size_bm.parquet", "Size-B/M"),
        (
            "profitability_metric",
            "op_quintile",
            PROCESSED_DIR / "portfolio_returns_size_op.parquet",
            "Size-OP",
        ),
        (
            "investment_metric",
            "inv_quintile",
            PROCESSED_DIR / "portfolio_returns_size_inv.parquet",
            "Size-Inv",
        ),
    ]

    for metric_col, quintile_col, output_path, label in specs:
        portfolio_df = compute_portfolios(panel.copy(), rf, month_list, metric_col, quintile_col)
        portfolio_df.to_parquet(output_path, index=False)
        print_summary(portfolio_df, quintile_col, label, output_path)


if __name__ == "__main__":
    main()
