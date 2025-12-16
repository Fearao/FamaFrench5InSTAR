"""从 processed parquet 生成月度收益/规模/财务/无风险利率面板。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed"
DEFAULT_OUTPUT = ROOT / "data" / "processed"
ISO_WEEK_SUFFIX = "-5"  # 使用周五代表周末


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help="输入 parquet 目录（默认 data/processed）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="输出 parquet 目录（默认 data/processed）",
    )
    parser.add_argument(
        "--start-month",
        type=str,
        default="2019-07",
        help="面板起始月份 (YYYY-MM)",
    )
    parser.add_argument(
        "--end-month",
        type=str,
        default="2025-12",
        help="面板结束月份 (YYYY-MM)",
    )
    return parser.parse_args()


def normalize_stkcd(series: pd.Series) -> pd.Series:
    values = series.astype("string")
    values = values.where(values.notna(), None)
    values = values.str.strip()
    values = values.str.replace(".SH", "", regex=False).str.replace(".SZ", "", regex=False)
    return values.str.zfill(6)


def build_month_range(start: str, end: str) -> pd.DatetimeIndex:
    start_period = pd.Period(start, freq="M")
    end_period = pd.Period(end, freq="M")
    if start_period > end_period:
        raise ValueError("start-month 晚于 end-month")
    return pd.period_range(start_period, end_period, freq="M").to_timestamp("M")


def parse_week_end(trdwnt: pd.Series) -> pd.Series:
    values = trdwnt.astype(str).str.strip() + ISO_WEEK_SUFFIX
    return pd.to_datetime(values, format="%G-%V-%u", errors="coerce")


def month_str(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.strftime("%Y-%m")


def compound_returns(values: pd.Series) -> float:
    return float(np.prod(1.0 + values) - 1.0)


def compute_monthly_returns(size_df: pd.DataFrame, month_range: pd.DatetimeIndex) -> pd.DataFrame:
    df = size_df.dropna(subset=["Wretwd"]).copy()
    df["week_end"] = parse_week_end(df["Trdwnt"])
    df = df.dropna(subset=["week_end"])
    df["month_ts"] = df["week_end"].dt.to_period("M").dt.to_timestamp("M")
    df = df[(df["month_ts"] >= month_range.min()) & (df["month_ts"] <= month_range.max())]
    grouped = df.groupby(["Stkcd", "month_ts"], as_index=False)["Wretwd"].agg(compound_returns)
    grouped = grouped.rename(columns={"month_ts": "month", "Wretwd": "monthly_return"})
    grouped["month"] = month_str(grouped["month"])
    return grouped


def compute_monthly_size(size_df: pd.DataFrame, month_range: pd.DatetimeIndex) -> pd.DataFrame:
    df = size_df.copy()
    df["week_end"] = parse_week_end(df["Trdwnt"])
    df = df.dropna(subset=["week_end"])
    df["month_ts"] = df["week_end"].dt.to_period("M").dt.to_timestamp("M")
    df = df[(df["month_ts"] >= month_range.min()) & (df["month_ts"] <= month_range.max())]
    last_week = (
        df.sort_values(["Stkcd", "month_ts", "week_end"]).groupby(["Stkcd", "month_ts"], as_index=False).tail(1)
    )
    result = last_week[["Stkcd", "month_ts", "Wsmvttl"]].rename(columns={"month_ts": "month", "Wsmvttl": "market_cap"})
    result["month"] = month_str(result["month"])
    return result


def _dedup_by_period(df: pd.DataFrame, sort_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(sort_cols).drop_duplicates(["Stkcd", "Accper"], keep="last")


def load_financial_tables(input_dir: Path) -> pd.DataFrame:
    book = pd.read_parquet(input_dir / "book_to_market.parquet")
    book = book[book["Typrep"] == "A"].copy()
    book["Accper"] = pd.to_datetime(book["Accper"])
    book = _dedup_by_period(book, ["Stkcd", "Accper"])
    book = book.rename(columns={"A003000000": "book_value"})[["Stkcd", "Accper", "book_value"]]
    book["Stkcd"] = normalize_stkcd(book["Stkcd"])
    book = book.dropna(subset=["Stkcd"])

    prof = pd.read_parquet(input_dir / "profitability.parquet")
    prof["Accper"] = pd.to_datetime(prof["Accper"])
    prof = _dedup_by_period(prof, ["Stkcd", "Accper", "Annodt"])
    prof = prof.rename(columns={"B130101": "profitability_metric"})[["Stkcd", "Accper", "profitability_metric"]]
    prof["Stkcd"] = normalize_stkcd(prof["Stkcd"])
    prof = prof.dropna(subset=["Stkcd"])

    inv = pd.read_parquet(input_dir / "investment.parquet")
    inv = inv[inv["Typrep"] == "A"].copy()
    inv["Accper"] = pd.to_datetime(inv["Accper"])
    inv = _dedup_by_period(inv, ["Stkcd", "Accper"])
    metric_candidates = [col for col in ["F080602A", "F080603A", "F080601A"] if col in inv.columns]
    if not metric_candidates:
        raise ValueError("investment.parquet 中缺少 F08060xA 指标列")
    metric_col = max(metric_candidates, key=lambda col: inv[col].notna().sum())
    inv = inv.rename(columns={metric_col: "investment_metric"})[["Stkcd", "Accper", "investment_metric"]]
    inv["Stkcd"] = normalize_stkcd(inv["Stkcd"])
    inv = inv.dropna(subset=["Stkcd"])

    merged = book.merge(prof, how="outer", on=["Stkcd", "Accper"])
    merged = merged.merge(inv, how="outer", on=["Stkcd", "Accper"])
    merged = merged.dropna(subset=["Accper"])
    merged["Stkcd"] = normalize_stkcd(merged["Stkcd"]).astype("string")
    merged = merged.dropna(subset=["Stkcd"])
    merged["available_month"] = merged["Accper"] + MonthEnd(6)
    merged = merged.dropna(subset=["available_month"])
    return merged.sort_values(["Stkcd", "available_month"])


def build_financial_panel(
    financials: pd.DataFrame,
    month_range: pd.DatetimeIndex,
    universe: Iterable[str],
) -> pd.DataFrame:
    columns = ["Stkcd", "month", "book_value", "profitability_metric", "investment_metric"]
    if financials.empty:
        return pd.DataFrame(columns=columns)

    month_template = pd.DataFrame({"month": month_range})
    stkcd_list = sorted({str(code) for code in universe if pd.notna(code)})
    aligned_frames = []
    needed_cols = ["available_month", "book_value", "profitability_metric", "investment_metric"]

    for stkcd in stkcd_list:
        grp = financials[financials["Stkcd"] == stkcd]
        if grp.empty:
            continue
        grp = grp.sort_values("available_month")
        aligned = pd.merge_asof(
            month_template,
            grp[needed_cols],
            left_on="month",
            right_on="available_month",
            direction="backward",
            allow_exact_matches=True,
        )
        aligned = aligned.drop(columns=["available_month"])
        aligned = aligned.dropna(subset=["book_value", "profitability_metric", "investment_metric"], how="all")
        if aligned.empty:
            continue
        aligned = aligned.copy()
        aligned["Stkcd"] = stkcd
        aligned_frames.append(aligned)

    if not aligned_frames:
        return pd.DataFrame(columns=columns)

    combined = pd.concat(aligned_frames, ignore_index=True)
    combined["month"] = month_str(combined["month"])
    return combined[columns]


def build_risk_free(input_dir: Path, month_range: pd.DatetimeIndex) -> pd.DataFrame:
    rf = pd.read_parquet(input_dir / "risk_free_rates.parquet")
    rf["ChangeDate"] = pd.to_datetime(rf["ChangeDate"])
    rf = rf.sort_values("ChangeDate")
    rf["rf_rate"] = rf["LumpFixed3Month"] / 100.0

    month_frame = pd.DataFrame({"month": month_range})
    monthly = pd.merge_asof(
        month_frame,
        rf[["ChangeDate", "rf_rate"]],
        left_on="month",
        right_on="ChangeDate",
        direction="backward",
        allow_exact_matches=True,
    )
    monthly = monthly.drop(columns=["ChangeDate"])

    # 回填历史缺失期间（2019-07 至 2022-08）的无风险利率
    # 使用固定假设值 1.35% 年化 = 0.0135
    monthly["rf_rate"] = monthly["rf_rate"].fillna(0.0135)

    monthly["month"] = month_str(monthly["month"])
    return monthly


def summarize(label: str, df: pd.DataFrame) -> None:
    if df.empty:
        print(f"{label}: 0 行，无有效月份")
        return
    months = pd.to_datetime(df["month"], format="%Y-%m")
    print(f"{label}: {len(df):,} 行，范围 {months.min().strftime('%Y-%m')} 至 {months.max().strftime('%Y-%m')}")


def main() -> None:
    args = parse_args()
    month_range = build_month_range(args.start_month, args.end_month)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    size_df = pd.read_parquet(args.input_dir / "size_weekly.parquet")
    size_df["Stkcd"] = normalize_stkcd(size_df["Stkcd"])

    monthly_returns = compute_monthly_returns(size_df, month_range)
    monthly_size = compute_monthly_size(size_df, month_range)

    monthly_returns.to_parquet(args.output_dir / "monthly_returns.parquet", index=False)
    monthly_size.to_parquet(args.output_dir / "monthly_size.parquet", index=False)

    financials = load_financial_tables(args.input_dir)
    universe = set(size_df["Stkcd"].dropna().unique()).union(set(financials["Stkcd"].dropna().unique()))
    financial_panel = build_financial_panel(financials, month_range, universe)
    financial_panel.to_parquet(args.output_dir / "financials.parquet", index=False)

    risk_free = build_risk_free(args.input_dir, month_range)
    risk_free.to_parquet(args.output_dir / "risk_free_monthly.parquet", index=False)

    summarize("monthly_returns", monthly_returns)
    summarize("monthly_size", monthly_size)
    summarize("financials", financial_panel)
    summarize("risk_free_monthly", risk_free)


if __name__ == "__main__":
    main()
