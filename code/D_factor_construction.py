"""构建 Fama-French 五因子 (MKT-RF, SMB, HML, RMW, CMA)。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_OUTPUT = PROCESSED_DIR / "factor_returns_ff5.parquet"
MONTH_FORMAT = "%Y-%m"
SIZE_LABELS = ("Small", "Big")
METRIC_BUCKETS = ("Low", "Medium", "High")
FACTOR_COLUMNS = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="输入 parquet 目录（默认 data/processed）",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="输出文件路径（默认 data/processed/factor_returns_ff5.parquet）",
    )
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
    parser.add_argument(
        "--summary",
        action="store_true",
        help="写入后打印因子均值/标准差/覆盖月数摘要",
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


def load_monthly_frame(path: Path, keep_cols: Sequence[str], months: Iterable[str]) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=list(keep_cols))
    if "Stkcd" in df.columns:
        df["Stkcd"] = normalize_stkcd(df["Stkcd"])
    if "month" in df.columns:
        df["month"] = normalize_month(df["month"])
        month_set = set(months)
        df = df[df["month"].isin(month_set)]
    return df.reset_index(drop=True)


def prepare_risk_free(path: Path, months: Sequence[str]) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(path)
    rf = pd.read_parquet(path, columns=["month", "rf_rate"])
    rf["month"] = normalize_month(rf["month"])
    rf = (
        rf.sort_values("month")
        .drop_duplicates("month", keep="last")
        .set_index("month")
    )
    rf = rf.reindex(months).ffill().bfill()
    return rf["rf_rate"].to_dict()


def mean_if_any(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return np.nan
    return float(values.mean())


def assign_size_group(series: pd.Series) -> pd.Series:
    result = pd.Series(data=None, index=series.index, dtype=object)
    valid = series.dropna()
    if valid.empty:
        return result
    median = valid.median()
    if pd.isna(median):
        return result
    result.loc[series <= median] = SIZE_LABELS[0]
    result.loc[series > median] = SIZE_LABELS[1]
    return result


def assign_metric_bucket(values: pd.Series) -> pd.Series:
    result = pd.Series(data=None, index=values.index, dtype=object)
    valid = values.dropna()
    if len(valid) < 3:
        return result
    low = float(np.nanquantile(valid, 0.3))
    high = float(np.nanquantile(valid, 0.7))
    if not np.isfinite(low) or not np.isfinite(high):
        return result
    result.loc[values <= low] = METRIC_BUCKETS[0]
    result.loc[values >= high] = METRIC_BUCKETS[2]
    between = (values > low) & (values < high)
    result.loc[between] = METRIC_BUCKETS[1]
    return result


def group_return(df: pd.DataFrame, size_label: str, metric_label: str, metric_col: str) -> float:
    if metric_col not in df.columns:
        return np.nan
    mask = (df["size_group"] == size_label) & (df[metric_col] == metric_label)
    if not mask.any():
        return np.nan
    return mean_if_any(df.loc[mask, "monthly_return"])


def compute_smb_component(df: pd.DataFrame, metric_col: str) -> float:
    small_vals = []
    big_vals = []
    for bucket in METRIC_BUCKETS:
        small_vals.append(group_return(df, SIZE_LABELS[0], bucket, metric_col))
        big_vals.append(group_return(df, SIZE_LABELS[1], bucket, metric_col))
    small_vals = [val for val in small_vals if pd.notna(val)]
    big_vals = [val for val in big_vals if pd.notna(val)]
    if not small_vals or not big_vals:
        return np.nan
    return float(np.mean(small_vals) - np.mean(big_vals))


def compute_hml(df: pd.DataFrame) -> float:
    sh = group_return(df, SIZE_LABELS[0], METRIC_BUCKETS[2], "bm_bucket")
    bh = group_return(df, SIZE_LABELS[1], METRIC_BUCKETS[2], "bm_bucket")
    sl = group_return(df, SIZE_LABELS[0], METRIC_BUCKETS[0], "bm_bucket")
    bl = group_return(df, SIZE_LABELS[1], METRIC_BUCKETS[0], "bm_bucket")
    needed = [sh, bh, sl, bl]
    if any(pd.isna(val) for val in needed):
        return np.nan
    return float(((sh + bh) / 2.0) - ((sl + bl) / 2.0))


def compute_rmw(df: pd.DataFrame) -> float:
    sh = group_return(df, SIZE_LABELS[0], METRIC_BUCKETS[2], "op_bucket")
    bh = group_return(df, SIZE_LABELS[1], METRIC_BUCKETS[2], "op_bucket")
    sl = group_return(df, SIZE_LABELS[0], METRIC_BUCKETS[0], "op_bucket")
    bl = group_return(df, SIZE_LABELS[1], METRIC_BUCKETS[0], "op_bucket")
    needed = [sh, bh, sl, bl]
    if any(pd.isna(val) for val in needed):
        return np.nan
    return float(((sh + bh) / 2.0) - ((sl + bl) / 2.0))


def compute_cma(df: pd.DataFrame) -> float:
    sl = group_return(df, SIZE_LABELS[0], METRIC_BUCKETS[0], "inv_bucket")
    bl = group_return(df, SIZE_LABELS[1], METRIC_BUCKETS[0], "inv_bucket")
    sh = group_return(df, SIZE_LABELS[0], METRIC_BUCKETS[2], "inv_bucket")
    bh = group_return(df, SIZE_LABELS[1], METRIC_BUCKETS[2], "inv_bucket")
    needed = [sl, bl, sh, bh]
    if any(pd.isna(val) for val in needed):
        return np.nan
    return float(((sl + bl) / 2.0) - ((sh + bh) / 2.0))


def compute_factor_row(month: str, month_df: pd.DataFrame, rf_rate: float) -> Dict[str, float]:
    row = {"month": month}
    for col in FACTOR_COLUMNS:
        row[col] = np.nan
    if month_df.empty:
        return row
    month_df = month_df.copy()
    month_df["size_group"] = assign_size_group(month_df["market_cap"])
    month_df["bm_bucket"] = assign_metric_bucket(month_df["bm_ratio"])
    month_df["op_bucket"] = assign_metric_bucket(month_df["profitability_metric"])
    month_df["inv_bucket"] = assign_metric_bucket(month_df["investment_metric"])

    market_avg = mean_if_any(month_df["monthly_return"])
    if pd.notna(market_avg) and pd.notna(rf_rate):
        row["MKT_RF"] = float(market_avg - rf_rate)
    else:
        row["MKT_RF"] = np.nan

    smb_parts = []
    for bucket_col in ("bm_bucket", "op_bucket", "inv_bucket"):
        value = compute_smb_component(month_df, bucket_col)
        if pd.notna(value):
            smb_parts.append(value)
    if smb_parts:
        row["SMB"] = float(np.mean(smb_parts))

    row["HML"] = compute_hml(month_df)

    if month_df["profitability_metric"].notna().sum() == 0:
        LOGGER.info("RMW: %s 无盈利数据，置为 NaN", month)
        row["RMW"] = np.nan
    else:
        row["RMW"] = compute_rmw(month_df)

    row["CMA"] = compute_cma(month_df)
    return row


def build_factor_series(input_dir: Path, month_list: Sequence[str]) -> pd.DataFrame:
    months = list(month_list)
    returns = load_monthly_frame(input_dir / "monthly_returns.parquet", ["Stkcd", "month", "monthly_return"], months)
    size = load_monthly_frame(input_dir / "monthly_size.parquet", ["Stkcd", "month", "market_cap"], months)
    financials = load_monthly_frame(
        input_dir / "financials.parquet",
        ["Stkcd", "month", "book_value", "profitability_metric", "investment_metric"],
        months,
    )
    rf_lookup = prepare_risk_free(input_dir / "risk_free_monthly.parquet", months)

    panel = (
        returns.merge(size, on=["Stkcd", "month"], how="inner")
        .merge(financials, on=["Stkcd", "month"], how="left")
    )
    if panel.empty:
        rows = [compute_factor_row(month, pd.DataFrame(), rf_lookup.get(month, np.nan)) for month in months]
        return pd.DataFrame(rows)

    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel.loc[panel["market_cap"] <= 0, "market_cap"] = np.nan
    panel = panel.dropna(subset=["monthly_return", "market_cap"])
    panel["bm_ratio"] = panel["book_value"] / panel["market_cap"]
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.reset_index(drop=True)

    rows = []
    for month in months:
        month_df = panel[panel["month"] == month]
        rows.append(compute_factor_row(month, month_df, rf_lookup.get(month, np.nan)))
    result = pd.DataFrame(rows)
    return result[["month"] + FACTOR_COLUMNS]


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("因子序列为空，无法输出摘要")
        return
    stats_rows = []
    for col in FACTOR_COLUMNS:
        series = df[col].dropna()
        stats_rows.append(
            {
                "factor": col,
                "mean": series.mean() if not series.empty else np.nan,
                "std": series.std(ddof=1) if len(series) > 1 else np.nan,
                "months": int(series.count()),
            }
        )
    stats = pd.DataFrame(stats_rows)
    print("因子统计摘要 (data/processed/factor_returns_ff5.parquet):")
    print(stats.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    month_list = build_month_list(args.start_month, args.end_month)
    factors = build_factor_series(args.input_dir, month_list)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    factors.to_parquet(args.output_file, index=False)
    LOGGER.info("五因子已写入 %s", args.output_file.relative_to(ROOT))
    if args.summary:
        print_summary(factors)


if __name__ == "__main__":
    main()
