"""构造 4/8/12/24 月等权动量因子并与 FF5 合并生成 multiwindow 因子。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
DOCS_TABLE_DIR = ROOT / "docs" / "tables"
RETURNS_FILE = PROCESSED_DIR / "monthly_returns.parquet"
RISK_FREE_FILE = PROCESSED_DIR / "risk_free_monthly.parquet"
FF5_FILE = PROCESSED_DIR / "factor_returns_ff5.parquet"
OUTPUT_FILE = PROCESSED_DIR / "factor_returns_multiwindow.parquet"
STATS_FILE = DOCS_TABLE_DIR / "mom_multiwindow_stats.csv"
CORR_FILE = DOCS_TABLE_DIR / "mom_correlation_matrix.csv"
MONTH_FORMAT = "%Y-%m"
WINDOWS = (4, 8, 12, 24)
MOM_COLUMNS = [f"MOM_{window}M" for window in WINDOWS]
FACTOR_COLUMNS = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
BUCKET_LABELS = ("Low", "Mid", "High")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--returns-file",
        type=Path,
        default=RETURNS_FILE,
        help="月度收益输入 parquet（默认 data/processed/monthly_returns.parquet）",
    )
    parser.add_argument(
        "--risk-free-file",
        type=Path,
        default=RISK_FREE_FILE,
        help="无风险利率输入 parquet（默认 data/processed/risk_free_monthly.parquet）",
    )
    parser.add_argument(
        "--ff5-file",
        type=Path,
        default=FF5_FILE,
        help="FF5 因子 parquet（默认 data/processed/factor_returns_ff5.parquet）",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=OUTPUT_FILE,
        help="multiwindow 因子输出 parquet（默认 data/processed/factor_returns_multiwindow.parquet）",
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=STATS_FILE,
        help="MOM 因子统计输出 CSV（默认 docs/tables/mom_multiwindow_stats.csv）",
    )
    parser.add_argument(
        "--corr-file",
        type=Path,
        default=CORR_FILE,
        help="FF5+MOM 相关矩阵输出 CSV（默认 docs/tables/mom_correlation_matrix.csv）",
    )
    parser.add_argument(
        "--start-month",
        type=str,
        default="2010-01",
        help="样本开始月份 (YYYY-MM)",
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
    periods = pd.period_range(start=start_period, end=end_period, freq="M")
    return periods.strftime(MONTH_FORMAT).tolist()


def normalize_stkcd(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    values = values.where(values.notna(), None)
    values = values.str.replace(".SH", "", regex=False).str.replace(".SZ", "", regex=False)
    return values.str.zfill(6)


def normalize_month(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.to_period("M").dt.strftime(MONTH_FORMAT)


def prepare_risk_free(path: Path, months: Sequence[str]) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(path)
    rf = pd.read_parquet(path, columns=["month", "rf_rate"])
    rf["month"] = normalize_month(rf["month"])
    rf = (
        rf.sort_values("month")
        .drop_duplicates("month", keep="last")
        .set_index("month")
        .reindex(months)
        .ffill()
        .bfill()
    )
    return rf["rf_rate"].to_dict()


def load_monthly_returns(path: Path, months: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    keep_cols = ["Stkcd", "month", "monthly_return"]
    df = pd.read_parquet(path, columns=keep_cols)
    missing = [col for col in keep_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少必须列: {missing}")
    df["Stkcd"] = normalize_stkcd(df["Stkcd"])
    df["month"] = normalize_month(df["month"])
    df = df[df["month"].isin(set(months))]
    df["monthly_return"] = pd.to_numeric(df["monthly_return"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Stkcd", "month", "monthly_return"])
    return df.reset_index(drop=True)


def load_ff5(path: Path, months: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    required = ["month"] + FACTOR_COLUMNS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少必须列: {missing}")
    df["month"] = normalize_month(df["month"])
    df = df.sort_values("month").drop_duplicates("month", keep="last")
    df = df[df["month"].isin(set(months))]
    return df.reset_index(drop=True)


def attach_excess_return(df: pd.DataFrame, rf_lookup: Dict[str, float]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    result = df.copy()
    result["rf_rate"] = result["month"].map(rf_lookup)
    result["excess_return"] = result["monthly_return"] - result["rf_rate"]
    return result.dropna(subset=["excess_return"])


def compute_window_momentum(df: pd.DataFrame, window: int, col_name: str) -> pd.DataFrame:
    if window < 3:
        raise ValueError("window 需至少 3 个月")
    if df.empty:
        result = df.copy()
        result[col_name] = np.nan
        return result
    result = df.copy()
    result["_month_order"] = pd.to_datetime(result["month"], format=MONTH_FORMAT, errors="coerce")
    result = result.dropna(subset=["_month_order"])
    result = result.sort_values(["Stkcd", "_month_order"])
    result["_gross_return"] = 1.0 + result["monthly_return"]
    grouped = result.groupby("Stkcd", group_keys=False)
    result["_shifted_gross"] = grouped["_gross_return"].shift(2)
    window_length = window - 1

    def rolling_prod(series: pd.Series) -> pd.Series:
        return series.rolling(window_length, min_periods=window_length).apply(np.prod, raw=True)

    prod = grouped["_shifted_gross"].transform(rolling_prod)
    result[col_name] = prod - 1.0
    return result.drop(columns=["_month_order", "_gross_return", "_shifted_gross"])


def assign_momentum_buckets(df: pd.DataFrame, momentum_col: str) -> pd.DataFrame:
    if df.empty:
        labeled = df.copy()
        labeled["momentum_bucket"] = pd.NA
        return labeled

    def label_month(month_df: pd.DataFrame) -> pd.DataFrame:
        labeled = month_df.copy()
        labeled["momentum_bucket"] = pd.NA
        valid = labeled[momentum_col].dropna()
        if len(valid) < 3:
            return labeled
        ranks = valid.rank(method="first", ascending=True)
        tercile = len(valid) / 3.0
        low_idx = ranks.index[ranks <= tercile]
        high_idx = ranks.index[ranks > 2 * tercile]
        mid_idx = ranks.index[(ranks > tercile) & (ranks <= 2 * tercile)]
        labeled.loc[low_idx, "momentum_bucket"] = BUCKET_LABELS[0]
        labeled.loc[mid_idx, "momentum_bucket"] = BUCKET_LABELS[1]
        labeled.loc[high_idx, "momentum_bucket"] = BUCKET_LABELS[2]
        return labeled

    labeled_df = df.groupby("month", group_keys=False, sort=False).apply(label_month)
    return labeled_df.reset_index(drop=True)


def build_factor_series(df: pd.DataFrame, months: Sequence[str], column_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"month": list(months), column_name: np.nan})
    filtered = df[df["momentum_bucket"].isin({BUCKET_LABELS[0], BUCKET_LABELS[2]})]
    if filtered.empty:
        return pd.DataFrame({"month": list(months), column_name: np.nan})
    pivot = (
        filtered.groupby(["month", "momentum_bucket"])["excess_return"]
        .mean()
        .unstack()
    )
    if pivot.empty:
        return pd.DataFrame({"month": list(months), column_name: np.nan})
    high = pivot[BUCKET_LABELS[2]] if BUCKET_LABELS[2] in pivot.columns else pd.Series(np.nan, index=pivot.index)
    low = pivot[BUCKET_LABELS[0]] if BUCKET_LABELS[0] in pivot.columns else pd.Series(np.nan, index=pivot.index)
    factor = (high - low).reindex(months)
    return factor.rename(column_name).reset_index().rename(columns={"index": "month"})


def build_momentum_factor(panel: pd.DataFrame, window: int, months: Sequence[str]) -> pd.DataFrame:
    momentum_col = f"momentum_{window}m"
    factor_col = f"MOM_{window}M"
    scored = compute_window_momentum(panel, window, momentum_col)
    scored = scored.dropna(subset=[momentum_col, "excess_return"])
    labeled = assign_momentum_buckets(scored, momentum_col)
    return build_factor_series(labeled, months, factor_col)


def build_momentum_frame(panel: pd.DataFrame, months: Sequence[str]) -> pd.DataFrame:
    base = pd.DataFrame({"month": list(months)})
    for window in WINDOWS:
        factor = build_momentum_factor(panel, window, months)
        base = base.merge(factor, on="month", how="left")
    if MOM_COLUMNS:
        valid_mask = base[MOM_COLUMNS].notna().all(axis=1)
        mean_values = base[MOM_COLUMNS].mean(axis=1)
        base["MOM_EQ"] = np.where(valid_mask, mean_values, np.nan)
    else:
        base["MOM_EQ"] = np.nan
    return base


def write_mom_stats(df: pd.DataFrame, columns: Sequence[str], path: Path) -> None:
    total = len(df)
    records = []
    for col in columns:
        series = df[col] if col in df.columns else pd.Series(dtype="float64")
        valid = series.dropna()
        record = {
            "Factor": col,
            "Mean": valid.mean() if not valid.empty else np.nan,
            "Std": valid.std(ddof=1) if len(valid) > 1 else np.nan,
            "Skew": valid.skew() if len(valid) > 2 else np.nan,
            "Kurtosis": valid.kurtosis() if len(valid) > 3 else np.nan,
            "Valid_Months": int(valid.count()),
            "Valid_Rate": (int(valid.count()) / total) if total else np.nan,
        }
        records.append(record)
    stats_df = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(path, index=False)
    LOGGER.info("MOM 统计已写入 %s", path.relative_to(ROOT))


def write_correlation(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    columns = [col for col in [*FACTOR_COLUMNS, *MOM_COLUMNS, "MOM_EQ"] if col in df.columns]
    if not columns:
        raise ValueError("无法计算相关矩阵：缺少因子列")
    corr = df[columns].corr()
    path.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(path, float_format="%.6f")
    LOGGER.info("因子相关矩阵已写入 %s", path.relative_to(ROOT))
    return corr


def print_diagnostics(mom_df: pd.DataFrame, corr: pd.DataFrame) -> None:
    total = len(mom_df)
    print("MOM 因子有效月份覆盖情况：")
    for col in [*MOM_COLUMNS, "MOM_EQ"]:
        series = mom_df[col] if col in mom_df.columns else pd.Series(dtype="float64")
        valid = int(series.count())
        rate = valid / total if total else np.nan
        rate_str = f"{rate:.2%}" if not np.isnan(rate) else "NaN"
        print(f"  {col}: {valid}/{total} ({rate_str})")

    if corr is not None:
        mom_cols = [c for c in [*MOM_COLUMNS, "MOM_EQ"] if c in corr.columns]
        ff_cols = [c for c in FACTOR_COLUMNS if c in corr.columns]
        if len(mom_cols) >= 2:
            mom_corr = corr.loc[mom_cols, mom_cols]
            values = []
            for i in range(len(mom_cols)):
                for j in range(i + 1, len(mom_cols)):
                    val = mom_corr.iloc[i, j]
                    if not np.isnan(val):
                        values.append(val)
            if values:
                print(
                    "MOM 因子间相关性：均值 {:.4f} / 最小 {:.4f} / 最大 {:.4f}".format(
                        float(np.mean(values)), float(np.min(values)), float(np.max(values))
                    )
                )
            else:
                print("MOM 因子间相关性：无有效样本")
        else:
            print("MOM 因子间相关性：列不足")

        if mom_cols and ff_cols:
            print("MOM 与 FF5 相关性摘要：")
            for col in mom_cols:
                cross = corr.loc[col, ff_cols]
                valid = cross.dropna()
                if valid.empty:
                    print(f"  {col}: 无有效相关系数")
                else:
                    print(
                        f"  {col}: 均值 {valid.mean():.4f} / 最小 {valid.min():.4f} / 最大 {valid.max():.4f}"
                    )
        else:
            print("MOM 与 FF5 相关性：因子列不足")
    else:
        print("相关矩阵未计算，无法生成诊断")

    print("注意：24M 形成期需要 78-24=54 个月历史，样本前 24 个月将缺少 MOM_24M。")


def build_output_frame(months: Sequence[str], ff5: pd.DataFrame, mom_frame: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame({"month": list(months)})
    merged = base.merge(ff5, on="month", how="left")
    merged = merged.merge(mom_frame, on="month", how="left")
    ordered_cols = ["month", *FACTOR_COLUMNS, *MOM_COLUMNS, "MOM_EQ"]
    existing_cols = [col for col in ordered_cols if col in merged.columns]
    return merged[existing_cols]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    month_list = build_month_list(args.start_month, args.end_month)
    returns = load_monthly_returns(args.returns_file, month_list)
    if returns.empty:
        raise ValueError("月度收益数据为空，无法构建动量因子")
    rf_lookup = prepare_risk_free(args.risk_free_file, month_list)
    panel = attach_excess_return(returns, rf_lookup)
    if panel.empty:
        raise ValueError("超额收益为空，检查输入数据")
    mom_frame = build_momentum_frame(panel, month_list)
    ff5 = load_ff5(args.ff5_file, month_list)
    output = build_output_frame(month_list, ff5, mom_frame)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output_file, index=False)
    LOGGER.info("multiwindow 因子已写入 %s", args.output_file.relative_to(ROOT))
    write_mom_stats(mom_frame, [*MOM_COLUMNS, "MOM_EQ"], args.stats_file)
    corr = write_correlation(output, args.corr_file)
    print_diagnostics(mom_frame, corr)


if __name__ == "__main__":
    main()
