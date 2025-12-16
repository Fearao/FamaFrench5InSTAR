"""构造 12 个月动量因子 (WML)，并与 FF5 合并得到 FF6。"""

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
FF6_FILE = PROCESSED_DIR / "factor_returns_ff6.parquet"
WML_STATS_FILE = DOCS_TABLE_DIR / "wml_stats.csv"
CORR_FILE = DOCS_TABLE_DIR / "ff6_corr.csv"
MONTH_FORMAT = "%Y-%m"
MOMENTUM_WINDOW = 12
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
        default=FF6_FILE,
        help="FF6 输出 parquet（默认 data/processed/factor_returns_ff6.parquet）",
    )
    parser.add_argument(
        "--wml-stats-file",
        type=Path,
        default=WML_STATS_FILE,
        help="WML 统计输出 CSV（默认 docs/tables/wml_stats.csv）",
    )
    parser.add_argument(
        "--corr-file",
        type=Path,
        default=CORR_FILE,
        help="FF6 因子相关矩阵输出 CSV（默认 docs/tables/ff6_corr.csv）",
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


def compute_rolling_momentum(df: pd.DataFrame, window: int) -> pd.DataFrame:
    result = df.copy()
    if result.empty:
        result["momentum_12m"] = np.nan
        return result
    result["_month_order"] = pd.to_datetime(result["month"], format=MONTH_FORMAT, errors="coerce")
    result = result.dropna(subset=["_month_order"])
    result = result.sort_values(["Stkcd", "_month_order"])
    result["gross_return"] = 1.0 + result["monthly_return"]
    rolling_prod = (
        result.groupby("Stkcd", group_keys=False)["gross_return"]
        .transform(lambda s: s.rolling(window, min_periods=window).apply(np.prod, raw=True))
    )
    result["momentum_12m"] = rolling_prod - 1.0
    result = result.drop(columns=["gross_return", "_month_order"])
    return result.sort_values(["month", "Stkcd"]).reset_index(drop=True)


def assign_momentum_buckets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df = df.copy()
        df["momentum_bucket"] = pd.NA
        return df

    def label_month(month_df: pd.DataFrame) -> pd.DataFrame:
        labeled = month_df.copy()
        labeled["momentum_bucket"] = pd.NA
        valid = labeled["momentum_12m"].dropna()
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


def build_wml_series(df: pd.DataFrame, months: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"month": list(months), "WML": np.nan})
    filtered = df[df["momentum_bucket"].isin({BUCKET_LABELS[0], BUCKET_LABELS[2]})]
    if filtered.empty:
        return pd.DataFrame({"month": list(months), "WML": np.nan})
    pivot = (
        filtered.groupby(["month", "momentum_bucket"])["excess_return"]
        .mean()
        .unstack()
    )
    if pivot.empty:
        return pd.DataFrame({"month": list(months), "WML": np.nan})
    high = pivot[BUCKET_LABELS[2]] if BUCKET_LABELS[2] in pivot.columns else pd.Series(np.nan, index=pivot.index)
    low = pivot[BUCKET_LABELS[0]] if BUCKET_LABELS[0] in pivot.columns else pd.Series(np.nan, index=pivot.index)
    wml = high - low
    wml = wml.reindex(months)
    return wml.rename("WML").reset_index().rename(columns={"index": "month"})


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


def build_ff6_frame(months: Sequence[str], ff5: pd.DataFrame, wml: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame({"month": list(months)})
    merged = base.merge(ff5, on="month", how="left").merge(wml, on="month", how="left")
    columns = ["month"] + FACTOR_COLUMNS + ["WML"]
    return merged[columns]


def write_wml_stats(series: pd.Series, path: Path) -> None:
    stats = {
        "mean": series.mean() if not series.empty else np.nan,
        "std": series.std(ddof=1) if len(series) > 1 else np.nan,
        "skew": series.skew() if len(series) > 2 else np.nan,
        "kurtosis": series.kurtosis() if len(series) > 3 else np.nan,
    }
    df = pd.DataFrame([stats])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOGGER.info("WML 统计已写入 %s", path.relative_to(ROOT))


def write_correlation(ff6: pd.DataFrame, path: Path) -> pd.DataFrame:
    cols = FACTOR_COLUMNS + ["WML"]
    corr = ff6[cols].corr()
    path.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(path, float_format="%.6f")
    LOGGER.info("FF6 因子相关矩阵已写入 %s", path.relative_to(ROOT))
    return corr


def print_summary(wml_series: pd.Series, corr: pd.DataFrame) -> None:
    coverage = int(wml_series.count())
    total = len(wml_series)
    print(f"WML 覆盖月数: {coverage} / {total}")
    if corr is not None and "WML" in corr.index:
        other = corr.loc["WML"].drop(labels=["WML"], errors="ignore")
        if other.empty or other.isna().all():
            print("WML 与其他因子相关性: 无有效数据")
        else:
            print("WML 与其他因子相关系数:")
            print(other.to_string(float_format=lambda x: f"{x:.4f}"))
    else:
        print("WML 与其他因子相关性: 未能计算")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    month_list = build_month_list(args.start_month, args.end_month)
    returns = load_monthly_returns(args.returns_file, month_list)
    rf_lookup = prepare_risk_free(args.risk_free_file, month_list)
    panel = compute_rolling_momentum(returns, MOMENTUM_WINDOW)
    panel["rf_rate"] = panel["month"].map(rf_lookup)
    panel["excess_return"] = panel["monthly_return"] - panel["rf_rate"]
    panel = panel.dropna(subset=["momentum_12m"])
    labeled = assign_momentum_buckets(panel)
    wml = build_wml_series(labeled, month_list)
    ff5 = load_ff5(args.ff5_file, month_list)
    ff6 = build_ff6_frame(month_list, ff5, wml)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    ff6.to_parquet(args.output_file, index=False)
    LOGGER.info("FF6 因子已写入 %s", args.output_file.relative_to(ROOT))
    wml_series = wml["WML"].dropna()
    write_wml_stats(wml_series, args.wml_stats_file)
    corr = write_correlation(ff6, args.corr_file)
    print_summary(wml["WML"], corr)


if __name__ == "__main__":
    main()
