"""Rolling-window robustness analysis for FF3+MOM_EQ."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


MOM_FACTOR = "MOM_EQ"
FACTOR_COLS = ["MKT_RF", "SMB", "HML", MOM_FACTOR]
PORTFOLIO_CONFIG: List[Tuple[str, str, str]] = [
    ("portfolio_returns_size_bm.parquet", "size_bm", "bm_quintile"),
    ("portfolio_returns_size_op.parquet", "size_op", "op_quintile"),
    ("portfolio_returns_size_inv.parquet", "size_inv", "inv_quintile"),
]

WINDOW_MONTHS = 24
STEP_MONTHS = 6
MIN_OBS = 24
HAC_MAX_LAGS = 4
START_MONTH = "2021-08"
END_MONTH = "2025-12"

OUTPUT_DIR = Path("docs/tables")
OUTPUT_FILE = OUTPUT_DIR / "robustness_rolling_window_EQ_stats.csv"
MOM24M_BASELINE_CORR = -0.37


def load_merged_data(input_dir: str = "data/processed") -> pd.DataFrame:
    """Load merged factor + portfolio panel used for regressions."""

    factors = pd.read_parquet(f"{input_dir}/factor_returns_multiwindow.parquet")
    factors = factors[["month"] + FACTOR_COLS].copy()
    factors["month"] = pd.to_datetime(factors["month"], format="%Y-%m")

    portfolios = []
    for filename, portfolio_type, factor_col in PORTFOLIO_CONFIG:
        df = pd.read_parquet(f"{input_dir}/{filename}")
        df["portfolio_type"] = portfolio_type
        df["factor_quintile"] = df[factor_col].astype(int)
        df["size_quintile"] = df["size_quintile"].astype(int)
        df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
        portfolios.append(
            df[["month", "portfolio_type", "size_quintile", "factor_quintile", "excess_return"]]
        )

    portfolios_df = pd.concat(portfolios, ignore_index=True)
    merged = portfolios_df.merge(factors, on="month", how="left")
    return merged


def generate_windows(
    start_month: str,
    end_month: str,
    window_months: int = WINDOW_MONTHS,
    step_months: int = STEP_MONTHS,
) -> List[Dict[str, pd.Timestamp]]:
    """Return list of rolling windows fully contained in [start, end]."""

    start_ts = pd.Period(start_month, freq="M").to_timestamp()
    end_ts = pd.Period(end_month, freq="M").to_timestamp()

    windows: List[Dict[str, pd.Timestamp]] = []
    current_start = start_ts

    while True:
        current_end = current_start + pd.DateOffset(months=window_months - 1)
        if current_end > end_ts:
            break
        windows.append({"start": current_start, "end": current_end})
        current_start = current_start + pd.DateOffset(months=step_months)

    return windows


def run_regressions(window_df: pd.DataFrame) -> pd.DataFrame:
    """Run FF3+MOM regressions for every portfolio within a window."""

    results = []
    group_cols = ["portfolio_type", "size_quintile", "factor_quintile"]

    for (ptype, size_q, factor_q), group in window_df.groupby(group_cols, sort=True):
        subset = group[["excess_return"] + FACTOR_COLS].dropna()
        n_obs = len(subset)

        record: Dict[str, float] = {
            "portfolio_type": ptype,
            "size_quintile": int(size_q),
            "factor_quintile": int(factor_q),
            "n_obs": n_obs,
            "alpha": np.nan,
            "p_alpha": np.nan,
            "r_squared": np.nan,
            "adj_r_squared": np.nan,
        }

        for col in FACTOR_COLS:
            record[f"beta_{col}"] = np.nan
            record[f"p_{col}"] = np.nan

        if n_obs < MIN_OBS:
            results.append(record)
            continue

        try:
            y = subset["excess_return"].astype(float)
            X = sm.add_constant(subset[FACTOR_COLS].astype(float))
            hac_lags = max(0, min(HAC_MAX_LAGS, n_obs - 1))
            fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

            record.update(
                {
                    "alpha": float(fit.params.get("const", np.nan)),
                    "p_alpha": float(fit.pvalues.get("const", np.nan)),
                    "r_squared": float(fit.rsquared),
                    "adj_r_squared": float(fit.rsquared_adj),
                }
            )

            for col in FACTOR_COLS:
                record[f"beta_{col}"] = float(fit.params.get(col, np.nan))
                record[f"p_{col}"] = float(fit.pvalues.get(col, np.nan))

        except Exception as exc:  # pragma: no cover - logging only
            print(f"回归失败: {ptype} s{size_q} f{factor_q}: {exc}")

        results.append(record)

    return pd.DataFrame(results)


def compute_window_stats(
    results_df: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    n_months: int,
) -> Dict[str, float]:
    """Aggregate metrics needed for CSV output for a single window."""

    stats: Dict[str, float] = {
        "window_start": window_start.strftime("%Y-%m"),
        "window_end": window_end.strftime("%Y-%m"),
        "n_months": n_months,
        "mean_r2": np.nan,
        "mean_adj_r2": np.nan,
        "mom_sig_rate": np.nan,
        "alpha_sig_rate": np.nan,
        "mean_abs_mom_beta": np.nan,
    }

    if results_df.empty:
        return stats

    valid = results_df[results_df["alpha"].notna()].copy()
    if valid.empty:
        return stats

    stats["mean_r2"] = valid["r_squared"].mean()
    stats["mean_adj_r2"] = valid["adj_r_squared"].mean()
    stats["alpha_sig_rate"] = (valid["p_alpha"] < 0.05).mean() * 100

    beta_col = f"beta_{MOM_FACTOR}"
    p_col = f"p_{MOM_FACTOR}"

    mom_valid = valid[beta_col].notna()
    if mom_valid.any():
        subset = valid.loc[mom_valid, beta_col]
        stats["mom_sig_rate"] = (valid.loc[mom_valid, p_col] < 0.05).mean() * 100
        stats["mean_abs_mom_beta"] = subset.abs().mean()

    return stats


def analyze_trend(summary_df: pd.DataFrame) -> float:
    """Compute correlation between window end date and MOM significance."""

    if summary_df.empty or summary_df["mom_sig_rate"].dropna().shape[0] < 2:
        return np.nan

    end_dates = pd.to_datetime(summary_df["window_end"], format="%Y-%m")
    valid = summary_df[["mom_sig_rate"]].copy()
    valid["end_ordinal"] = end_dates.map(pd.Timestamp.toordinal)
    valid = valid.dropna()

    if valid.shape[0] < 2:
        return np.nan

    corr = np.corrcoef(valid["end_ordinal"], valid["mom_sig_rate"])[0, 1]
    return float(corr)


def verdict_from_corr(corr: float) -> str:
    if np.isnan(corr):
        return "Stable"
    if corr > 0.3:
        return "Improving"
    if corr < -0.3:
        return "Declining"
    return "Stable"


def main():
    print("加载并合并数据...(MOM_EQ)")
    merged = load_merged_data()

    windows = generate_windows(START_MONTH, END_MONTH)
    print(f"MOM_EQ 滚动窗口总数(24M, 步长6M): {len(windows)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_records = []

    for idx, window in enumerate(windows, start=1):
        start_ts = window["start"]
        end_ts = window["end"]
        mask = (merged["month"] >= start_ts) & (merged["month"] <= end_ts)
        window_df = merged.loc[mask].copy()
        n_months = window_df["month"].nunique()

        print(f"窗口 {idx}: {start_ts:%Y-%m} 至 {end_ts:%Y-%m}, 月数 {n_months}")

        if n_months < WINDOW_MONTHS:
            print("  警告：有效月份不足 24，结果可能不稳定。")

        results_df = run_regressions(window_df)
        stats = compute_window_stats(results_df, start_ts, end_ts, int(n_months))
        summary_records.append(stats)

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(OUTPUT_FILE, index=False)
    print(f"MOM_EQ 统计已写入 {OUTPUT_FILE}")

    corr = analyze_trend(summary_df)
    verdict = verdict_from_corr(corr)

    mom_sig = summary_df["mom_sig_rate"].dropna()
    mom_range = float(mom_sig.max() - mom_sig.min()) if not mom_sig.empty else np.nan
    mom_std = float(mom_sig.std(ddof=0)) if not mom_sig.empty else np.nan
    mom_mean = float(mom_sig.mean()) if not mom_sig.empty else np.nan
    mom_cv = float(mom_std / mom_mean) if mom_mean and not np.isnan(mom_std) else np.nan

    print("\nMOM_EQ 滚动汇总")
    print("----------------")
    print(f"窗口总数: {len(summary_df)} (预期5)")
    if np.isnan(corr):
        print("MOM_EQ 显著率相关系数: NaN (有效窗口不足)")
    else:
        print(
            f"MOM_EQ 显著率相关系数: {corr:.4f} (MOM_24M 基准 {MOM24M_BASELINE_CORR:+.2f})"
        )

    if mom_sig.empty:
        print("MOM_EQ 显著率波动指标: 无可用样本")
    else:
        print(f"MOM_EQ 显著率极差: {mom_range:.2f}pp")
        print(f"MOM_EQ 显著率标准差: {mom_std:.2f}pp")
        if np.isnan(mom_cv):
            print("MOM_EQ 显著率CV: NaN")
        else:
            print(f"MOM_EQ 显著率CV: {mom_cv:.3f}")

    is_declining = verdict == "Declining"
    trend_label = "下降" if is_declining else "非下降"
    print(f"是否呈现下降趋势: {trend_label} ({verdict})")


if __name__ == "__main__":
    main()
