"""Temporal sample split robustness test for FF3+MOM_24M."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


# 常量配置
FACTOR_COLS = ["MKT_RF", "SMB", "HML", "MOM_24M"]
PORTFOLIO_CONFIG: List[Tuple[str, str, str]] = [
    ("portfolio_returns_size_bm.parquet", "size_bm", "bm_quintile"),
    ("portfolio_returns_size_op.parquet", "size_op", "op_quintile"),
    ("portfolio_returns_size_inv.parquet", "size_inv", "inv_quintile"),
]

PERIOD_DEFS = [
    {
        "key": "period1",
        "label": "Period 1 (2021-08 to 2023-09)",
        "start": "2021-08",
        "end": "2023-09",
    },
    {
        "key": "period2",
        "label": "Period 2 (2023-10 to 2025-12)",
        "start": "2023-10",
        "end": "2025-12",
    },
]

MIN_OBS = 8
HAC_MAX_LAGS = 4
OUTPUT_DIR = Path("docs/tables")


def load_merged_data(input_dir: str = "data/processed") -> pd.DataFrame:
    """加载因子与75组合并合并成 panel。"""

    factors = pd.read_parquet(f"{input_dir}/factor_returns_multiwindow.parquet")
    factors = factors[["month"] + FACTOR_COLS].copy()
    factors["month"] = pd.to_datetime(factors["month"], format="%Y-%m")

    portfolios = []
    for filename, portfolio_type, factor_col in PORTFOLIO_CONFIG:
        df = pd.read_parquet(f"{input_dir}/{filename}")
        df["portfolio_type"] = portfolio_type
        df["factor_quintile"] = df[factor_col].astype(int)
        df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
        portfolios.append(
            df[["month", "portfolio_type", "size_quintile", "factor_quintile", "excess_return"]]
        )

    portfolios_df = pd.concat(portfolios, ignore_index=True)
    portfolios_df["size_quintile"] = portfolios_df["size_quintile"].astype(int)

    merged = portfolios_df.merge(factors, on="month", how="left")
    return merged


def filter_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """返回特定月份区间的数据副本（含首尾）。"""

    start_ts = pd.to_datetime(start, format="%Y-%m")
    end_ts = pd.to_datetime(end, format="%Y-%m")
    mask = (df["month"] >= start_ts) & (df["month"] <= end_ts)
    return df.loc[mask].copy()


def run_period_regressions(period_df: pd.DataFrame, period_key: str) -> pd.DataFrame:
    """对某一时期的所有组合运行 FF3+MOM_24M 回归。"""

    results = []
    group_cols = ["portfolio_type", "size_quintile", "factor_quintile"]

    for (ptype, size_q, factor_q), group in period_df.groupby(group_cols, sort=True):
        subset = group[["excess_return"] + FACTOR_COLS].dropna()
        n_obs = len(subset)

        record: Dict[str, float] = {
            "period": period_key,
            "portfolio_type": ptype,
            "size_quintile": int(size_q),
            "factor_quintile": int(factor_q),
            "n_obs": n_obs,
            "alpha": np.nan,
            "t_alpha": np.nan,
            "p_alpha": np.nan,
            "r_squared": np.nan,
            "adj_r_squared": np.nan,
        }

        for col in FACTOR_COLS:
            record[f"beta_{col}"] = np.nan
            record[f"t_{col}"] = np.nan
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
                    "t_alpha": float(fit.tvalues.get("const", np.nan)),
                    "p_alpha": float(fit.pvalues.get("const", np.nan)),
                    "r_squared": float(fit.rsquared),
                    "adj_r_squared": float(fit.rsquared_adj),
                }
            )

            for col in FACTOR_COLS:
                record[f"beta_{col}"] = float(fit.params.get(col, np.nan))
                record[f"t_{col}"] = float(fit.tvalues.get(col, np.nan))
                record[f"p_{col}"] = float(fit.pvalues.get(col, np.nan))

        except Exception as exc:
            print(f"回归失败 {period_key}: {ptype} s{size_q} f{factor_q}: {exc}")

        results.append(record)

    return pd.DataFrame(results)


def compute_period_metrics(results_df: pd.DataFrame) -> Dict[str, float]:
    """计算单期统计指标。"""

    metrics = {
        "mean_r2": np.nan,
        "mean_adj_r2": np.nan,
        "alpha_sig_rate": np.nan,
        "mom_sig_rate": np.nan,
        "mean_beta_mom": np.nan,
        "std_beta_mom": np.nan,
        "valid_portfolios": 0,
        "total_portfolios": len(results_df),
    }

    valid = results_df[results_df["alpha"].notna()]
    metrics["valid_portfolios"] = len(valid)

    if metrics["valid_portfolios"] == 0:
        return metrics

    metrics["mean_r2"] = valid["r_squared"].mean()
    metrics["mean_adj_r2"] = valid["adj_r_squared"].mean()

    metrics["alpha_sig_rate"] = (valid["p_alpha"] < 0.05).mean() * 100

    mom_mask = valid["beta_MOM_24M"].notna()
    valid_mom = valid.loc[mom_mask]
    if len(valid_mom) > 0:
        metrics["mom_sig_rate"] = (valid_mom["p_MOM_24M"] < 0.05).mean() * 100
        metrics["mean_beta_mom"] = valid_mom["beta_MOM_24M"].mean()
        metrics["std_beta_mom"] = valid_mom["beta_MOM_24M"].std(ddof=1)

    return metrics


def build_comparison_table(
    metrics_early: Dict[str, float],
    metrics_late: Dict[str, float],
    delta_r2: float,
    delta_mom_sig: float,
    coef_stability: float,
) -> pd.DataFrame:
    """生成并排比较表。"""

    rows = [
        {
            "metric": "Mean_R2",
            "period1": metrics_early["mean_r2"],
            "period2": metrics_late["mean_r2"],
            "delta": delta_r2,
        },
        {
            "metric": "Mean_Adj_R2",
            "period1": metrics_early["mean_adj_r2"],
            "period2": metrics_late["mean_adj_r2"],
            "delta": metrics_late["mean_adj_r2"] - metrics_early["mean_adj_r2"],
        },
        {
            "metric": "Alpha_Sig_Rate_pct",
            "period1": metrics_early["alpha_sig_rate"],
            "period2": metrics_late["alpha_sig_rate"],
            "delta": metrics_late["alpha_sig_rate"] - metrics_early["alpha_sig_rate"],
        },
        {
            "metric": "MOM_Sig_Rate_pct",
            "period1": metrics_early["mom_sig_rate"],
            "period2": metrics_late["mom_sig_rate"],
            "delta": delta_mom_sig,
        },
        {
            "metric": "Mean_MOM_Beta",
            "period1": metrics_early["mean_beta_mom"],
            "period2": metrics_late["mean_beta_mom"],
            "delta": metrics_late["mean_beta_mom"] - metrics_early["mean_beta_mom"],
        },
        {
            "metric": "Std_MOM_Beta",
            "period1": metrics_early["std_beta_mom"],
            "period2": metrics_late["std_beta_mom"],
            "delta": metrics_late["std_beta_mom"] - metrics_early["std_beta_mom"],
        },
        {
            "metric": "Delta_R2",
            "period1": np.nan,
            "period2": np.nan,
            "delta": delta_r2,
        },
        {
            "metric": "Delta_MOM_Sig_pct",
            "period1": np.nan,
            "period2": np.nan,
            "delta": delta_mom_sig,
        },
        {
            "metric": "Coefficient_Stability",
            "period1": np.nan,
            "period2": np.nan,
            "delta": coef_stability,
        },
    ]

    return pd.DataFrame(rows)


def print_period_summary(label: str, metrics: Dict[str, float]) -> None:
    """在控制台打印单期表现。"""

    print(f"\n{label}")
    print("-" * len(label))
    valid_count = metrics["valid_portfolios"]
    total_count = metrics["total_portfolios"]
    mean_r2 = metrics["mean_r2"]
    mean_adj_r2 = metrics["mean_adj_r2"]
    alpha_sig = metrics["alpha_sig_rate"]
    mom_sig = metrics["mom_sig_rate"]
    mean_beta = metrics["mean_beta_mom"]
    std_beta = metrics["std_beta_mom"]

    print(f"有效组合: {valid_count} / {total_count}")
    print(f"平均 R^2: {mean_r2:.4f}" if not np.isnan(mean_r2) else "平均 R^2: N/A")
    print(
        f"平均 Adj R^2: {mean_adj_r2:.4f}"
        if not np.isnan(mean_adj_r2)
        else "平均 Adj R^2: N/A"
    )
    print(f"Alpha 显著占比: {alpha_sig:.1f}%" if not np.isnan(alpha_sig) else "Alpha 显著占比: N/A")
    print(
        f"MOM_24M 显著占比: {mom_sig:.1f}%"
        if not np.isnan(mom_sig)
        else "MOM_24M 显著占比: N/A"
    )
    print(
        f"MOM_24M β 平均值: {mean_beta:.4f}" if not np.isnan(mean_beta) else "MOM_24M β 平均值: N/A"
    )
    print(
        f"MOM_24M β 标准差: {std_beta:.4f}"
        if not np.isnan(std_beta)
        else "MOM_24M β 标准差: N/A"
    )


def main():
    print("加载合并数据...")
    merged = load_merged_data()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    period_results = {}
    period_metrics = {}

    for period in PERIOD_DEFS:
        key = period["key"]
        label = period["label"]
        print(f"处理 {label}...")
        subset = filter_period(merged, period["start"], period["end"])
        results_df = run_period_regressions(subset, key)
        period_results[key] = results_df
        period_metrics[key] = compute_period_metrics(results_df)

        output_file = OUTPUT_DIR / f"robustness_sample_split_{key}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"  回归结果写入 {output_file}")

    early_key, late_key = PERIOD_DEFS[0]["key"], PERIOD_DEFS[1]["key"]
    early_metrics = period_metrics[early_key]
    late_metrics = period_metrics[late_key]

    beta_early = period_results[early_key]["beta_MOM_24M"].dropna()
    beta_late = period_results[late_key]["beta_MOM_24M"].dropna()
    combined_abs = pd.concat([beta_early.abs(), beta_late.abs()], ignore_index=True)
    mean_abs_beta = combined_abs.mean() if len(combined_abs) > 0 else np.nan

    delta_r2 = late_metrics["mean_r2"] - early_metrics["mean_r2"]
    delta_mom_sig = late_metrics["mom_sig_rate"] - early_metrics["mom_sig_rate"]
    numerator = abs(late_metrics["mean_beta_mom"] - early_metrics["mean_beta_mom"])

    if np.isnan(numerator) or np.isnan(mean_abs_beta) or mean_abs_beta == 0:
        coef_stability = np.nan
    else:
        coef_stability = numerator / mean_abs_beta

    comparison_df = build_comparison_table(early_metrics, late_metrics, delta_r2, delta_mom_sig, coef_stability)
    comparison_file = OUTPUT_DIR / "robustness_sample_split_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"比较结果写入 {comparison_file}")

    print_period_summary(PERIOD_DEFS[0]["label"], early_metrics)
    print_period_summary(PERIOD_DEFS[1]["label"], late_metrics)

    verdict = "Stable"
    if np.isnan(delta_r2) or np.isnan(delta_mom_sig):
        verdict = "Inconclusive"
    else:
        if abs(delta_r2) >= 0.05 or abs(delta_mom_sig) >= 15:
            verdict = "Unstable"

    print("\nStability Verdict:", verdict)


if __name__ == "__main__":
    main()
