"""Model comparison utilities for FF5 vs FF6 regression outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

SIGNIFICANCE_P_THRESHOLD = 0.05
T_ABS_THRESHOLD = 1.96

FF5_FACTORS: Sequence[str] = ("MKT_RF", "SMB", "HML", "RMW", "CMA")
FF6_FACTORS: Sequence[str] = (*FF5_FACTORS, "WML")

MODEL_SUMMARY_COLUMNS = (
    "Model",
    "Portfolio_Type",
    "Avg_Alpha",
    "Alpha_Sig_Rate",
    "Mean_R2",
    "Mean_AdjR2",
    "Valid_Portfolios",
)
SUMMARY_EXPORT_COLUMNS = (
    "Model",
    "Avg_Alpha",
    "Alpha_Sig_Rate",
    "Mean_R2",
    "Mean_AdjR2",
    "Valid_Portfolios",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare FF5/FF6 regression outputs without full GRS setup."
    )
    parser.add_argument(
        "--ff5-results",
        type=Path,
        default=Path("docs/tables/regression_results_ff5.csv"),
        help="Path to FF5 regression results CSV.",
    )
    parser.add_argument(
        "--ff6-results",
        type=Path,
        default=Path("docs/tables/regression_results_ff6.csv"),
        help="Path to FF6 regression results CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/tables"),
        help="Directory to write comparison tables.",
    )
    return parser.parse_args()


def load_regression_results(path: Path, model_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing regression results file: {path}")
    df = pd.read_csv(path).copy()
    required_cols = {
        "portfolio_type",
        "size_quintile",
        "factor_quintile",
        "alpha",
        "t_alpha",
        "p_alpha",
        "r_squared",
        "adj_r_squared",
        "n_obs",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df["model"] = model_name
    df["size_quintile"] = pd.to_numeric(df["size_quintile"], errors="coerce").astype(int)
    df["factor_quintile"] = pd.to_numeric(df["factor_quintile"], errors="coerce").astype(int)
    df["n_obs"] = pd.to_numeric(df["n_obs"], errors="coerce").astype(int)
    return df


def alpha_significance_rate(df: pd.DataFrame) -> float:
    series = df["p_alpha"].dropna()
    if series.empty:
        t_vals = df["t_alpha"].abs().dropna()
        if t_vals.empty:
            return float("nan")
        return float((t_vals >= T_ABS_THRESHOLD).mean())
    return float((series < SIGNIFICANCE_P_THRESHOLD).mean())


def compute_model_stats(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "Avg_Alpha": float(df["alpha"].mean(skipna=True)),
        "Alpha_Sig_Rate": alpha_significance_rate(df),
        "Mean_R2": float(df["r_squared"].mean(skipna=True)),
        "Mean_AdjR2": float(df["adj_r_squared"].mean(skipna=True)),
        "Valid_Portfolios": int(df["alpha"].notna().sum()),
    }


def summarize_models(df: pd.DataFrame, group_fields: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    for keys, group in df.groupby(group_fields, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        record: Dict[str, float | str] = {"Portfolio_Type": "ALL"}
        for field, value in zip(group_fields, key_values):
            if field == "model":
                record["Model"] = value
            elif field == "portfolio_type":
                record["Portfolio_Type"] = value
        stats = compute_model_stats(group)
        record.update(stats)
        rows.append(record)
    df_out = pd.DataFrame(rows)
    return df_out.loc[:, MODEL_SUMMARY_COLUMNS]


def factor_significance_rates(df: pd.DataFrame, factors: Sequence[str]) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    for factor in factors:
        t_col = f"t_{factor}"
        if t_col not in df.columns:
            continue
        t_vals = df[t_col].abs().dropna()
        rates[factor] = float((t_vals >= T_ABS_THRESHOLD).mean()) if not t_vals.empty else float("nan")
    return rates


def build_factor_comparison(ff5: pd.DataFrame, ff6: pd.DataFrame) -> pd.DataFrame:
    ff5_rates = factor_significance_rates(ff5, FF5_FACTORS)
    ff6_rates = factor_significance_rates(ff6, FF6_FACTORS)
    factors = list(FF6_FACTORS)
    rows: List[Dict[str, float | str]] = []
    for factor in factors:
        ff5_rate = ff5_rates.get(factor, 0.0 if factor == "WML" else np.nan)
        ff6_rate = ff6_rates.get(factor, np.nan)
        diff = (
            ff6_rate - ff5_rate
            if np.isfinite(ff6_rate) and np.isfinite(ff5_rate)
            else (ff6_rate - ff5_rate if factor == "WML" else np.nan)
        )
        rows.append(
            {
                "Factor": factor,
                "FF5_Sig_Rate": ff5_rate,
                "FF6_Sig_Rate": ff6_rate,
                "Sig_Rate_Diff": diff,
            }
        )
    return pd.DataFrame(rows)


def compute_wml_marginal_stats(ff5: pd.DataFrame, ff6: pd.DataFrame) -> Dict[str, pd.DataFrame | float]:
    key_cols = ["portfolio_type", "size_quintile", "factor_quintile"]
    ff5_subset = ff5.loc[:, key_cols + ["r_squared", "adj_r_squared"]].rename(
        columns={"r_squared": "r_squared_ff5", "adj_r_squared": "adj_r_squared_ff5"}
    )
    ff6_subset = ff6.loc[:, key_cols + ["r_squared", "adj_r_squared", "t_WML"]].rename(
        columns={"r_squared": "r_squared_ff6", "adj_r_squared": "adj_r_squared_ff6"}
    )
    merged = ff5_subset.merge(ff6_subset, on=key_cols, how="inner")
    if merged.empty:
        raise ValueError("FF5/FF6 merge for WML comparison produced no rows.")
    merged["delta_r2"] = merged["r_squared_ff6"] - merged["r_squared_ff5"]
    merged["delta_adj_r2"] = merged["adj_r_squared_ff6"] - merged["adj_r_squared_ff5"]
    overall_delta = float(merged["delta_r2"].mean(skipna=True))
    overall_delta_adj = float(merged["delta_adj_r2"].mean(skipna=True))
    wml_t = merged["t_WML"].abs().dropna()
    wml_sig_rate = float((wml_t >= T_ABS_THRESHOLD).mean()) if not wml_t.empty else float("nan")
    per_portfolio = (
        merged.groupby("portfolio_type")[["delta_r2", "delta_adj_r2"]]
        .mean()
        .reset_index()
        .rename(columns={"delta_r2": "WML_Delta_R2", "delta_adj_r2": "WML_Delta_AdjR2"})
    )
    return {
        "overall_delta_r2": overall_delta,
        "overall_delta_adj_r2": overall_delta_adj,
        "wml_sig_rate": wml_sig_rate,
        "per_portfolio": per_portfolio,
    }


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def format_percent(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.1%}"


def choose_model(summary: pd.DataFrame) -> str:
    ordered = summary.sort_values(["Mean_R2", "Alpha_Sig_Rate"], ascending=[False, True])
    winner = ordered.iloc[0]
    return str(winner["Model"])


def attach_wml_deltas(comparison: pd.DataFrame, wml_stats: Dict[str, object]) -> pd.DataFrame:
    df = comparison.copy()
    df["WML_Delta_R2"] = np.nan
    df["WML_Delta_AdjR2"] = np.nan
    ff6_all_mask = (df["Model"] == "FF6") & (df["Portfolio_Type"] == "ALL")
    df.loc[ff6_all_mask, ["WML_Delta_R2", "WML_Delta_AdjR2"]] = (
        wml_stats["overall_delta_r2"],
        wml_stats["overall_delta_adj_r2"],
    )
    per_portfolio = wml_stats.get("per_portfolio")
    if isinstance(per_portfolio, pd.DataFrame):
        for _, row in per_portfolio.iterrows():
            mask = (df["Model"] == "FF6") & (df["Portfolio_Type"] == row["portfolio_type"])
            df.loc[mask, ["WML_Delta_R2", "WML_Delta_AdjR2"]] = (
                row["WML_Delta_R2"],
                row["WML_Delta_AdjR2"],
            )
    return df


def print_recommendation(summary: pd.DataFrame, wml_stats: Dict[str, object]) -> None:
    print("\n模型关键指标 (ALL):")
    for _, row in summary.iterrows():
        print(
            f"  {row['Model']}: α={row['Avg_Alpha']:.4f}, α显著率={format_percent(row['Alpha_Sig_Rate'])}, "
            f"R²={row['Mean_R2']:.3f}, 调整R²={row['Mean_AdjR2']:.3f}, 有效组合={int(row['Valid_Portfolios'])}"
        )
    print(
        f"\nWML 平均 R² 增量: {wml_stats['overall_delta_r2']:.4f}, 调整R² 增量: {wml_stats['overall_delta_adj_r2']:.4f}, "
        f"WML 显著率: {format_percent(wml_stats['wml_sig_rate'])}"
    )
    recommended = choose_model(summary)
    reason = "R² 提升显著且 α 显著率受控" if recommended == "FF6" else "α 控制更好"
    print(f"\n建议模型：{recommended}（{reason}）")


def main() -> None:
    args = parse_args()
    ff5_df = load_regression_results(args.ff5_results, "FF5")
    ff6_df = load_regression_results(args.ff6_results, "FF6")
    combined = pd.concat([ff5_df, ff6_df], ignore_index=True)
    overall = summarize_models(combined, ["model"])
    per_portfolio = summarize_models(combined, ["model", "portfolio_type"])
    factor_table = build_factor_comparison(ff5_df, ff6_df)
    wml_stats = compute_wml_marginal_stats(ff5_df, ff6_df)

    output_dir = ensure_output_dir(args.output_dir)
    summary_path = output_dir / "model_comparison_summary.csv"
    comparison_path = output_dir / "model_comparison.csv"
    factor_path = output_dir / "factor_significance.csv"

    save_csv(overall.loc[:, SUMMARY_EXPORT_COLUMNS], summary_path)
    comparison_df = pd.concat([overall, per_portfolio], ignore_index=True)
    comparison_df = attach_wml_deltas(comparison_df, wml_stats)
    save_csv(comparison_df, comparison_path)
    save_csv(factor_table, factor_path)
    print_recommendation(overall, wml_stats)
    print(f"\n输出文件：{summary_path}, {comparison_path}, {factor_path}")


if __name__ == "__main__":
    main()
