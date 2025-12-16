"""Create comparison tables for multi-window momentum regressions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


TABLE_DIR = Path("docs/tables")
OUTPUT_MATRIX = TABLE_DIR / "multiwindow_performance_matrix.csv"
OUTPUT_BREAKDOWN = TABLE_DIR / "portfolio_type_breakdown.csv"

SINGLE_FACTORS = ["MOM_4M", "MOM_8M", "MOM_12M", "MOM_24M"]
EQ_FACTOR = "MOM_EQ"
BASELINE_LABEL = "baseline"

INPUT_FILES = {
    "FF5+MOM_4M": "regression_results_ff5_mom_4m.csv",
    "FF5+MOM_8M": "regression_results_ff5_mom_8m.csv",
    "FF5+MOM_12M": "regression_results_ff5_mom_12m.csv",
    "FF5+MOM_24M": "regression_results_ff5_mom_24m.csv",
    "FF5+MOM_EQ": "regression_results_ff5_mom_eq.csv",
    "FF5_baseline": "regression_results_ff5_ff5_baseline.csv",
}

PREDICTOR_COUNTS = {
    "FF5_baseline": 5,  # FF5 only
    "FF5+MOM_4M": 6,
    "FF5+MOM_8M": 6,
    "FF5+MOM_12M": 6,
    "FF5+MOM_24M": 6,
    "FF5+MOM_EQ": 6,
}

KEY_COLS = ["portfolio_type", "size_quintile", "factor_quintile"]
ALPHA = 0.05


def read_model_frame(model_key: str) -> pd.DataFrame:
    file_name = INPUT_FILES[model_key]
    file_path = TABLE_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"missing input file: {file_path}")
    return pd.read_csv(file_path)


def compute_p_values(t_values: pd.Series, n_obs: pd.Series, predictor_count: int) -> pd.Series:
    dof = n_obs - (predictor_count + 1)
    p_vals = pd.Series(np.nan, index=t_values.index, dtype=float)
    valid = (~t_values.isna()) & (~dof.isna()) & (dof > 0)
    if valid.any():
        p_vals.loc[valid] = 2 * stats.t.sf(np.abs(t_values.loc[valid]), dof.loc[valid])
    return p_vals


def prepare_panel(df: pd.DataFrame, suffix: str, mom_factor: str | None, predictor_count: int | None) -> pd.DataFrame:
    required = KEY_COLS + ["n_obs", "alpha", "p_alpha", "r_squared", "adj_r_squared"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"columns {missing} missing for panel {suffix}")

    subset = df[required].copy()
    rename_map = {
        "alpha": f"alpha_{suffix}",
        "p_alpha": f"p_alpha_{suffix}",
        "r_squared": f"r_squared_{suffix}",
        "adj_r_squared": f"adj_r_squared_{suffix}",
        "n_obs": f"n_obs_{suffix}",
    }
    panel = subset.rename(columns=rename_map)

    # Only convert numeric KEY_COLS to int (size_quintile, factor_quintile), not portfolio_type
    for col in ["size_quintile", "factor_quintile"]:
        if col in panel.columns:
            panel[col] = panel[col].astype(int)

    if mom_factor:
        beta_col = f"beta_{mom_factor}"
        t_col = f"t_{mom_factor}"
        if beta_col not in df.columns or t_col not in df.columns:
            raise ValueError(f"missing MOM columns for {mom_factor}")
        panel[f"beta_mom_{suffix}"] = df[beta_col]
        panel[f"t_mom_{suffix}"] = df[t_col]
        if predictor_count is None:
            raise ValueError("predictor_count required when mom_factor set")
        panel[f"mom_p_{suffix}"] = compute_p_values(df[t_col], df["n_obs"], predictor_count)
    else:
        panel[f"beta_mom_{suffix}"] = np.nan
        panel[f"t_mom_{suffix}"] = np.nan
        panel[f"mom_p_{suffix}"] = np.nan

    return panel


def share_true(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return float(valid.mean())


def share_positive(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return float((valid > 0).mean())


def safe_stat(series: pd.Series, fn) -> float:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return float(fn(valid))


def pvalue_to_sig(series: pd.Series) -> pd.Series:
    mask = series < ALPHA
    return mask.where(series.notna())


def attach_comparison_columns(df: pd.DataFrame, factor: str) -> pd.DataFrame:
    work = df.copy()
    work["delta_r2_single"] = work[f"r_squared_{factor}"] - work["r_squared_baseline"]
    work["delta_r2_eq"] = work[f"r_squared_{EQ_FACTOR}"] - work["r_squared_baseline"]
    work["relative_improvement_vs_single"] = work["delta_r2_eq"] - work["delta_r2_single"]
    p_alpha_single = work[f"p_alpha_{factor}"]
    work["alpha_sig_single"] = pvalue_to_sig(p_alpha_single)
    mom_p_single = work[f"mom_p_{factor}"]
    work["mom_sig_single"] = pvalue_to_sig(mom_p_single)
    mom_p_eq = work[f"mom_p_{EQ_FACTOR}"]
    work["mom_sig_eq"] = pvalue_to_sig(mom_p_eq)
    p_alpha_eq = work[f"p_alpha_{EQ_FACTOR}"]
    work["alpha_sig_eq"] = pvalue_to_sig(p_alpha_eq)
    return work


def summarize_factor(df: pd.DataFrame, factor: str) -> Dict[str, float | str]:
    delta_single = df["delta_r2_single"]
    delta_eq = df["delta_r2_eq"]
    mean_delta_single = safe_stat(delta_single, pd.Series.mean)
    mean_delta_eq = safe_stat(delta_eq, pd.Series.mean)
    return {
        "factor": factor,
        "mean_delta_r2_single": mean_delta_single,
        "mean_delta_r2_eq": mean_delta_eq,
        "relative_improvement_eq_vs_single": None if pd.isna(mean_delta_single) or pd.isna(mean_delta_eq) else mean_delta_eq - mean_delta_single,
        "median_delta_r2_single": safe_stat(delta_single, pd.Series.median),
        "median_delta_r2_eq": safe_stat(delta_eq, pd.Series.median),
        "positive_delta_share_single": share_positive(delta_single),
        "positive_delta_share_eq": share_positive(delta_eq),
        "alpha_sig_rate_single_pct": share_true(df["alpha_sig_single"]) * 100,
        "alpha_sig_rate_eq_pct": share_true(df["alpha_sig_eq"]) * 100,
        "mom_sig_rate_single_pct": share_true(df["mom_sig_single"]) * 100,
        "mom_sig_rate_eq_pct": share_true(df["mom_sig_eq"]) * 100,
        "n_portfolios": len(df),
    }


def summarize_breakdown(df: pd.DataFrame, factor: str) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for ptype, group in df.groupby("portfolio_type"):
        row = {
            "portfolio_type": ptype,
            "factor": factor,
            "mean_delta_r2_single": safe_stat(group["delta_r2_single"], pd.Series.mean),
            "median_delta_r2_single": safe_stat(group["delta_r2_single"], pd.Series.median),
            "positive_delta_share_single": share_positive(group["delta_r2_single"]),
            "mom_sig_rate_single_pct": share_true(group["mom_sig_single"]) * 100,
            "alpha_sig_rate_single_pct": share_true(group["alpha_sig_single"]) * 100,
            "n_portfolios": len(group),
        }
        if "delta_r2_eq" in df.columns:
            row["mean_delta_r2_eq"] = safe_stat(group["delta_r2_eq"], pd.Series.mean)
            row["median_delta_r2_eq"] = safe_stat(group["delta_r2_eq"], pd.Series.median)
            row["positive_delta_share_eq"] = share_positive(group["delta_r2_eq"])
            row["mom_sig_rate_eq_pct"] = share_true(group["mom_sig_eq"]) * 100
            row["alpha_sig_rate_eq_pct"] = share_true(group["alpha_sig_eq"]) * 100
        rows.append(row)
    return rows


def print_summary(matrix_df: pd.DataFrame, eq_row: Dict[str, float | str], breakdown_df: pd.DataFrame) -> None:
    singles = matrix_df[matrix_df["factor"].isin(SINGLE_FACTORS)].copy()
    singles = singles.dropna(subset=["mean_delta_r2_single"])
    if singles.empty:
        print("=== Summary ===")
        print("No single-window results available.")
        return

    best_idx = singles["mean_delta_r2_single"].idxmax()
    best_single = singles.loc[best_idx]
    eq_mean_delta = eq_row.get("mean_delta_r2_eq")
    best_name = best_single["factor"]
    best_delta = best_single["mean_delta_r2_single"]
    if pd.isna(eq_mean_delta) or pd.isna(best_delta):
        delta_gap = None
    else:
        delta_gap = float(eq_mean_delta - best_delta)

    eq_breakdown = breakdown_df[breakdown_df["factor"] == EQ_FACTOR].copy()
    eq_breakdown = eq_breakdown.dropna(subset=["mean_delta_r2_single"])
    eq_best_ptype = None
    eq_best_delta = None
    if not eq_breakdown.empty:
        best_ptype_idx = eq_breakdown["mean_delta_r2_single"].idxmax()
        eq_best_ptype = eq_breakdown.loc[best_ptype_idx, "portfolio_type"]
        eq_best_delta = eq_breakdown.loc[best_ptype_idx, "mean_delta_r2_single"]

    print("=== Summary ===")
    print(f"Best single factor (ΔR² avg): {best_name} @ {best_delta:.6f}")
    if pd.isna(eq_mean_delta):
        print("MOM_EQ ΔR² avg: n/a")
    else:
        relation = "matches"
        if delta_gap is not None:
            if delta_gap > 1e-9:
                relation = "outperforms"
            elif delta_gap < -1e-9:
                relation = "lags"
        gap_text = f" by {delta_gap:.6f}" if delta_gap is not None else ""
        print(f"MOM_EQ ΔR² avg: {eq_mean_delta:.6f} ({relation} best single{gap_text})")
    if eq_best_ptype is not None and eq_best_delta is not None:
        print(f"Top portfolio_type for MOM_EQ: {eq_best_ptype} (mean ΔR² {eq_best_delta:.6f})")

    if delta_gap is None or pd.isna(delta_gap):
        recommendation = best_name
    elif delta_gap >= 0:
        recommendation = EQ_FACTOR
    else:
        recommendation = best_name
    print(f"Recommendation: use {recommendation} based on average ΔR².")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    baseline_df = read_model_frame("FF5_baseline")
    baseline_panel = prepare_panel(baseline_df, BASELINE_LABEL, mom_factor=None, predictor_count=None)

    eq_df = read_model_frame("FF5+MOM_EQ")
    eq_panel = prepare_panel(eq_df, EQ_FACTOR, mom_factor=EQ_FACTOR, predictor_count=PREDICTOR_COUNTS["FF5+MOM_EQ"])

    baseline_eq = baseline_panel.merge(eq_panel, on=KEY_COLS, how="inner")
    baseline_eq = attach_comparison_columns(baseline_eq, EQ_FACTOR)

    comparison_frames: Dict[str, pd.DataFrame] = {}
    matrix_rows: List[Dict[str, float | str]] = []

    for factor in SINGLE_FACTORS:
        model_key = f"FF5+{factor}"
        single_df = read_model_frame(model_key)
        single_panel = prepare_panel(single_df, factor, mom_factor=factor, predictor_count=PREDICTOR_COUNTS[model_key])
        merged = baseline_eq.merge(single_panel, on=KEY_COLS, how="inner")
        merged = attach_comparison_columns(merged, factor)
        comparison_frames[factor] = merged
        matrix_rows.append(summarize_factor(merged, factor))

    matrix_rows.append(summarize_factor(baseline_eq, EQ_FACTOR))

    matrix_df = pd.DataFrame(matrix_rows)
    single_mask = matrix_df["factor"].isin(SINGLE_FACTORS)
    avg_single_delta = matrix_df.loc[single_mask, "mean_delta_r2_single"].mean()
    matrix_df["relative_improvement_vs_avg_single"] = matrix_df["mean_delta_r2_eq"] - avg_single_delta

    eq_mask = matrix_df["factor"] == EQ_FACTOR
    single_only_cols = [
        "mean_delta_r2_single",
        "median_delta_r2_single",
        "positive_delta_share_single",
        "alpha_sig_rate_single_pct",
        "mom_sig_rate_single_pct",
        "relative_improvement_eq_vs_single",
    ]
    matrix_df.loc[eq_mask, single_only_cols] = np.nan

    matrix_df.to_csv(OUTPUT_MATRIX, index=False)

    breakdown_rows: List[Dict[str, float | str]] = []
    for factor, frame in comparison_frames.items():
        breakdown_rows.extend(summarize_breakdown(frame, factor))
    breakdown_rows.extend(summarize_breakdown(baseline_eq, EQ_FACTOR))

    breakdown_df = pd.DataFrame(breakdown_rows)
    breakdown_df.to_csv(OUTPUT_BREAKDOWN, index=False)

    eq_row_dict = matrix_df[matrix_df["factor"] == EQ_FACTOR].iloc[0].to_dict()
    print_summary(matrix_df, eq_row_dict, breakdown_df)


if __name__ == "__main__":
    main()
