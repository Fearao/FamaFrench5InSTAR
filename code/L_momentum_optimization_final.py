"""Momentum model optimization scorecard generator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


MOMENTUM_MODELS: Sequence[tuple[str, bool]] = (
    ("MOM_4M", False),
    ("MOM_8M", False),
    ("MOM_12M", False),
    ("MOM_24M", False),
    ("MOM_EQ", True),
)

WEIGHTS = {
    "delta_r2": 0.40,
    "mom_sig_rate": 0.30,
    "simplicity": 0.20,
    "economic": 0.10,
}

SIMPLICITY_SCORES = {False: 100.0, True: 70.0}
ECONOMIC_SCORES = {False: 100.0, True: 80.0}

PERFORMANCE_REQUIRED_COLUMNS = {
    "factor",
    "mean_delta_r2_single",
    "mean_delta_r2_eq",
    "mom_sig_rate_single_pct",
    "mom_sig_rate_eq_pct",
}

SUMMARY_REQUIRED_COLUMNS = {
    "Model",
    "Avg_Alpha",
    "Alpha_Sig_Rate",
    "Mean_R2",
    "Mean_AdjR2",
    "Valid_Portfolios",
    "MOM_Sig_Rate",
}


@dataclass
class ScorecardPaths:
    performance_matrix: Path
    regression_summary: Path
    scorecard_output: Path
    report_output: Path


def parse_args() -> ScorecardPaths:
    parser = argparse.ArgumentParser(
        description="Score five momentum candidates and create an optimization report."
    )
    parser.add_argument(
        "--performance-matrix",
        type=Path,
        default=Path("docs/tables/multiwindow_performance_matrix.csv"),
        help="Path to the multiwindow performance matrix CSV.",
    )
    parser.add_argument(
        "--regression-summary",
        type=Path,
        default=Path("docs/tables/multiwindow_regression_summary.csv"),
        help="Path to the regression summary CSV.",
    )
    parser.add_argument(
        "--scorecard-output",
        type=Path,
        default=Path("docs/tables/optimization_scorecard.csv"),
        help="Where to write the optimization scorecard CSV.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("docs/momentum_optimization_report.md"),
        help="Where to write the markdown report.",
    )
    args = parser.parse_args()
    return ScorecardPaths(
        performance_matrix=args.performance_matrix,
        regression_summary=args.regression_summary,
        scorecard_output=args.scorecard_output,
        report_output=args.report_output,
    )


def load_table(path: Path, required_columns: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required table: {path}")
    df = pd.read_csv(path)
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def normalize_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    clean = numeric.replace([np.inf, -np.inf], np.nan)
    valid = clean.dropna()
    if valid.empty:
        return pd.Series(100.0, index=series.index)
    min_val = float(valid.min())
    max_val = float(valid.max())
    if np.isclose(max_val, min_val):
        return pd.Series(100.0, index=series.index)
    return ((clean - min_val) / (max_val - min_val)) * 100.0


def _extract_summary_lookup(summary_df: pd.DataFrame) -> pd.DataFrame:
    frame = summary_df.copy()
    frame["Model_Key"] = frame["Model"].astype(str).str.replace("FF5+", "", regex=False).str.strip()
    return frame.set_index("Model_Key")


def _safe_float(value: object) -> float:
    try:
        return float(pd.to_numeric([value], errors="coerce")[0])
    except (TypeError, ValueError):
        return float("nan")


def build_model_metrics(perf_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    summary_lookup = _extract_summary_lookup(summary_df)
    rows: List[dict[str, float | str]] = []
    for factor, is_eq in MOMENTUM_MODELS:
        match = perf_df.loc[perf_df["factor"] == factor]
        if match.empty:
            raise ValueError(f"No performance entry for {factor}")
        perf_row = match.iloc[0]
        delta_col = "mean_delta_r2_eq" if is_eq else "mean_delta_r2_single"
        sig_col = "mom_sig_rate_eq_pct" if is_eq else "mom_sig_rate_single_pct"
        delta_val = _safe_float(perf_row.get(delta_col))
        sig_val = _safe_float(perf_row.get(sig_col))
        summary_row = summary_lookup.loc[factor] if factor in summary_lookup.index else None
        mean_r2 = _safe_float(summary_row["Mean_R2"]) if summary_row is not None else float("nan")
        mean_adj_r2 = _safe_float(summary_row["Mean_AdjR2"]) if summary_row is not None else float("nan")
        mom_sig_summary = (
            _safe_float(summary_row["MOM_Sig_Rate"]) * 100 if summary_row is not None else float("nan")
        )
        rows.append(
            {
                "Model": factor,
                "Mean_Delta_R2": delta_val,
                "MOM_Significance_Rate_pct": sig_val,
                "Simplicity_Score": SIMPLICITY_SCORES[is_eq],
                "Economic_Score": ECONOMIC_SCORES[is_eq],
                "Mean_R2": mean_r2,
                "Mean_AdjR2": mean_adj_r2,
                "Regression_MOM_Sig_Rate_pct": mom_sig_summary,
            }
        )
    df = pd.DataFrame(rows)
    df["Mean_Delta_R2_pct"] = df["Mean_Delta_R2"] * 100
    df["Delta_R2_Score"] = normalize_series(df["Mean_Delta_R2"])
    df["Optimization_Score"] = (
        df["Delta_R2_Score"] * WEIGHTS["delta_r2"]
        + df["MOM_Significance_Rate_pct"] * WEIGHTS["mom_sig_rate"]
        + df["Simplicity_Score"] * WEIGHTS["simplicity"]
        + df["Economic_Score"] * WEIGHTS["economic"]
    )
    df = df.sort_values("Optimization_Score", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    df["Is_Winner"] = df["Rank"] == 1
    return df


def write_scorecard(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_columns = [
        "Rank",
        "Model",
        "Mean_Delta_R2",
        "Mean_Delta_R2_pct",
        "Delta_R2_Score",
        "MOM_Significance_Rate_pct",
        "Simplicity_Score",
        "Economic_Score",
        "Optimization_Score",
        "Mean_R2",
        "Mean_AdjR2",
        "Regression_MOM_Sig_Rate_pct",
        "Is_Winner",
    ]
    df.loc[:, ordered_columns].to_csv(path, index=False)


def _format_percent(value: float, digits: int = 2) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}%"


def generate_report(df: pd.DataFrame, paths: ScorecardPaths) -> str:
    path = paths.report_output
    path.parent.mkdir(parents=True, exist_ok=True)
    winner = df.iloc[0]
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    table_lines = ["| Rank | Model | ΔR² (pp) | MOM Sig. (%) | Simplicity | Economic | Score |"]
    table_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for _, row in df.iterrows():
        table_lines.append(
            "| {rank} | {model} | {delta:.2f} | {sig:.1f} | {simp:.0f} | {econ:.0f} | {score:.1f} |".format(
                rank=int(row["Rank"]),
                model=row["Model"],
                delta=row["Mean_Delta_R2_pct"],
                sig=row["MOM_Significance_Rate_pct"],
                simp=row["Simplicity_Score"],
                econ=row["Economic_Score"],
                score=row["Optimization_Score"],
            )
        )
    recommendation = (
        f"**Recommendation:** {winner['Model']} wins with a score of {winner['Optimization_Score']:.1f}, "
        f"delivering {winner['Mean_Delta_R2_pct']:.2f} pp average ΔR² improvement and "
        f"{winner['MOM_Significance_Rate_pct']:.1f}% MOM significance."
    )
    methodology = (
        "Weights → ΔR²: {delta:.0%}, MOM sig.: {sig:.0%}, Simplicity: {simp:.0%}, Economic: {econ:.0%}."
    ).format(
        delta=WEIGHTS["delta_r2"],
        sig=WEIGHTS["mom_sig_rate"],
        simp=WEIGHTS["simplicity"],
        econ=WEIGHTS["economic"],
    )
    sources = (
        f"Sources: `{paths.performance_matrix}` (performance matrix) and "
        f"`{paths.regression_summary}` (regression summary)."
    )
    content = "\n".join(
        [
            "# Momentum Optimization Report",
            f"Generated: {timestamp}",
            sources,
            "",
            recommendation,
            "",
            "## Ranking",
            *table_lines,
            "",
            "## Methodology",
            methodology,
            "",
            "The scorecard rewards models that improve FF5 ΔR², retain significant MOM coefficients, "
            "and stay simple for implementation.",
        ]
    )
    path.write_text(content, encoding="utf-8")
    return recommendation


def main() -> None:
    paths = parse_args()
    perf_df = load_table(paths.performance_matrix, PERFORMANCE_REQUIRED_COLUMNS)
    summary_df = load_table(paths.regression_summary, SUMMARY_REQUIRED_COLUMNS)
    scorecard = build_model_metrics(perf_df, summary_df)
    write_scorecard(scorecard, paths.scorecard_output)
    recommendation = generate_report(scorecard, paths)
    winner = scorecard.iloc[0]
    summary_lines = [
        "Momentum optimization complete.",
        f"Scorecard → {paths.scorecard_output}",
        f"Report → {paths.report_output}",
        f"Winner: {winner['Model']} (score {winner['Optimization_Score']:.1f}).",
        f"ΔR²: {winner['Mean_Delta_R2_pct']:.2f} pp | MOM sig.: {winner['MOM_Significance_Rate_pct']:.1f}%",
        recommendation,
    ]
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
