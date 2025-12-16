"""Run Fama-MacBeth cross-sectional regressions for FF5 and FF6 factors."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

PORTFOLIO_CONFIG = (
    ("portfolio_returns_size_bm.parquet", "size_bm", "bm_quintile"),
    ("portfolio_returns_size_op.parquet", "size_op", "op_quintile"),
    ("portfolio_returns_size_inv.parquet", "size_inv", "inv_quintile"),
)


@dataclass(frozen=True)
class ModelConfig:
    label: str
    factor_path: Path
    betas_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Fama-MacBeth cross-sectional regressions for FF5 and FF6."
    )
    parser.add_argument(
        "--portfolio-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing portfolio parquet files.",
    )
    parser.add_argument(
        "--factor-ff5",
        type=Path,
        default=Path("data/processed/factor_returns_ff5.parquet"),
        help="Path to FF5 factor returns parquet.",
    )
    parser.add_argument(
        "--factor-ff6",
        type=Path,
        default=Path("data/processed/factor_returns_ff6.parquet"),
        help="Path to FF6 factor returns parquet.",
    )
    parser.add_argument(
        "--betas-ff5",
        type=Path,
        default=Path("docs/tables/regression_results_ff5.csv"),
        help="Path to FF5 time-series regression betas (from G_regression_analysis.py).",
    )
    parser.add_argument(
        "--betas-ff6",
        type=Path,
        default=Path("docs/tables/regression_results_ff6.csv"),
        help="Path to FF6 time-series regression betas (from G_regression_analysis.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/tables"),
        help="Directory for Fama-MacBeth summary tables.",
    )
    parser.add_argument(
        "--min-portfolios",
        type=int,
        default=30,
        help="Minimum #portfolios per month required to run a cross-sectional regression.",
    )
    return parser.parse_args()


def load_factor_returns(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing factor file: {path}")
    df = pd.read_parquet(path).copy()
    if "month" not in df.columns:
        raise ValueError(f"Factor file {path} missing month column")
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    return df


def infer_factor_names(factor_df: pd.DataFrame) -> List[str]:
    factor_cols = [c for c in factor_df.columns if c != "month"]
    if not factor_cols:
        raise ValueError("Factor dataset has no factor columns")
    return factor_cols


def load_portfolio_returns(input_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for filename, portfolio_type, quantile_col in PORTFOLIO_CONFIG:
        file_path = input_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing portfolio file: {file_path}")

        df = pd.read_parquet(file_path).copy()
        required = {"month", "size_quintile", quantile_col, "excess_return"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{filename} missing columns: {missing}")

        subset = df.loc[:, ["month", "size_quintile", quantile_col, "excess_return"]]
        subset = subset.rename(columns={quantile_col: "factor_quintile"})
        subset["portfolio_type"] = portfolio_type
        subset["month"] = pd.to_datetime(subset["month"], format="%Y-%m")
        frames.append(subset)

    portfolios = pd.concat(frames, ignore_index=True)
    portfolios["size_quintile"] = portfolios["size_quintile"].astype(int)
    portfolios["factor_quintile"] = portfolios["factor_quintile"].astype(int)
    return portfolios


def load_betas(path: Path, factor_names: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing regression result file: {path}")

    df = pd.read_csv(path).copy()
    key_cols = ["portfolio_type", "size_quintile", "factor_quintile"]
    missing = set(key_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing key columns: {missing}")

    beta_cols = [f"beta_{name}" for name in factor_names]
    missing_betas = [col for col in beta_cols if col not in df.columns]
    if missing_betas:
        raise ValueError(
            "Regression outputs do not contain required beta columns: "
            + ", ".join(missing_betas)
        )

    subset = df.loc[:, key_cols + beta_cols].copy()
    subset["size_quintile"] = subset["size_quintile"].astype(int)
    subset["factor_quintile"] = subset["factor_quintile"].astype(int)
    return subset


def merge_portfolios_with_betas(
    portfolios: pd.DataFrame, betas: pd.DataFrame
) -> pd.DataFrame:
    merged = portfolios.merge(
        betas, on=["portfolio_type", "size_quintile", "factor_quintile"], how="inner"
    )
    return merged


def run_monthly_cross_sections(
    panel: pd.DataFrame,
    factor_names: Sequence[str],
    *,
    min_portfolios: int,
) -> Dict[str, List[float]]:
    beta_cols = [f"beta_{name}" for name in factor_names]
    needed = ["excess_return"] + beta_cols
    panel = panel.dropna(subset=needed)

    lambda_history: Dict[str, List[float]] = {"const": []}
    for factor in factor_names:
        lambda_history[factor] = []

    for month, group in panel.groupby("month", sort=True):
        if len(group) < min_portfolios or len(group) <= len(beta_cols):
            continue

        X = group.loc[:, beta_cols].astype(float)
        X = sm.add_constant(X)
        y = group["excess_return"].astype(float)

        # Skip ill-conditioned cross-sections to avoid singular matrices.
        if np.linalg.matrix_rank(X) < X.shape[1]:
            continue

        fit = sm.OLS(y, X).fit()
        params = fit.params.to_dict()
        for param_name, value in params.items():
            lambda_history.setdefault(param_name, []).append(float(value))

    return lambda_history


def summarize_lambdas(lambda_history: Dict[str, List[float]]) -> pd.DataFrame:
    records = []
    for factor, values in lambda_history.items():
        if not values:
            records.append(
                {
                    "factor": factor,
                    "lambda_mean": np.nan,
                    "lambda_std": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                }
            )
            continue

        series = np.asarray(values, dtype=float)
        mean_val = float(series.mean())
        std_val = float(series.std(ddof=1)) if len(series) > 1 else np.nan
        if len(series) > 1 and np.isfinite(std_val) and std_val > 0:
            se = std_val / np.sqrt(len(series))
            t_stat = mean_val / se
            p_val = 2 * stats.t.sf(abs(t_stat), df=len(series) - 1)
        else:
            t_stat = np.nan
            p_val = np.nan

        records.append(
            {
                "factor": factor,
                "lambda_mean": mean_val,
                "lambda_std": std_val,
                "t_stat": t_stat,
                "p_value": p_val,
            }
        )

    df = pd.DataFrame.from_records(records)
    return df


def identify_significant(summary: pd.DataFrame, alpha: float = 0.05) -> List[str]:
    mask = summary["p_value"].notna() & (summary["p_value"] < alpha)
    return summary.loc[mask, "factor"].tolist()


def run_model(
    *,
    config: ModelConfig,
    portfolios: pd.DataFrame,
    factor_df: pd.DataFrame,
    min_portfolios: int,
    output_dir: Path,
) -> pd.DataFrame:
    factor_names = infer_factor_names(factor_df)
    betas = load_betas(config.betas_path, factor_names)

    panel = merge_portfolios_with_betas(portfolios, betas)
    valid_months = factor_df["month"].dropna().unique()
    panel = panel[panel["month"].isin(valid_months)]

    lambda_history = run_monthly_cross_sections(
        panel, factor_names, min_portfolios=min_portfolios
    )
    summary = summarize_lambdas(lambda_history)
    summary_path = output_dir / f"fama_macbeth_{config.label}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    return summary


def format_significance_message(
    model_label: str, factors: Sequence[str], reference: str
) -> str:
    if not factors:
        return f"{model_label}: no significant factors (alpha<0.05); reference {reference}."
    joined = ", ".join(factors)
    return f"{model_label}: significant factors {joined}; reference {reference}."


def compare_and_print(ff5_summary: pd.DataFrame, ff6_summary: pd.DataFrame) -> None:
    ff5_sig = identify_significant(ff5_summary)
    ff6_sig = identify_significant(ff6_summary)

    print(
        format_significance_message(
            "FF5",
            ff5_sig,
            "code/G_regression_analysis.py & data/processed/factor_returns_ff5.parquet",
        )
    )
    print(
        format_significance_message(
            "FF6",
            ff6_sig,
            "code/G_regression_analysis.py & data/processed/factor_returns_ff6.parquet",
        )
    )

    ff5_only = sorted(set(ff5_sig) - set(ff6_sig))
    ff6_only = sorted(set(ff6_sig) - set(ff5_sig))
    overlap = sorted(set(ff5_sig) & set(ff6_sig))

    overlap_str = ", ".join(overlap) if overlap else "None"
    ff5_only_str = ", ".join(ff5_only) if ff5_only else "None"
    ff6_only_str = ", ".join(ff6_only) if ff6_only else "None"

    print(f"Common significant: {overlap_str}")
    print(f"Only FF5 significant: {ff5_only_str}")
    print(f"Only FF6 significant: {ff6_only_str}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir

    portfolios = load_portfolio_returns(args.portfolio_dir)

    ff5_config = ModelConfig(
        label="ff5",
        factor_path=args.factor_ff5,
        betas_path=args.betas_ff5,
    )
    ff6_config = ModelConfig(
        label="ff6",
        factor_path=args.factor_ff6,
        betas_path=args.betas_ff6,
    )

    ff5_factors = load_factor_returns(ff5_config.factor_path)
    ff6_factors = load_factor_returns(ff6_config.factor_path)

    ff5_summary = run_model(
        config=ff5_config,
        portfolios=portfolios,
        factor_df=ff5_factors,
        min_portfolios=args.min_portfolios,
        output_dir=output_dir,
    )
    ff6_summary = run_model(
        config=ff6_config,
        portfolios=portfolios,
        factor_df=ff6_factors,
        min_portfolios=args.min_portfolios,
        output_dir=output_dir,
    )

    compare_and_print(ff5_summary, ff6_summary)


if __name__ == "__main__":
    main()
