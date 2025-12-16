"""Run Fama-French 5-factor time-series regressions on portfolio returns."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm

FACTOR_COLUMNS: List[str] = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]

PORTFOLIO_CONFIG = (
    ("portfolio_returns_size_bm.parquet", "size_bm", "bm_quintile"),
    ("portfolio_returns_size_op.parquet", "size_op", "op_quintile"),
    ("portfolio_returns_size_inv.parquet", "size_inv", "inv_quintile"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=
        "Run FF5 time-series regressions (with Newey-West SEs) across 75 portfolios."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing portfolio and factor parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/tables"),
        help="Directory to place regression result tables.",
    )
    parser.add_argument(
        "--maxlags",
        type=int,
        default=4,
        help="Newey-West lag length for HAC covariance estimates.",
    )
    parser.add_argument(
        "--min-obs",
        type=int,
        default=8,
        help=(
            "Minimum observations per portfolio (after dropping NA rows) required to "
            "fit each regression."
        ),
    )
    return parser.parse_args()


def load_factor_data(path: Path) -> pd.DataFrame:
    factors = pd.read_parquet(path).copy()
    factors["month"] = pd.to_datetime(factors["month"], format="%Y-%m")
    return factors


def load_portfolio_data(input_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for filename, portfolio_type, factor_col in PORTFOLIO_CONFIG:
        full_path = input_dir / filename
        if not full_path.exists():
            raise FileNotFoundError(f"Missing portfolio file: {full_path}")

        df = pd.read_parquet(full_path).copy()
        required_cols = {"month", "size_quintile", factor_col, "excess_return"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{filename} missing columns: {missing}")

        subset = df.loc[:, ["month", "size_quintile", factor_col, "excess_return"]]
        subset = subset.rename(columns={factor_col: "factor_quintile"})
        subset["portfolio_type"] = portfolio_type
        subset["month"] = pd.to_datetime(subset["month"], format="%Y-%m")
        frames.append(subset)

    portfolios = pd.concat(frames, ignore_index=True)
    portfolios["size_quintile"] = portfolios["size_quintile"].astype(int)
    portfolios["factor_quintile"] = portfolios["factor_quintile"].astype(int)
    return portfolios


def format_result_record(meta: Dict, model_stats: Dict | None) -> Dict:
    base = {
        "portfolio_type": meta["portfolio_type"],
        "size_quintile": meta["size_quintile"],
        "factor_quintile": meta["factor_quintile"],
        "alpha": np.nan,
        "t_alpha": np.nan,
        "p_alpha": np.nan,
        "beta_MKT_RF": np.nan,
        "t_MKT_RF": np.nan,
        "beta_SMB": np.nan,
        "t_SMB": np.nan,
        "beta_HML": np.nan,
        "t_HML": np.nan,
        "beta_RMW": np.nan,
        "t_RMW": np.nan,
        "beta_CMA": np.nan,
        "t_CMA": np.nan,
        "r_squared": np.nan,
        "adj_r_squared": np.nan,
        "n_obs": meta["n_obs"],
    }

    if model_stats is None:
        return base

    base.update(model_stats)
    return base


def run_regressions(
    portfolios: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    maxlags: int,
    min_obs: int,
) -> pd.DataFrame:
    merged = portfolios.merge(factors, on="month", how="left", validate="m:1")
    records: List[Dict] = []

    for (ptype, size_q, factor_q), group in merged.groupby(
        ["portfolio_type", "size_quintile", "factor_quintile"], sort=True
    ):
        subset = group.loc[:, ["excess_return"] + FACTOR_COLUMNS].dropna()
        n_obs = int(len(subset))
        meta = {
            "portfolio_type": ptype,
            "size_quintile": int(size_q),
            "factor_quintile": int(factor_q),
            "n_obs": n_obs,
        }

        if n_obs < min_obs:
            records.append(format_result_record(meta, None))
            continue

        y = subset["excess_return"].astype(float)
        X = sm.add_constant(subset[FACTOR_COLUMNS].astype(float))

        hac_lags = max(0, min(maxlags, n_obs - 1))
        try:
            fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        except Exception as exc:  # pragma: no cover - defensive logging
            print(
                f"Failed regression for {ptype} size={size_q} factor={factor_q}: {exc}"
            )
            records.append(format_result_record(meta, None))
            continue

        stats = {
            "alpha": fit.params.get("const", np.nan),
            "t_alpha": fit.tvalues.get("const", np.nan),
            "p_alpha": fit.pvalues.get("const", np.nan),
            "beta_MKT_RF": fit.params.get("MKT_RF", np.nan),
            "t_MKT_RF": fit.tvalues.get("MKT_RF", np.nan),
            "beta_SMB": fit.params.get("SMB", np.nan),
            "t_SMB": fit.tvalues.get("SMB", np.nan),
            "beta_HML": fit.params.get("HML", np.nan),
            "t_HML": fit.tvalues.get("HML", np.nan),
            "beta_RMW": fit.params.get("RMW", np.nan),
            "t_RMW": fit.tvalues.get("RMW", np.nan),
            "beta_CMA": fit.params.get("CMA", np.nan),
            "t_CMA": fit.tvalues.get("CMA", np.nan),
            "r_squared": fit.rsquared,
            "adj_r_squared": fit.rsquared_adj,
        }
        records.append(format_result_record(meta, stats))

    return pd.DataFrame.from_records(records)


def summarize_by_type(results: pd.DataFrame) -> pd.DataFrame:
    def alpha_share(series: pd.Series) -> float:
        valid = series.dropna()
        if valid.empty:
            return np.nan
        return (valid < 0.05).mean()

    grouped = (
        results.groupby("portfolio_type", dropna=False)
        .agg(
            mean_alpha=("alpha", "mean"),
            alpha_sig_share=("p_alpha", alpha_share),
            mean_r_squared=("r_squared", "mean"),
            mean_adj_r_squared=("adj_r_squared", "mean"),
            mean_n_obs=("n_obs", "mean"),
            n_portfolios=("portfolio_type", "count"),
        )
        .reset_index()
    )
    return grouped


def save_outputs(results: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = output_dir / "regression_results_ff5.csv"
    summary_path = output_dir / "regression_summary.csv"

    column_order = [
        "portfolio_type",
        "size_quintile",
        "factor_quintile",
        "alpha",
        "t_alpha",
        "p_alpha",
        "beta_MKT_RF",
        "t_MKT_RF",
        "beta_SMB",
        "t_SMB",
        "beta_HML",
        "t_HML",
        "beta_RMW",
        "t_RMW",
        "beta_CMA",
        "t_CMA",
        "r_squared",
        "adj_r_squared",
        "n_obs",
    ]
    results.loc[:, column_order].to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)


def print_summary(results: pd.DataFrame) -> None:
    total = len(results)
    successful = results["alpha"].notna().sum()
    mean_r2 = results["r_squared"].mean(skipna=True)
    valid_alpha_p = results["p_alpha"].dropna()
    alpha_sig_share = (
        (valid_alpha_p < 0.05).mean() if not valid_alpha_p.empty else np.nan
    )
    alpha_sig_str = f"{alpha_sig_share:.3f}" if not np.isnan(alpha_sig_share) else "nan"
    print(
        "FF5 regression summary: "
        f"{successful}/{total} portfolios with valid fits | "
        f"avg R^2={mean_r2:.3f} | "
        f"alpha sig share (p<0.05)={alpha_sig_str}"
    )


def print_grs_recommendation(
    input_dir: Path, output_dir: Path, maxlags: int, min_obs: int
) -> None:
    try:
        from J_grs_test import run_grs_analysis, select_best_model
    except ImportError:
        print("GRS test skipped: missing code/J_grs_test.py")
        return

    grs_inputs = {
        "input_dir": input_dir,
        "factor_ff5_path": input_dir / "factor_returns_ff5.parquet",
        "factor_ff6_path": input_dir / "factor_returns_ff6.parquet",
        "ff5_results_path": output_dir / "regression_results_ff5.csv",
        "output_dir": output_dir,
    }

    try:
        grs_df, comparison_df = run_grs_analysis(
            grs_inputs["input_dir"],
            grs_inputs["factor_ff5_path"],
            grs_inputs["factor_ff6_path"],
            grs_inputs["ff5_results_path"],
            grs_inputs["output_dir"],
            maxlags=max(maxlags, 0),
            min_obs=max(min_obs, 1),
            verbose=False,
            write_outputs=True,
        )
    except FileNotFoundError as exc:
        print(f"GRS test skipped: {exc}")
        return
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"GRS test skipped due to error: {exc}")
        return

    if grs_df.empty:
        print("GRS test produced no valid output.")
        return

    for row in grs_df.itertuples(index=False):
        if np.isnan(row.GRS_stat) or np.isnan(row.p_value):
            print(f"{row.Model}: insufficient data for GRS test")
            continue
        verdict = "显著" if row.p_value < 0.05 else "不显著"
        print(f"{row.Model} GRS={row.GRS_stat:.3f}, p={row.p_value:.3g} -> {verdict}")

    best = select_best_model(comparison_df)
    if best:
        print(f"模型优选建议：{best}")


def main() -> None:
    args = parse_args()
    factor_path = args.input_dir / "factor_returns_ff5.parquet"
    if not factor_path.exists():
        raise FileNotFoundError(f"Missing factor file: {factor_path}")

    factors = load_factor_data(factor_path)
    portfolios = load_portfolio_data(args.input_dir)

    results = run_regressions(
        portfolios,
        factors,
        maxlags=max(args.maxlags, 0),
        min_obs=max(args.min_obs, 1),
    )
    summary = summarize_by_type(results)
    save_outputs(results, summary, args.output_dir)
    print_summary(results)
    print_grs_recommendation(args.input_dir, args.output_dir, args.maxlags, args.min_obs)


if __name__ == "__main__":
    main()
