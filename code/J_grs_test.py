"""GRS joint significance test and FF-model comparison utilities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

PORTFOLIO_CONFIG: Tuple[Tuple[str, str, str], ...] = (
    ("portfolio_returns_size_bm.parquet", "size_bm", "bm_quintile"),
    ("portfolio_returns_size_op.parquet", "size_op", "op_quintile"),
    ("portfolio_returns_size_inv.parquet", "size_inv", "inv_quintile"),
)

FF3_COLUMNS: Sequence[str] = ("MKT_RF", "SMB", "HML")
FF5_COLUMNS: Sequence[str] = ("MKT_RF", "SMB", "HML", "RMW", "CMA")


@dataclass(frozen=True)
class ModelInputs:
    name: str
    factor_cols: Sequence[str]
    factor_df: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FF3/FF5/FF6 regressions, compute GRS tests, and export tables."
    )
    parser.add_argument(
        "--input-dir",
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
        help="Path to FF6 factor returns parquet (FF5 plus the sixth factor).",
    )
    parser.add_argument(
        "--ff5-results",
        type=Path,
        default=Path("docs/tables/regression_results_ff5.csv"),
        help="Existing FF5 regression table from G_regression_analysis.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/tables"),
        help="Directory to write grs_test.csv and model_comparison.csv.",
    )
    parser.add_argument(
        "--maxlags",
        type=int,
        default=4,
        help="Newey-West lag length for HAC covariance.",
    )
    parser.add_argument(
        "--min-obs",
        type=int,
        default=8,
        help="Minimum observations per portfolio required to run regressions.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console prints; useful when invoked from other scripts.",
    )
    return parser.parse_args()


def load_factor_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing factor file: {path}")
    df = pd.read_parquet(path).copy()
    if "month" not in df.columns:
        raise ValueError(f"Factor file {path} missing month column")
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    return df.sort_values("month").reset_index(drop=True)


def load_portfolio_data(input_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for filename, portfolio_type, factor_col in PORTFOLIO_CONFIG:
        file_path = input_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing portfolio file: {file_path}")

        df = pd.read_parquet(file_path).copy()
        required = {"month", "size_quintile", factor_col, "excess_return"}
        missing = required - set(df.columns)
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
    portfolios["portfolio_id"] = (
        portfolios["portfolio_type"]
        + "_s"
        + portfolios["size_quintile"].astype(str)
        + "_f"
        + portfolios["factor_quintile"].astype(str)
    )
    return portfolios


def alpha_sig_rate(p_values: pd.Series, threshold: float = 0.05) -> float:
    valid = p_values.dropna()
    if valid.empty:
        return float("nan")
    return float((valid < threshold).mean())


def summarize_model(results: pd.DataFrame) -> Dict[str, float]:
    return {
        "avg_r_squared": float(results["r_squared"].mean(skipna=True)),
        "avg_adj_r_squared": float(results["adj_r_squared"].mean(skipna=True)),
        "avg_alpha": float(results["alpha"].mean(skipna=True)),
        "alpha_sig_rate": alpha_sig_rate(results["p_alpha"]),
    }


def init_result_record(meta: Dict[str, int | str], factor_cols: Sequence[str], model: str) -> Dict:
    record: Dict[str, float | int | str] = {
        "model": model,
        "portfolio_type": meta["portfolio_type"],
        "size_quintile": meta["size_quintile"],
        "factor_quintile": meta["factor_quintile"],
        "portfolio_id": meta["portfolio_id"],
        "n_obs": meta["n_obs"],
        "alpha": np.nan,
        "t_alpha": np.nan,
        "p_alpha": np.nan,
        "r_squared": np.nan,
        "adj_r_squared": np.nan,
    }
    for col in factor_cols:
        record[f"beta_{col}"] = np.nan
        record[f"t_{col}"] = np.nan
    return record


def run_time_series_regressions(
    portfolios: pd.DataFrame,
    factors: pd.DataFrame,
    factor_cols: Sequence[str],
    *,
    model_name: str,
    maxlags: int,
    min_obs: int,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    missing = [col for col in factor_cols if col not in factors.columns]
    if missing:
        raise ValueError(f"Factor dataset missing required columns: {missing}")

    merged = portfolios.merge(factors, on="month", how="left", validate="m:1")
    records: List[Dict] = []
    residuals: Dict[str, pd.Series] = {}
    for (ptype, size_q, factor_q), group in merged.groupby(
        ["portfolio_type", "size_quintile", "factor_quintile"], sort=True
    ):
        subset = group.loc[:, ["month", "excess_return", *factor_cols]].dropna()
        n_obs = int(len(subset))
        portfolio_id = f"{ptype}_s{int(size_q)}_f{int(factor_q)}"
        meta = {
            "portfolio_type": ptype,
            "size_quintile": int(size_q),
            "factor_quintile": int(factor_q),
            "portfolio_id": portfolio_id,
            "n_obs": n_obs,
        }
        record = init_result_record(meta, factor_cols, model_name)
        if n_obs < min_obs:
            records.append(record)
            continue

        y = subset["excess_return"].astype(float)
        X = sm.add_constant(subset[list(factor_cols)].astype(float))
        hac_lags = max(0, min(maxlags, n_obs - 1))
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
        for col in factor_cols:
            record[f"beta_{col}"] = float(fit.params.get(col, np.nan))
            record[f"t_{col}"] = float(fit.tvalues.get(col, np.nan))

        records.append(record)

        residual_series = pd.Series(
            (y - fit.predict(X)).to_numpy(),
            index=subset["month"],
            name=portfolio_id,
        )
        residuals[portfolio_id] = residual_series

    return pd.DataFrame.from_records(records), residuals


def residual_panel(residuals: Dict[str, pd.Series]) -> pd.DataFrame:
    if not residuals:
        return pd.DataFrame()
    frames = [series.rename(pid) for pid, series in residuals.items()]
    panel = pd.concat(frames, axis=1, join="outer").sort_index()
    return panel.dropna()


def align_with_factors(
    panel: pd.DataFrame, factor_df: pd.DataFrame, factor_cols: Sequence[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if panel.empty:
        return panel, pd.DataFrame()

    factors = factor_df.set_index("month").loc[:, list(factor_cols)]
    common_index = panel.index.intersection(factors.index)
    if common_index.empty:
        return pd.DataFrame(), pd.DataFrame()

    panel_sub = panel.loc[common_index]
    factor_sub = factors.loc[common_index].dropna()
    aligned_index = panel_sub.index.intersection(factor_sub.index)
    if aligned_index.empty:
        return pd.DataFrame(), pd.DataFrame()

    return panel_sub.loc[aligned_index], factor_sub.loc[aligned_index]


def compute_grs_components(
    results: pd.DataFrame,
    resids: Dict[str, pd.Series],
    factor_df: pd.DataFrame,
    factor_cols: Sequence[str],
) -> Tuple[float, float]:
    panel = residual_panel(resids)
    panel, factors = align_with_factors(panel, factor_df, factor_cols)
    if panel.empty or factors.empty:
        return float("nan"), float("nan")

    ordered_alpha = (
        results.set_index("portfolio_id")
        .reindex(panel.columns)["alpha"]
        .astype(float)
    )
    valid_mask = ordered_alpha.notna()
    if not valid_mask.any():
        return float("nan"), float("nan")

    use_columns = list(valid_mask.index[valid_mask])
    panel = panel.loc[:, use_columns]
    alpha_vec = ordered_alpha.loc[use_columns].to_numpy(dtype=float)

    if panel.shape[1] == 0:
        return float("nan"), float("nan")

    T = panel.shape[0]
    N = panel.shape[1]
    k = len(factor_cols)
    if T <= N + k or N == 0 or k == 0:
        return float("nan"), float("nan")

    resid_cov = panel.cov().to_numpy()
    factor_cov = factors.cov().to_numpy()
    factor_means = factors.mean().to_numpy().reshape(-1, 1)

    inv_resid = np.linalg.pinv(resid_cov)
    inv_factor = np.linalg.pinv(factor_cov)
    alpha_vec = alpha_vec.reshape(-1, 1)
    numerator = float(alpha_vec.T @ inv_resid @ alpha_vec)
    lambda_term = float(factor_means.T @ inv_factor @ factor_means)
    denom = 1.0 + lambda_term
    if denom <= 0:
        return float("nan"), float("nan")

    prefactor = (T - N - k) / N
    if prefactor <= 0:
        return float("nan"), float("nan")

    grs_stat = float(prefactor * (numerator / denom))
    df1 = N
    df2 = T - N - k
    if df2 <= 0:
        return float("nan"), float("nan")

    p_value = float(1.0 - stats.f.cdf(grs_stat, df1, df2))
    return grs_stat, p_value


def run_grs_analysis(
    input_dir: Path,
    factor_ff5_path: Path,
    factor_ff6_path: Path,
    ff5_results_path: Path,
    output_dir: Path,
    *,
    maxlags: int,
    min_obs: int,
    verbose: bool = True,
    write_outputs: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ff5_existing = pd.read_csv(ff5_results_path)
    required_cols = {"alpha", "p_alpha", "r_squared", "adj_r_squared"}
    if not required_cols <= set(ff5_existing.columns):
        missing = required_cols - set(ff5_existing.columns)
        raise ValueError(
            f"FF5 regression output missing required columns: {sorted(missing)}"
        )

    portfolios = load_portfolio_data(input_dir)
    factor_ff5 = load_factor_data(factor_ff5_path)
    factor_ff6 = load_factor_data(factor_ff6_path)
    ff6_cols = [col for col in factor_ff6.columns if col != "month"]
    if len(ff6_cols) < 6:
        raise ValueError("FF6 factor file must contain at least six factor columns")

    model_inputs: Dict[str, ModelInputs] = {
        "FF3": ModelInputs("FF3", FF3_COLUMNS, factor_ff5),
        "FF5": ModelInputs("FF5", FF5_COLUMNS, factor_ff5),
        "FF6": ModelInputs("FF6", ff6_cols, factor_ff6),
    }

    model_runs: Dict[str, Dict[str, object]] = {}
    for name, config in model_inputs.items():
        results, resids = run_time_series_regressions(
            portfolios,
            config.factor_df,
            config.factor_cols,
            model_name=name,
            maxlags=maxlags,
            min_obs=min_obs,
        )
        model_runs[name] = {
            "results": results,
            "residuals": resids,
            "factors": config.factor_df,
            "factor_cols": config.factor_cols,
        }

    grs_records: List[Dict[str, float]] = []
    for name, run in model_runs.items():
        grs_stat, p_value = compute_grs_components(
            run["results"],
            run["residuals"],
            run["factors"],
            run["factor_cols"],
        )
        summary = summarize_model(run["results"])
        grs_records.append(
            {
                "Model": name,
                "GRS_stat": grs_stat,
                "p_value": p_value,
                "avg_alpha": summary["avg_alpha"],
                "alpha_sig_rate": summary["alpha_sig_rate"],
            }
        )

    grs_df = pd.DataFrame(grs_records).sort_values("Model").reset_index(drop=True)

    model_comparison_records: List[Dict[str, float]] = []
    for name in ("FF3", "FF5", "FF6"):
        if name == "FF5":
            summary = summarize_model(ff5_existing)
        else:
            summary = summarize_model(model_runs[name]["results"])
        record = {"Model": name}
        record.update(summary)
        model_comparison_records.append(record)

    model_comparison_df = (
        pd.DataFrame(model_comparison_records)
        .sort_values("Model")
        .reset_index(drop=True)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if write_outputs:
        grs_df.to_csv(output_dir / "grs_test.csv", index=False)
        model_comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)

    if verbose:
        for row in grs_df.itertuples(index=False):
            if np.isnan(row.GRS_stat) or np.isnan(row.p_value):
                print(f"{row.Model}: insufficient data for GRS test")
                continue
            verdict = "显著" if row.p_value < 0.05 else "不显著"
            print(
                f"{row.Model}: GRS={row.GRS_stat:.3f}, p={row.p_value:.3g} -> {verdict}"
            )
        best_model = select_best_model(model_comparison_df)
        if best_model:
            print(f"模型优选建议：{best_model}")

    return grs_df, model_comparison_df


def select_best_model(model_comp: pd.DataFrame) -> str | None:
    if model_comp.empty:
        return None
    ranked = model_comp.sort_values(
        by=["alpha_sig_rate", "avg_adj_r_squared", "avg_r_squared"],
        ascending=[True, False, False],
    )
    return str(ranked.iloc[0]["Model"])


def main() -> None:
    args = parse_args()
    run_grs_analysis(
        args.input_dir,
        args.factor_ff5,
        args.factor_ff6,
        args.ff5_results,
        args.output_dir,
        maxlags=max(args.maxlags, 0),
        min_obs=max(args.min_obs, 1),
        verbose=not args.quiet,
        write_outputs=True,
    )


if __name__ == "__main__":
    main()
