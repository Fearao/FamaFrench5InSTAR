"""Portfolio-type and quintile-level robustness for FF3+MOM_EQ."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
SIG_LEVEL = 0.10

OUTPUT_DIR = Path("docs/tables")
TYPE_OUTPUT = OUTPUT_DIR / "robustness_by_portfolio_type_EQ.csv"
QUINTILE_OUTPUT = OUTPUT_DIR / "robustness_quintile_breakdown_EQ.csv"


def load_merged_data(input_dir: str = "data/processed") -> pd.DataFrame:
    """Return merged factor/panel data for all 75 portfolios."""

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


def filter_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.to_datetime(start, format="%Y-%m")
    end_ts = pd.to_datetime(end, format="%Y-%m")
    mask = (df["month"] >= start_ts) & (df["month"] <= end_ts)
    return df.loc[mask].copy()


def run_period_regressions(period_df: pd.DataFrame) -> pd.DataFrame:
    """Run FF3+MOM regressions for all portfolios inside one period."""

    results = []
    group_cols = ["portfolio_type", "size_quintile", "factor_quintile"]

    for (ptype, size_q, factor_q), group in period_df.groupby(group_cols, sort=True):
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
                }
            )

            for col in FACTOR_COLS:
                record[f"beta_{col}"] = float(fit.params.get(col, np.nan))
                record[f"p_{col}"] = float(fit.pvalues.get(col, np.nan))

        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"回归失败 {ptype} s{size_q} f{factor_q}: {exc}")

        results.append(record)

    return pd.DataFrame(results)


def summarize_groups(results_df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Aggregate MOM metrics for the provided grouping columns."""

    beta_col = f"beta_{MOM_FACTOR}"
    p_col = f"p_{MOM_FACTOR}"
    rows: List[Dict[str, float]] = []

    for keys, group in results_df.groupby(list(group_cols), sort=True):
        if isinstance(keys, tuple):
            key_values = list(keys)
        else:
            key_values = [keys]

        record: Dict[str, float] = {col: val for col, val in zip(group_cols, key_values)}

        valid = group[group["alpha"].notna()].copy()
        beta_valid = valid[beta_col].notna()
        beta_df = valid.loc[beta_valid].copy()

        record["n_portfolios"] = int(beta_df.shape[0])

        if beta_df.empty:
            record.update(
                {
                    "mom_sig": np.nan,
                    "mean_r2": np.nan,
                    "mean_beta": np.nan,
                    "mean_abs_beta": np.nan,
                }
            )
        else:
            record["mom_sig"] = (beta_df[p_col] < SIG_LEVEL).mean() * 100
            record["mean_r2"] = beta_df["r_squared"].mean()
            record["mean_beta"] = beta_df[beta_col].mean()
            record["mean_abs_beta"] = beta_df[beta_col].abs().mean()

        rows.append(record)

    return pd.DataFrame(rows)


def build_type_table(type_period_summaries: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine per-period summaries into the type-level CSV."""

    metrics = ["mom_sig", "mean_r2", "mean_abs_beta", "n_portfolios"]
    base = None

    for period_key, df in type_period_summaries.items():
        rename_map = {col: f"{period_key}_{col}" for col in metrics}
        current = df.rename(columns=rename_map)
        if base is None:
            base = current
        else:
            base = base.merge(current, on="portfolio_type", how="outer")

    if base is None:
        return pd.DataFrame()

    base["delta_mom_sig"] = base["period2_mom_sig"] - base["period1_mom_sig"]
    base["delta_r2"] = base["period2_mean_r2"] - base["period1_mean_r2"]
    base["stability_score"] = base["delta_mom_sig"].abs() + base["delta_r2"].abs() * 100

    result_cols = [
        "portfolio_type",
        "period1_mom_sig",
        "period2_mom_sig",
        "delta_mom_sig",
        "period1_mean_r2",
        "period2_mean_r2",
        "delta_r2",
        "period1_mean_abs_beta",
        "period2_mean_abs_beta",
        "stability_score",
        "period1_n_portfolios",
        "period2_n_portfolios",
    ]

    base = base[result_cols]
    base = base.rename(
        columns={
            "period1_mean_r2": "period1_r2",
            "period2_mean_r2": "period2_r2",
            "period1_n_portfolios": "n_portfolios_p1",
            "period2_n_portfolios": "n_portfolios_p2",
        }
    )

    return base.sort_values("stability_score", ascending=True)


def build_quintile_table(size_period_summaries: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine per-period summaries into size-quintile level CSV."""

    metrics = ["mom_sig", "mean_r2", "mean_beta", "n_portfolios"]
    base = None

    for period_key, df in size_period_summaries.items():
        rename_map = {col: f"{period_key}_{col}" for col in metrics}
        current = df.rename(columns=rename_map)
        if base is None:
            base = current
        else:
            base = base.merge(current, on=["portfolio_type", "size_quintile"], how="outer")

    if base is None:
        return pd.DataFrame()

    base["delta_mom_sig"] = base["period2_mom_sig"] - base["period1_mom_sig"]
    base["factor_quintile"] = "all"
    base["n_portfolios"] = base[["period1_n_portfolios", "period2_n_portfolios"]].max(axis=1)

    result_cols = [
        "portfolio_type",
        "size_quintile",
        "factor_quintile",
        "period1_mom_sig",
        "period2_mom_sig",
        "delta_mom_sig",
        "period1_mean_beta",
        "period2_mean_beta",
        "period1_mean_r2",
        "period2_mean_r2",
        "n_portfolios",
    ]

    base = base[result_cols]
    return base.sort_values(["portfolio_type", "size_quintile"])


def main() -> None:
    print("加载合并数据 (MOM_EQ)...")
    merged = load_merged_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    type_summaries: Dict[str, pd.DataFrame] = {}
    size_summaries: Dict[str, pd.DataFrame] = {}

    for period in PERIOD_DEFS:
        key = period["key"]
        print(f"处理 {period['label']}...")
        subset = filter_period(merged, period["start"], period["end"])
        results_df = run_period_regressions(subset)
        type_summaries[key] = summarize_groups(results_df, ["portfolio_type"])
        size_summaries[key] = summarize_groups(results_df, ["portfolio_type", "size_quintile"])

    type_table = build_type_table(type_summaries)
    quintile_table = build_quintile_table(size_summaries)

    type_table.to_csv(TYPE_OUTPUT, index=False)
    quintile_table.to_csv(QUINTILE_OUTPUT, index=False)

    print(f"类型级稳健性已写入 {TYPE_OUTPUT}")
    print(f"Quintile 细分已写入 {QUINTILE_OUTPUT}")


if __name__ == "__main__":
    main()
