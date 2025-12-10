"""Simple helpers to run time-series factor regressions."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
DOCS_DIR = Path(__file__).parent.parent / "docs"
FIVE_FACTOR_PATH = DATA_DIR / "factor_returns_f5.parquet"
MOM_FACTOR_PATH = DATA_DIR / "factor_returns_mom.parquet"


def _load_portfolio_returns() -> pd.DataFrame:
    weekly = pd.read_parquet(DATA_DIR / "weekly_panel.parquet", columns=["trade_date", "weekly_return"])
    weekly = weekly.dropna(subset=["weekly_return"])  # remove missing returns
    portfolio = (
        weekly.groupby("trade_date")
        ["weekly_return"]
        .mean()
        .reset_index()
        .rename(columns={"weekly_return": "portfolio_return"})
    )
    return portfolio


def _load_factor_data(include_momentum: bool = False) -> pd.DataFrame:
    factors = pd.read_parquet(FIVE_FACTOR_PATH)
    if include_momentum:
        momentum = pd.read_parquet(MOM_FACTOR_PATH)
        factors = factors.merge(momentum, on="trade_date", how="inner")
    return factors


def _prepare_regression_frame(factor_cols: list[str]) -> pd.DataFrame:
    portfolio = _load_portfolio_returns()
    factors = _load_factor_data(include_momentum="MOM" in factor_cols)
    merged = portfolio.merge(factors, on="trade_date", how="inner").sort_values("trade_date")
    merged["excess_portfolio"] = merged["portfolio_return"] - merged["RF"]
    merged = merged.dropna(subset=factor_cols + ["RF"])
    return merged


def run_regression(label: str, factor_cols: list[str]) -> dict:
    frame = _prepare_regression_frame(factor_cols)
    X = sm.add_constant(frame[factor_cols])
    model = sm.OLS(frame["excess_portfolio"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})

    summary = pd.DataFrame(
        {
            "term": model.params.index,
            "coef": model.params.values,
            "tvalue": model.tvalues,
            "pvalue": model.pvalues,
        }
    )
    meta = {
        "label": label,
        "n_obs": int(model.nobs),
        "r_squared": float(model.rsquared),
        "r_squared_adj": float(model.rsquared_adj),
    }

    DOCS_DIR.mkdir(exist_ok=True)
    summary_path = DOCS_DIR / f"{label}_coeffs.csv"
    meta_path = DOCS_DIR / f"{label}_meta.json"
    summary.to_csv(summary_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"Saved regression outputs to {summary_path.name} and {meta_path.name} "
        f"(R^2={meta['r_squared']:.3f}, n={meta['n_obs']})"
    )
    return {"model": model, "summary": summary, "meta": meta}


if __name__ == "__main__":
    run_regression("regression_f5", ["MKT_RF", "SMB", "HML", "RMW", "CMA"])
    run_regression(
        "regression_f5_mom",
        ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"],
    )
