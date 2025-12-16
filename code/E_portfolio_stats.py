"""计算 5×5 尺寸-风格组合的描述性统计并导出 CSV。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_OUTPUT = ROOT / "docs" / "tables" / "desc_stats_portfolios.csv"
PORTFOLIO_SPECS: List[Tuple[str, Path, str]] = [
    ("Size-B/M", PROCESSED_DIR / "portfolio_returns_size_bm.parquet", "bm_quintile"),
    ("Size-OP", PROCESSED_DIR / "portfolio_returns_size_op.parquet", "op_quintile"),
    ("Size-Inv", PROCESSED_DIR / "portfolio_returns_size_inv.parquet", "inv_quintile"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="输出 CSV 路径，默认为 docs/tables/desc_stats_portfolios.csv",
    )
    return parser.parse_args()


def _validate_columns(df: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        cols = ", ".join(sorted(missing))
        raise ValueError(f"{path.name} 缺少列: {cols}")


def load_portfolio(path: Path, factor_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件: {path}")
    df = pd.read_parquet(path)
    required = ["month", "size_quintile", factor_col, "n_stocks", "excess_return"]
    _validate_columns(df, required, path)
    df = df.rename(columns={factor_col: "factor_quintile"})
    return df[["month", "size_quintile", "factor_quintile", "n_stocks", "excess_return"]]


def compute_statistics(df: pd.DataFrame, portfolio_type: str) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    grouped = df.groupby(["size_quintile", "factor_quintile"], sort=True)
    for (size_q, factor_q), group in grouped:
        valid_mask = (group["n_stocks"].fillna(0) > 0) & group["excess_return"].notna()
        valid_returns = group.loc[valid_mask, "excess_return"]
        valid_stocks = group.loc[valid_mask, "n_stocks"].astype(float)
        n_months = int(valid_returns.shape[0])
        mean_return = valid_returns.mean() if n_months else np.nan
        std_return = valid_returns.std(ddof=1) if n_months > 1 else np.nan
        t_stat = np.nan
        p_value = np.nan
        sharpe_ratio = np.nan
        if n_months and std_return is not None and not np.isnan(std_return) and std_return > 0:
            se = std_return / np.sqrt(n_months)
            if se > 0:
                t_stat = mean_return / se
                dof = n_months - 1
                if dof > 0:
                    p_value = 2 * stats.t.sf(np.abs(t_stat), dof)
            sharpe_ratio = (mean_return / std_return) * np.sqrt(12)
        avg_n_stocks = valid_stocks.mean() if n_months else np.nan
        records.append(
            {
                "porfolio_type": portfolio_type,
                "size_quintile": int(size_q),
                "factor_quintile": int(factor_q),
                "mean_excess_return": mean_return,
                "std_excess_return": std_return,
                "t_stat": t_stat,
                "p_value": p_value,
                "sharpe_ratio": sharpe_ratio,
                "n_months": n_months,
                "avg_n_stocks": avg_n_stocks,
            }
        )
    return records


def summarize_counts(results: pd.DataFrame) -> None:
    total = len(results)
    effective = results[results["n_months"] > 0]
    print(f"总组合数: {total}")
    print("各类型有效组合数：")
    for label in [spec[0] for spec in PORTFOLIO_SPECS]:
        count = int(effective[effective["porfolio_type"] == label].shape[0])
        print(f"  {label}: {count}")


def main() -> None:
    args = parse_args()
    output_path = args.output
    if not output_path.is_absolute():
        output_path = (ROOT / output_path).resolve()
    records: List[Dict[str, float]] = []
    for portfolio_type, path, factor_col in PORTFOLIO_SPECS:
        df = load_portfolio(path, factor_col)
        records.extend(compute_statistics(df, portfolio_type))
    result_df = pd.DataFrame.from_records(records)
    cat = pd.CategoricalDtype(categories=[spec[0] for spec in PORTFOLIO_SPECS], ordered=True)
    result_df["porfolio_type"] = result_df["porfolio_type"].astype(cat)
    result_df = result_df.sort_values(["porfolio_type", "size_quintile", "factor_quintile"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    summarize_counts(result_df)


if __name__ == "__main__":
    main()
