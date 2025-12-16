'''Fama-French 五因子诊断：相关性、VIF、互回归 R^2 与统计摘要。'''

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Tuple

warnings.filterwarnings('ignore', message='.*joblib will operate in serial mode.*')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'data' / 'processed' / 'factor_returns_ff5.parquet'
DEFAULT_CORR_CSV = ROOT / 'docs' / 'tables' / 'factor_corr.csv'
DEFAULT_VIF_CSV = ROOT / 'docs' / 'tables' / 'factor_vif.csv'
DEFAULT_STATS_CSV = ROOT / 'docs' / 'tables' / 'factor_stats.csv'
DEFAULT_HEATMAP = ROOT / 'docs' / 'figs' / 'factor_corr_heatmap.png'
FACTOR_COLUMNS = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input-file',
        type=Path,
        default=DEFAULT_INPUT,
        help='输入 Parquet（默认 data/processed/factor_returns_ff5.parquet）',
    )
    parser.add_argument(
        '--corr-table',
        type=Path,
        default=DEFAULT_CORR_CSV,
        help='相关矩阵 CSV 输出路径',
    )
    parser.add_argument(
        '--vif-table',
        type=Path,
        default=DEFAULT_VIF_CSV,
        help='VIF+R^2 CSV 输出路径',
    )
    parser.add_argument(
        '--stats-table',
        type=Path,
        default=DEFAULT_STATS_CSV,
        help='因子统计 CSV 输出路径',
    )
    parser.add_argument(
        '--corr-fig',
        type=Path,
        default=DEFAULT_HEATMAP,
        help='相关系数热力图输出路径',
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_factors(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    missing = [col for col in FACTOR_COLUMNS if col not in df.columns]
    if missing:
        missing_cols = ', '.join(missing)
        raise ValueError(f'缺失因子列: {missing_cols}')
    factors = df[FACTOR_COLUMNS].dropna(how='any').reset_index(drop=True)
    if factors.empty:
        raise ValueError('有效因子截面为空，无法诊断')
    return factors


def compute_correlations(factors: pd.DataFrame) -> pd.DataFrame:
    return factors.corr(method='pearson')


def compute_vif_and_r2(factors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in FACTOR_COLUMNS:
        y = factors[target]
        X = factors.drop(columns=target)
        model = sm.OLS(y, sm.add_constant(X, has_constant='add'), missing='drop').fit()
        r2 = float(model.rsquared)
        if np.isclose(r2, 1.0):
            vif = float('inf')
        else:
            vif = float(1.0 / (1.0 - r2))
        rows.append({'factor': target, 'r_squared': r2, 'vif': vif})
    return pd.DataFrame(rows)


def compute_stats(factors: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame(index=FACTOR_COLUMNS)
    stats['mean'] = factors.mean()
    stats['std'] = factors.std(ddof=1)
    stats['skew'] = factors.skew()
    stats['kurtosis'] = factors.kurtosis()
    return stats.reset_index(names='factor')


def correlation_extrema(corr: pd.DataFrame) -> Tuple[Tuple[str, str, float], Tuple[str, str, float]]:
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    stacked = corr.where(mask).stack()
    if stacked.empty:
        raise ValueError('相关矩阵缺少可用的非对角元素')
    max_idx = stacked.idxmax()
    min_idx = stacked.idxmin()
    return (max_idx[0], max_idx[1], float(stacked.loc[max_idx])), (
        min_idx[0],
        min_idx[1],
        float(stacked.loc[min_idx]),
    )


def plot_heatmap(corr: pd.DataFrame, output: Path) -> None:
    ensure_parent(output)
    plt.close('all')
    sns.set_theme(style='white', context='talk')
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'shrink': 0.7},
    )
    ax.set_title('FF5 factor correlation')
    fig.savefig(output, dpi=200)
    plt.close(fig)


def print_summary(corr: pd.DataFrame, vif: pd.DataFrame, sample_size: int) -> None:
    high, low = correlation_extrema(corr)
    top_idx = vif['vif'].replace(np.inf, np.nan).idxmax()
    if pd.isna(top_idx):
        top_idx = vif['vif'].idxmax()
    top = vif.loc[top_idx]
    top_factor = top['factor']
    top_r2 = top['r_squared']
    vif_val = top['vif']
    vif_text = 'inf' if not np.isfinite(vif_val) else f'{vif_val:.2f}'
    print(f'样本行数: {sample_size}')
    print(f'最高相关: {high[0]} vs {high[1]} = {high[2]:.4f}')
    print(f'最低相关: {low[0]} vs {low[1]} = {low[2]:.4f}')
    print(f'最高 VIF: {top_factor} (R^2={top_r2:.4f}, VIF={vif_text})')


def main() -> None:
    args = parse_args()
    factors = load_factors(args.input_file)

    corr = compute_correlations(factors)
    vif = compute_vif_and_r2(factors)
    stats = compute_stats(factors)

    ensure_parent(args.corr_table)
    corr.to_csv(args.corr_table, float_format='%.6f')

    ensure_parent(args.vif_table)
    vif.to_csv(args.vif_table, index=False, float_format='%.6f')

    ensure_parent(args.stats_table)
    stats.to_csv(args.stats_table, index=False, float_format='%.6f')

    plot_heatmap(corr, args.corr_fig)
    print_summary(corr, vif, len(factors))


if __name__ == '__main__':
    main()
