"""Run 6 regression models: FF5+MOM_4M/8M/12M/24M/EQ and FF5 baseline."""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# 配置
FF5_COLS = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
MOM_FACTORS = ["MOM_4M", "MOM_8M", "MOM_12M", "MOM_24M", "MOM_EQ"]
PORTFOLIO_CONFIG = [
    ("portfolio_returns_size_bm.parquet", "size_bm", "bm_quintile"),
    ("portfolio_returns_size_op.parquet", "size_op", "op_quintile"),
    ("portfolio_returns_size_inv.parquet", "size_inv", "inv_quintile"),
]

def load_data(input_dir="data/processed"):
    """加载75个投资组合和多window因子"""
    # 加载因子
    factors = pd.read_parquet(f"{input_dir}/factor_returns_multiwindow.parquet")
    factors["month"] = pd.to_datetime(factors["month"], format="%Y-%m")

    # 加载投资组合
    portfolios = []
    for filename, portfolio_type, factor_col in PORTFOLIO_CONFIG:
        df = pd.read_parquet(f"{input_dir}/{filename}")
        df["portfolio_type"] = portfolio_type
        df["factor_quintile"] = df[factor_col].astype(int)
        df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
        portfolios.append(df[["month", "portfolio_type", "size_quintile", "factor_quintile", "excess_return"]])

    portfolios_df = pd.concat(portfolios, ignore_index=True)

    # 合并
    merged = portfolios_df.merge(factors, on="month", how="left")
    return merged

def run_regressions(merged_df, model_name, factor_cols):
    """对所有投资组合运行一个模型的回归"""
    results = []

    for (ptype, size_q, factor_q), group in merged_df.groupby(
        ["portfolio_type", "size_quintile", "factor_quintile"], sort=True
    ):
        # 数据清理
        subset = group[["excess_return"] + factor_cols].dropna()
        n_obs = len(subset)

        # 初始化记录
        record = {
            "model": model_name,
            "portfolio_type": ptype,
            "size_quintile": int(size_q),
            "factor_quintile": int(factor_q),
            "n_obs": n_obs,
            "alpha": np.nan,
            "t_alpha": np.nan,
            "p_alpha": np.nan,
            "r_squared": np.nan,
            "adj_r_squared": np.nan,
        }

        # 为所有因子添加beta和t值列
        for col in factor_cols:
            record[f"beta_{col}"] = np.nan
            record[f"t_{col}"] = np.nan

        # 检查样本量
        if n_obs < 8:
            results.append(record)
            continue

        # OLS回归
        try:
            y = subset["excess_return"].astype(float)
            X = sm.add_constant(subset[factor_cols].astype(float))
            hac_lags = max(0, min(4, n_obs - 1))
            fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

            # 提取结果
            record.update({
                "alpha": float(fit.params.get("const", np.nan)),
                "t_alpha": float(fit.tvalues.get("const", np.nan)),
                "p_alpha": float(fit.pvalues.get("const", np.nan)),
                "r_squared": float(fit.rsquared),
                "adj_r_squared": float(fit.rsquared_adj),
            })

            for col in factor_cols:
                record[f"beta_{col}"] = float(fit.params.get(col, np.nan))
                record[f"t_{col}"] = float(fit.tvalues.get(col, np.nan))

        except Exception as e:
            print(f"回归失败 {model_name}: {ptype} s{size_q}f{factor_q}: {e}")

        results.append(record)

    return pd.DataFrame(results)

def main():
    """主函数"""
    print("加载数据...")
    merged = load_data()

    # 定义6个模型
    models = [
        ("FF5+MOM_4M", FF5_COLS + ["MOM_4M"]),
        ("FF5+MOM_8M", FF5_COLS + ["MOM_8M"]),
        ("FF5+MOM_12M", FF5_COLS + ["MOM_12M"]),
        ("FF5+MOM_24M", FF5_COLS + ["MOM_24M"]),
        ("FF5+MOM_EQ", FF5_COLS + ["MOM_EQ"]),
        ("FF5_baseline", FF5_COLS),
    ]

    # 运行回归
    all_results = {}
    output_dir = Path("docs/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, factor_cols in models:
        print(f"运行 {model_name}...")
        results_df = run_regressions(merged, model_name, factor_cols)
        all_results[model_name] = results_df

        # 保存详细结果
        output_file = output_dir / f"regression_results_{model_name.lower().replace('+', '_').replace('_baseline', '_ff5_baseline')}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"  已保存到 {output_file}")

    # 生成对标汇总
    summary_rows = []
    for model_name, results_df in all_results.items():
        valid = results_df[results_df["alpha"].notna()]

        if len(valid) > 0:
            alpha_sig = (valid["p_alpha"] < 0.05).sum() / len(valid)
            mom_col = None
            for col in results_df.columns:
                if col.startswith("beta_MOM"):
                    mom_col = col
                    break

            mom_sig = 0
            if mom_col:
                mom_beta_col = mom_col
                mom_t_col = mom_col.replace("beta_", "t_")
                valid_mom = results_df[results_df[mom_beta_col].notna()]
                if len(valid_mom) > 0:
                    mom_sig = (valid_mom[mom_t_col].abs() > 1.96).sum() / len(valid_mom)

            hml_t_col = "t_HML"
            valid_hml = results_df[results_df[hml_t_col].notna()]
            hml_sig = 0
            if len(valid_hml) > 0:
                hml_sig = (valid_hml[hml_t_col].abs() > 1.96).sum() / len(valid_hml)

            summary_rows.append({
                "Model": model_name,
                "Avg_Alpha": valid["alpha"].mean(),
                "Alpha_Sig_Rate": alpha_sig,
                "Mean_R2": valid["r_squared"].mean(),
                "Mean_AdjR2": valid["adj_r_squared"].mean(),
                "Valid_Portfolios": len(valid),
                "MOM_Sig_Rate": mom_sig if mom_col else np.nan,
                "HML_Sig_Rate": hml_sig,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_dir / "multiwindow_regression_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\n对标汇总已保存到 {summary_file}")
    print("\n" + summary_df.to_string())

if __name__ == "__main__":
    main()
