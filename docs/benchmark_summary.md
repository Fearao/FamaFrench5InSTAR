# Benchmark Implementation Summary

## 执行步骤
1. `python code/factor_pipeline.py`
   - 构建 `data/processed/weekly_panel.parquet`（行情 + 无风险利率）
   - 构建 `data/processed/weekly_fundamentals.parquet`（最新可得账面/盈利/投资指标）
   - 输出 `data/processed/factor_returns_f5.parquet` 与 `factor_returns_mom.parquet`
2. `python code/factor_regression.py`
   - 对等权组合运行五因子与五因子+动量回归
   - 结果写入 `docs/regression_*.csv` 与 `docs/regression_*.json`

## 主要数据资产
| 文件 | 描述 |
| --- | --- |
| `data/processed/weekly_panel.parquet` | 590 只股票 × 129,767 周度观测，含 `weekly_return`、`market_cap`、`rf_weekly`、`excess_return` 等。 |
| `data/processed/weekly_fundamentals.parquet` | 为每个交易周附上滞后 30 天的账面权益、盈利、投资指标，可直接用于分组。 |
| `data/processed/factor_returns_f5.parquet` | 2019-08-09 ~ 2025-12-05 的周度五因子序列。 |
| `data/processed/factor_returns_mom.parquet` | 同期的 12-2 动量因子（基于 log-return 窗口 48 周、跳过最近 4 周）。 |
| `docs/regression_results.md` | 汇总五因子与扩展模型的系数、显著性与结论。 |

## 回归亮点
- 五因子模型：R²=0.995，α≈-6.3e-05（不显著），`MKT` 与 `SMB` 主导解释力；`RMW/CMA` 在 10% 显著性附近。
- 五因子+动量：R²=0.996，α 依旧不显著；`MOM` 系数为 -0.012（|t|≈1.47）未通过 10% 阈值，但盈利因子 t 值升至 2.07，说明动量吸收了部分变异。
- 动量因子的极值范围约 -0.173 ~ 0.140，构建逻辑基于 rolling log-return，适合作为简单基准。

## 后续建议
1. **优化分组权重**：目前所有风格分组使用市值加权 + 三分位；可改为 2×3/2×3×3 分组以更贴近原版 Fama-French。
2. **引入行业/状态切片**：在行业或规模子样本内重复计算因子与回归，验证稳健性。
3. **扩展组合对象**：当前示例仅为等权组合，可按策略或单只股票构建 `portfolio_return` 并复用 `run_regression` 接口。
4. **丰富诊断**：可保存残差序列、Q-Q 图或信息比率，便于写入正式研究报告。
