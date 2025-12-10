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
| `data/processed/factor_returns_f5.parquet` | 2019-08-09 ~ 2025-12-05 的周度五因子序列（每期 5 分位分组，取 top-bottom 组合差）。 |
| `data/processed/factor_returns_mom.parquet` | 同期的 12-2 动量因子（log-return 窗口 48 周、跳过最近 4 周，同样以 5 分位 High-Low 构造）。 |
| `docs/regression_results.md` | 汇总五因子与扩展模型的系数、显著性与结论。 |

## 回归亮点
- 五因子模型：R²≈0.956，α≈-6.4e-04（不显著），`MKT` 仍接近 1；`SMB`、`HML` 在 5σ 以上显著，说明 5 分位 top-bottom 构建后放大了规模与价值暴露。
- 五因子+动量：R²≈0.956，与五因子几乎一致；`MOM` 系数 ~0.0016（t≈0.23）仍不显著。
- 动量因子范围扩大到约 -0.34 ~ 0.42，反映 5 分位极端组合的波动更大。

## 后续建议
1. **探索其它分组方式**：现已切换至 5 分位 top-bottom，可对照 2×3 或 2×3×3 网格，比较不同构造对解释力的影响。
2. **引入行业/状态切片**：在行业或规模子样本内重复计算因子与回归，验证稳健性。
3. **扩展组合对象**：当前示例仅为等权组合，可按策略或单只股票构建 `portfolio_return` 并复用 `run_regression` 接口。
4. **丰富诊断**：可保存残差序列、Q-Q 图或信息比率，便于写入正式研究报告。
