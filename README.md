# MOM_EQ 因子研究

本仓库仅保留了复现最新动量研究所需的最小代码和数据流程，包括：

1. `code/B_data_prepare.py`：把 `data/raw/` 中的原始 Excel 转换为 `data/processed/` 下的基础 parquet（行情、账面、盈利、投资、无风险利率等）。
2. `code/factor_pipeline.py`：基于 `data/processed/` 生成周频交易面板、滞后财务指标，以及五因子基准序列（`factor_returns_f5.parquet`）。
3. `code/momentum_comparison.py`：在同一数据上构建多个动量窗口（4/8/16/26/52/78 周，均为 quintile High-Low），并输出各自及等权平均 `MOM_EQ` 的回归结果。

## 使用流程
1. 准备原始数据到 `data/raw/`，执行 `python code/B_data_prepare.py` 生成 `data/processed/*.parquet`。
2. 运行 `python code/factor_pipeline.py`，得到 `weekly_panel.parquet`、`weekly_fundamentals.parquet`、`factor_returns_f5.parquet` 以及最新的 `factor_returns_mom.parquet`（5 分位构造）。
3. 执行 `python code/momentum_comparison.py`：
   - 在 `data/processed/` 下生成 `factor_returns_mom_w{window}_s{skip}.parquet`。
   - 对每个动量因子（含 `MOM_EQ` 等权平均）做 HAC(4) 回归，结果写入 `docs/mom_w*_s*.csv/.json` 及 `docs/momentum_comparison.md`。

## 输出
`docs/momentum_comparison.md` 中列出了各动量窗口与 `MOM_EQ` 的 R²、α、MOM 系数与 t 值，可直接作为最新研究结论；对应的详细系数表和 meta JSON 亦位于 `docs/` 目录下。
