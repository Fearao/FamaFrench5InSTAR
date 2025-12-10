# Fama-French 五因子 + 动量基准方案

最后更新：2025-12-10（基于 `data/processed` 目录最新 parquet 与 schema）

## 目标
- 以周频数据复现 Fama-French 五因子（MKT-RF、SMB、HML、RMW、CMA）作为 benchmark。
- 在同一回归框架中加入标准 12-2 动量因子，评估动量对解释力的增量贡献。
- 输出可复现的回归脚本、因子序列与诊断图表，为后续策略回测提供基准比较。

## 数据输入
| 文件 | 作用 |
| --- | --- |
| `market_prices.parquet` | 每周 OHLC、成交量与市值（595 股，2019-07-26 ~ 2025-12-08）。
| `size_factor.parquet` | 含市值、收益率 `Wretwd` 的周度记录（590 股，`Trdwnt` 为 `YYYY-WW`）。
| `bm_factor.parquet` | 账面价值 `A003000000`（585 股，2019-01-01 ~ 2025-09-30）。
| `investment_factor.parquet` | 投资强度指标 F080601A~C（590 股，2018-12-31 ~ 2025-09-30）。
| `profitability_factor.parquet` | 盈利指标 `B130101`（110 股，2017-12-31 ~ 2025-09-30）。
| `risk_free_rate.parquet` | 3 个月期无风险利率跳点序列（2022-09-15 ~ 2025-05-20）。
| `schema.json` | 字段、dtype、缺失值基线，便于数据校验。

## 方法步骤
1. **频率与样本**：
   - 采用周频；交易周以 `market_prices` 中的 `时间` 为基准，确保与 `size_factor.Trdwnt`（转换为周一日期）一致。
   - 股票池取交集（≥ 2019-07-26 有可用行情且拥有所需 fundamental 指标）。
2. **数据清洗与对齐**：
   - 处理缺失：删除行情中 8 条完全缺失行；对 `investment_factor`/`size_factor` 缺值按股票+报告期插值或剔除。
   - `risk_free_rate` 按周前向填充，生成 `RF_t`。
   - 将 `Accper` 与 `Annodt` 推迟一个披露滞后（例如财报发布后一个月），只在滞后后参与分组以避免前视。
3. **因子构建**：
   - **市场因子**：`MKT-RF` = 市值加权的全部股票周收益 − `RF_t`。
   - **SMB**：按期末市值（大/小）× 风格（如 HML、RMW、CMA 中的高/低）形成 2×3 或 2×3×3 组合，再按照 Fama-French 指南求平均。
   - **HML**：账面市值比 = 账面权益 / 市值，分高/中/低；计算高减低组合收益。
   - **RMW**：盈利率 = `B130101` / 总资产（或市值），定义 Robust/Neutral/Weak。
   - **CMA**：投资率 = 资产增长（例如 F080601A 系列），定义 Conservative/Neutral/Aggressive。
   - 所有组合每年/每季重组，期间使用固定权重并按周收益滚动。
4. **动量因子**：
   - 计算每只股票的过去 12 个月累计收益（skip 最近 1 个月），排除停牌/缺失样本。
   - 形成 High−Low 组合（可 3 组或 10 分位），得到 `MOM_t`。
5. **回归分析**：
   - **基准**：`R_it - RF_t = α_i + β_i MKT_t + s_i SMB_t + h_i HML_t + r_i RMW_t + c_i CMA_t + ε_it`。
   - **扩展**：在上式基础上加入 `m_i MOM_t`。
   - 输出：系数估计、t 值、R²/调整 R²、α 显著性、Newey-West 纠偏、残差诊断（自相关、异方差）。
   - 可选：做 Fama-MacBeth 截面回归以验证时间序结果。
6. **比较与解读**：
   - 对比两套模型 α 的变化；若动量显著降低 α，则认为动量解释了残差部分。
   - 记录各因子暴露的符号/大小是否符合预期（例如成长股应对 HML 为负）。
   - 分子样本（行业、市值、时期）重复上述回归，检验稳健性。

## TODO
- [ ] 搭建周频数据管道：合并 `market_prices` 与 `size_factor`，补齐 `risk_free_rate` 并生成超额收益序列。
- [ ] 处理财报数据滞后与缺失：对 `bm_factor`、`investment_factor`、`profitability_factor` 建立可滚动的最新可得指标。
- [ ] 构建 MKT-RF、SMB、HML、RMW、CMA 因子收益，并验证与参考值（如中金/中证因子）方向一致。
- [ ] 生成 MOM 动量因子（12-2 规则），检查尾部样本是否受停牌或极端收益影响。
- [ ] 运行五因子 benchmark 回归，输出系数表、残差诊断和 α 显著性。
- [ ] 在 benchmark 基础上加入 MOM 因子，再次运行回归并比较指标（α、R²、信息比率）。
- [ ] 形成总结报告：包含方法、关键图表（因子收益曲线、回归系数）及结论，提交到 `docs/` 下的新文件。
