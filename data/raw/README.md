# Raw Data (原始数据)

## 概述 (Overview)

本目录包含用于构建 **Fama-French 5因子模型** 的原始数据。数据主要来源于中国A股市场，包括股票周度行情数据和财务基本面数据。

This directory contains raw data for building the **Fama-French 5-Factor Model**. The data primarily consists of weekly stock price data and fundamental financial data from the Chinese A-share market.

---

## 数据文件 (Data Files)

### 1. 行情序列后复权.xlsx (Post-Adjustment Stock Price Series)

**描述 (Description):**
- 包含中国A股（科创板）股票的周度行情数据，已进行后复权处理
- Contains weekly price data for Chinese A-share (STAR Board) stocks with post-adjustment (split/dividend adjusted)

**文件大小 (File Size):** ~34 MB

**数据结构 (Data Structure):**
- **Sheet数量:** 8个 (8 sheets)
- **总行数:** ~90,000行 (约90K rows)
- **时间范围:** 2019年7月 ~ 2023年10月 (July 2019 - October 2023)
- **覆盖股票:** ~30个不同的股票代码

**Sheet说明 (Sheet Details):**

| Sheet | 行数 (Rows) | 描述 (Description) |
|-------|---------|---------|
| Sheet1 | 24,217 | 早期上市科创板股票 (Early-listed STAR board stocks) |
| Sheet2 | 22,721 | 中期上市科创板股票 (Mid-period STAR board stocks) |
| Sheet3 | 20,618 | 科创板股票 (STAR board stocks) |
| Sheet4 | 18,599 | 科创板股票 (STAR board stocks) |
| Sheet5 | 13,688 | 科创板股票 (STAR board stocks) |
| Sheet6 | 15,644 | 科创板股票 (STAR board stocks) |
| Sheet7 | 13,503 | 科创板股票 (STAR board stocks) |
| Sheet8 | 1,694 | 近期上市科创板股票 (Recently-listed STAR board stocks) |

**列定义 (Column Definitions):**

| 列名 (Column) | 数据类型 (Type) | 说明 (Description) |
|---------|---------|---------|
| 代码 (Code) | string | 股票代码，格式为 XXXXXX.SH |
| 简称 (Short Name) | string | 股票简称 (Stock name abbreviation) |
| 时间 (Date) | string | 交易周的截止日期，格式YYYY-MM-DD |
| 周开盘价(元) (Weekly Open) | float | 周开盘价，单位：元/股 (Yuan per share) |
| 周最高价(元) (Weekly High) | float | 周最高价，单位：元/股 |
| 周收盘价(元) (Weekly Close) | float | 周收盘价，单位：元/股 |
| 周最低价(元) (Weekly Low) | float | 周最低价，单位：元/股 |
| 周成交量(万股) (Weekly Volume) | float | 周成交量，单位：万股 (10,000 shares) |
| 周总市值(万元) (Weekly Market Cap) | float | 周末总市值，单位：万元 (10,000 Yuan) |

**数据质量 (Data Quality):**
- 缺失值 (Missing values): 每个Sheet都有2行缺失（可能是空行或错误行）
- 数据类型 (Data types): 价格和市值为float64，代码和日期为object (string)
- 数值范围 (Value ranges):
  - 开盘价：约 10~170 元
  - 交易量：约 300~20,000 万股
  - 市值：约 50万~260万 万元

**使用场景 (Use Cases):**
```python
import pandas as pd

# 读取某个Sheet
df = pd.read_excel('行情序列后复权.xlsx', sheet_name='Sheet1')

# 筛选某只股票的数据
stock_code = '688001.SH'
stock_data = df[df['代码'] == stock_code]

# 计算周回报率
stock_data['weekly_return'] = stock_data['周收盘价(元)'].pct_change()
```

---

### 2. 基础数据.xlsx (Basic Fundamental Data)

**描述 (Description):**
- 包含股票的财务基本面数据，用于构建Fama-French因子
- Contains fundamental financial data for constructing Fama-French factors

**文件大小 (File Size):** ~50 MB

**Sheet数量和说明 (Sheet Details):**

#### Sheet 1: 市场因子 (Market Factor)
- **行数:** 12行
- **列数:** 2列
- **内容:** 无风险利率数据（定期存款利率）
- **数据样本:**
  - ChangeDate: 变动日期 (Change date)
  - LumpFixed3Month: 定期存款：整存整取：三个月(%) (3-month fixed deposit rate)

**注意:** 前两行为表头说明，实际数据从第3行开始

#### Sheet 2: 规模因子 (Size Factor)
- **行数:** 129,769行
- **列数:** 7列
- **描述:** 周度股票行情和回报数据
- **覆盖范围:** 多个股票，多个交易周

| 列名 (Column) | 说明 (Description) |
|---------|---------|
| Stkcd | 证券代码 (Stock code) |
| Trdwnt | 交易周份 (Trading week, format: YYYY-WW) |
| Wopnprc | 周开盘价 (Weekly open price, 元/股) |
| Wclsprc | 周收盘价 (Weekly closing price, 元/股) |
| Wnshrtrd | 周个股交易股数 (Weekly shares traded) |
| Wsmvttl | 周个股总市值 (Weekly market value, 千元) |
| Wretwd | 考虑现金红利再投资的周个股回报率 (Weekly return with dividend reinvestment) |

**数据质量:**
- Wretwd列缺失值: 590行
- 前2行为表头说明，实际数据从第3行开始

#### Sheet 3: 账面市值比因子 (Book-to-Market Factor)
- **行数:** 20,929行
- **列数:** 5列
- **描述:** 股票的会计权益数据

| 列名 (Column) | 说明 (Description) |
|---------|---------|
| Stkcd | 证券代码 |
| ShortName | 证券简称 |
| Accper | 统计截止日期 (Accounting period) |
| Typrep | 报表类型 (Report type) |
| A003000000 | 所有者权益合计 (Total shareholders' equity, 元) |

**数据质量:** 所有列完整，无缺失值

#### Sheet 4: 盈利因子 (Profitability Factor)
- **行数:** 468行
- **列数:** 6列
- **描述:** 股票的盈利能力数据

| 列名 (Column) | 说明 (Description) |
|---------|---------|
| Stkcd | 证券代码 |
| Annodt | 公告日期 (Announcement date) |
| Accper | 统计截止日期 (Accounting period) |
| Sign | 行为标识 (Behavior flag) |
| Adtsgn | 审计标识 (Audit flag) |
| B130101 | 营业利润 (Operating profit, 元) |

**数据质量:** 数据量相对较小，可能为样本数据

#### Sheet 5: 投资因子 (Investment Factor)
- **行数:** 20,738行
- **列数:** 8列
- **描述:** 股票的资产增长率数据

| 列名 (Column) | 说明 (Description) |
|---------|---------|
| Stkcd | 股票代码 |
| ShortName | 股票简称 |
| Accper | 统计截止日期 |
| Typrep | 报表类型编码 |
| Source | 公告来源 |
| F080601A | 总资产增长率A |
| F080602A | 总资产增长率B |
| F080603A | 总资产增长率C |

**数据质量:**
- F080601A缺失值: 4,485行
- F080602A缺失值: 3,863行
- F080603A缺失值: 3,863行

---

## 数据特征总结 (Data Summary)

| 特征 (Feature) | 值 (Value) |
|---------|---------|
| 总文件数 | 2个 |
| 总大小 | ~84 MB |
| 总Sheet数 | 13个 |
| 时间范围 | 2019年7月 ~ 2023年10月 |
| 股票覆盖 | ~30+只科创板股票 |
| 主要数据类型 | 周度行情数据、财务数据 |

---

## 数据质量注意事项 (Data Quality Notes)

### 缺失值 (Missing Values)
- 行情数据: 每个sheet顶部有2行不完整数据
- 财务数据: 某些指标存在缺失（如盈利因子和投资因子）
- **处理建议:** 在使用前进行数据清洗，删除缺失行或进行插补

### 数据类型 (Data Types)
- 日期列以字符串形式存储，需要转换为datetime
- 某些数字列可能被读为字符串，需要显式转换

### 表头说明 (Header Rows)
- 基础数据中的多个sheet在前两行包含中文表头说明
- 实际数据通常从第3行开始
- **处理建议:** 读取时使用 `skiprows=2` 参数

---

## 使用示例 (Usage Examples)

### 基础用法 (Basic Usage)

```python
import pandas as pd

# 读取行情数据
price_data = pd.read_excel('行情序列后复权.xlsx', sheet_name='Sheet1')
print(price_data.head())
print(price_data.info())

# 读取财务数据（跳过表头说明行）
size_factor = pd.read_excel('基础数据.xlsx', sheet_name='规模因子', skiprows=2)
print(size_factor.head())
```

### 数据清洗 (Data Cleaning)

```python
# 删除缺失值
price_data_clean = price_data.dropna()

# 转换日期格式
price_data['时间'] = pd.to_datetime(price_data['时间'])

# 提取某只股票的数据
stock_data = price_data[price_data['代码'] == '688001.SH']

# 计算周回报率
stock_data['return'] = stock_data['周收盘价(元)'].pct_change()
```

### 因子构建 (Factor Construction)

```python
# 构建规模因子（使用市值数据）
size_factor_data = price_data[['时间', '代码', '周总市值(万元)']].copy()
size_factor_data['size'] = size_factor_data['周总市值(万元)'].apply(lambda x: 'small' if x < 100000 else 'large')

# 构建账面市值比因子
bm_data = pd.read_excel('基础数据.xlsx', sheet_name='账面市值比因子', skiprows=2)
bm_data['BM'] = bm_data['A003000000'] / bm_data['周个股总市值']
```

---

## 数据来源与更新 (Data Source & Updates)

- **来源 (Source):** 中国A股市场数据（科创板 / STAR Board）
- **更新频率 (Update Frequency):** 周度更新（Weekly）
- **最后更新 (Last Updated):** 2023年10月

---

## 联系与反馈 (Contact & Feedback)

如有数据问题或建议，请提交issue或联系数据管理团队。

For data issues or feedback, please submit an issue or contact the data management team.
