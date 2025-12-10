"""Lightweight utilities for building factor inputs.

Currently focuses on Step 1 of the plan: assembling a weekly panel that
combines market prices, size-factor data, and a forward-filled risk-free rate
series so that weekly excess returns are ready for downstream factor work.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
WEEKLY_PANEL_PATH = DATA_PROCESSED_DIR / "weekly_panel.parquet"
WEEKLY_FUNDAMENTALS_PATH = DATA_PROCESSED_DIR / "weekly_fundamentals.parquet"
FUNDAMENTAL_LAG_DAYS = 30


def _load_market_prices() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PROCESSED_DIR / "market_prices.parquet")
    df = df.rename(
        columns=
        {
            "代码": "stock_code",
            "简称": "short_name",
            "时间": "trade_date",
            "周开盘价(元)": "open",
            "周最高价(元)": "high",
            "周收盘价(元)": "close",
            "周最低价(元)": "low",
            "周成交量(万股)": "volume",
            "周总市值(万元)": "market_cap",
        }
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df[df["trade_date"].notna()].copy()
    df["stock_id"] = df["stock_code"].str[:6]
    df = df[
        [
            "stock_id",
            "trade_date",
            "stock_code",
            "short_name",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "market_cap",
        ]
    ]
    return df


def _load_size_factor() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PROCESSED_DIR / "size_factor.parquet")
    df["stock_id"] = df["Stkcd"].astype(str).str.zfill(6)
    # `Trdwnt` looks like YYYY-WW (week number). Take the Monday of that week and shift to Friday
    trade_week = pd.to_datetime(
        df["Trdwnt"].astype(str) + "-1",
        format="%Y-%W-%w",
        errors="coerce",
    )
    df["trade_date"] = trade_week + pd.to_timedelta(4, unit="D")
    df = df[df["trade_date"].notna()].copy()
    df = df.rename(
        columns=
        {
            "Wopnprc": "sf_open",
            "Wclsprc": "sf_close",
            "Wnshrtrd": "shares_traded",
            "Wsmvttl": "sf_market_cap",
            "Wretwd": "weekly_return",
        }
    )
    df = df[
        [
            "stock_id",
            "trade_date",
            "sf_open",
            "sf_close",
            "shares_traded",
            "sf_market_cap",
            "weekly_return",
        ]
    ]
    return df


def _build_weekly_rf_series(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    rf = pd.read_parquet(DATA_PROCESSED_DIR / "risk_free_rate.parquet")
    rf = rf.rename(columns={"ChangeDate": "change_date", "LumpFixed3Month": "rf_3m"})
    rf["change_date"] = pd.to_datetime(rf["change_date"], errors="coerce")
    rf = rf.sort_values("change_date").dropna(subset=["change_date"])  # keep clean timeline
    # Simple assumption: quoted rate is annualized percent. Approximate weekly equivalent by dividing by 52.
    rf["rf_weekly"] = rf["rf_3m"].astype(float) / 100.0 / 52.0

    calendar = pd.DataFrame({"trade_date": pd.date_range(start_date, end_date, freq="W-FRI")})
    calendar = pd.merge_asof(
        calendar,
        rf[["change_date", "rf_weekly"]],
        left_on="trade_date",
        right_on="change_date",
        direction="backward",
    )
    calendar["rf_weekly"] = calendar["rf_weekly"].bfill()
    calendar = calendar.drop(columns=["change_date"])
    return calendar


def build_weekly_panel() -> pd.DataFrame:
    """Public helper that wires steps together and writes the parquet."""
    market = _load_market_prices()
    size = _load_size_factor()

    weekly = size.merge(
        market,
        on=["stock_id", "trade_date"],
        how="left",
        validate="m:1",
    )

    start_date = weekly["trade_date"].min()
    end_date = weekly["trade_date"].max()
    rf = _build_weekly_rf_series(start_date, end_date)
    weekly = weekly.merge(rf, on="trade_date", how="left")
    weekly["excess_return"] = weekly["weekly_return"] - weekly["rf_weekly"]

    WEEKLY_PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    weekly.sort_values(["trade_date", "stock_id"]).to_parquet(WEEKLY_PANEL_PATH, index=False)
    print(
        f"Saved {WEEKLY_PANEL_PATH} with {len(weekly):,} rows across "
        f"{weekly['stock_id'].nunique()} stocks from {start_date.date()} to {end_date.date()}"
    )
    return weekly


def _prepare_bm_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PROCESSED_DIR / "bm_factor.parquet")
    df["stock_id"] = df["Stkcd"].astype(str).str.zfill(6)
    df["accper"] = pd.to_datetime(df["Accper"], errors="coerce")
    df = df[df["accper"].notna()].copy()
    df["available_date"] = df["accper"] + pd.to_timedelta(FUNDAMENTAL_LAG_DAYS, unit="D")
    df["book_equity"] = pd.to_numeric(df["A003000000"], errors="coerce")
    return df[["stock_id", "available_date", "book_equity"]]


def _prepare_profitability_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PROCESSED_DIR / "profitability_factor.parquet")
    df["stock_id"] = df["Stkcd"].astype(str).str.zfill(6)
    df["accper"] = pd.to_datetime(df["Accper"], errors="coerce")
    df = df[df["accper"].notna()].copy()
    df["available_date"] = df["accper"] + pd.to_timedelta(FUNDAMENTAL_LAG_DAYS, unit="D")
    df["profit_metric"] = pd.to_numeric(df["B130101"], errors="coerce")
    return df[["stock_id", "available_date", "profit_metric"]]


def _prepare_investment_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PROCESSED_DIR / "investment_factor.parquet")
    df["stock_id"] = df["Stkcd"].astype(str).str.zfill(6)
    df["accper"] = pd.to_datetime(df["Accper"], errors="coerce")
    df = df[df["accper"].notna()].copy()
    df["available_date"] = df["accper"] + pd.to_timedelta(FUNDAMENTAL_LAG_DAYS, unit="D")
    df["invest_F080601A"] = pd.to_numeric(df["F080601A"], errors="coerce")
    df["invest_F080602A"] = pd.to_numeric(df["F080602A"], errors="coerce")
    df["invest_F080603A"] = pd.to_numeric(df["F080603A"], errors="coerce")
    return df[
        [
            "stock_id",
            "available_date",
            "invest_F080601A",
            "invest_F080602A",
            "invest_F080603A",
        ]
    ]


def _merge_lagged(base: pd.DataFrame, lagged: pd.DataFrame, suffix: str) -> pd.DataFrame:
    value_cols = [c for c in lagged.columns if c not in {"stock_id", "available_date"}]
    lagged_groups = {
        sid: grp.drop(columns="stock_id").sort_values("available_date")
        for sid, grp in lagged.groupby("stock_id", sort=False)
    }

    outputs = []
    for sid, chunk in base.groupby("stock_id", group_keys=False):
        chunk = chunk.sort_values("trade_date")
        right = lagged_groups.get(sid)
        if right is None or right.empty:
            filler = {col: pd.NA for col in value_cols}
            filler[f"{suffix}_available_date"] = pd.NaT
            outputs.append(chunk.assign(**filler))
            continue

        merged = pd.merge_asof(
            chunk,
            right,
            left_on="trade_date",
            right_on="available_date",
            direction="backward",
        )
        merged = merged.rename(columns={"available_date": f"{suffix}_available_date"})
        outputs.append(merged)

    return pd.concat(outputs, ignore_index=True)


def build_weekly_fundamentals() -> pd.DataFrame:
    if not WEEKLY_PANEL_PATH.exists():
        raise FileNotFoundError(
            "weekly_panel.parquet not found. Run build_weekly_panel() first."
        )

    base = pd.read_parquet(WEEKLY_PANEL_PATH, columns=["stock_id", "trade_date"])
    bm = _prepare_bm_data()
    profit = _prepare_profitability_data()
    investment = _prepare_investment_data()

    enriched = _merge_lagged(base, bm, "book")
    enriched = _merge_lagged(enriched, profit, "profit")
    enriched = _merge_lagged(enriched, investment, "investment")

    value_cols = [
        "book_equity",
        "profit_metric",
        "invest_F080601A",
        "invest_F080602A",
        "invest_F080603A",
    ]
    enriched[value_cols] = (
        enriched.groupby("stock_id", group_keys=False)[value_cols].ffill()
    )

    enriched.sort_values(["trade_date", "stock_id"]).to_parquet(
        WEEKLY_FUNDAMENTALS_PATH, index=False
    )
    print(
        f"Saved {WEEKLY_FUNDAMENTALS_PATH} with columns: "
        f"{', '.join(value_cols)}"
    )
    return enriched


if __name__ == "__main__":
    build_weekly_panel()
    build_weekly_fundamentals()
