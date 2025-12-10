"""Lightweight utilities for building factor inputs.

Currently focuses on Step 1 of the plan: assembling a weekly panel that
combines market prices, size-factor data, and a forward-filled risk-free rate
series so that weekly excess returns are ready for downstream factor work.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_PATH = DATA_PROCESSED_DIR / "weekly_panel.parquet"


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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    weekly.sort_values(["trade_date", "stock_id"]).to_parquet(OUTPUT_PATH, index=False)
    print(
        f"Saved {OUTPUT_PATH} with {len(weekly):,} rows across "
        f"{weekly['stock_id'].nunique()} stocks from {start_date.date()} to {end_date.date()}"
    )
    return weekly


if __name__ == "__main__":
    build_weekly_panel()
