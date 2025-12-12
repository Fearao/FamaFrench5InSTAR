"""将 data/raw 下的 Excel 转换为 data/processed 下的 parquet."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


BASE_BOOK = RAW_DIR / "基础数据.xlsx"
PRICE_BOOK = RAW_DIR / "行情序列后复权.xlsx"


BASE_SHEETS: Dict[str, str] = {
    "市场因子": "risk_free_rates",
    "规模因子": "size_weekly",
    "账面市值比因子": "book_to_market",
    "盈利因子": "profitability",
    "投资因子": "investment",
}


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
    df = df.dropna(how="all")
    return df


def convert_base_sheets(meta_rows: int = 2) -> Iterable[Path]:
    if not BASE_BOOK.exists():
        raise FileNotFoundError(f"未找到 {BASE_BOOK}")

    written_paths = []
    for sheet, slug in BASE_SHEETS.items():
        df = pd.read_excel(BASE_BOOK, sheet_name=sheet)
        if meta_rows:
            df = df.iloc[meta_rows:].reset_index(drop=True)
        cleaned = _clean_dataframe(df)
        output_path = PROCESSED_DIR / f"{slug}.parquet"
        cleaned.to_parquet(output_path, index=False)
        written_paths.append(output_path)
    return written_paths


def convert_price_sheets() -> Path:
    if not PRICE_BOOK.exists():
        raise FileNotFoundError(f"未找到 {PRICE_BOOK}")

    merged_frames = []
    xls = pd.ExcelFile(PRICE_BOOK)
    for sheet in xls.sheet_names:
        frame = pd.read_excel(xls, sheet_name=sheet)
        frame = frame.assign(sheet_name=sheet)
        merged_frames.append(frame)

    combined = _clean_dataframe(pd.concat(merged_frames, ignore_index=True))
    output_path = PROCESSED_DIR / "weekly_prices.parquet"
    combined.to_parquet(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meta-rows",
        type=int,
        default=2,
        help="基础数据工作表开头需要丢弃的元数据行数（包含中文列名与单位）",
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    written = list(convert_base_sheets(meta_rows=args.meta_rows))
    written.append(convert_price_sheets())

    for path in written:
        print(f"写入 {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
