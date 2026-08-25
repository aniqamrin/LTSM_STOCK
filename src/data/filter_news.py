"""
Filter the Kaggle "massive-stock-news" dataset for our 5 tickers.

Expected input:  raw_data/news/kaggle_financial_news.csv
Expected schema: columns include 'stock_symbol' (or 'ticker') and 'headline' (or 'title')
                 and a date column ('date' or 'publish_date').

Output: raw_data/news/{TICKER}_headlines.csv  (columns: date, headline)
"""

import pandas as pd
from pathlib import Path

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]
IN_PATH = Path("raw_data/news/kaggle_financial_news.csv")
OUT_DIR = Path("raw_data/news")

# Column name aliases — handles different Kaggle dataset schemas
TICKER_COL_ALIASES = ["stock_symbol", "ticker", "symbol", "Stock Symbol"]
DATE_COL_ALIASES = ["date", "publish_date", "publishedAt", "Date"]
HEADLINE_COL_ALIASES = ["headline", "title", "headline_text", "Title", "Headline"]


def _resolve_col(df: pd.DataFrame, aliases: list[str]) -> str:
    for name in aliases:
        if name in df.columns:
            return name
    raise KeyError(f"None of {aliases} found in columns: {list(df.columns)}")


def filter_news():
    if not IN_PATH.exists():
        print(f"[ERROR] {IN_PATH} not found.")
        print(
            "Download from: https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests\n"
            "Place as: raw_data/news/kaggle_financial_news.csv"
        )
        return

    print(f"Loading {IN_PATH} ...")
    df = pd.read_csv(IN_PATH, low_memory=False)
    print(f"  Total rows: {len(df):,}  |  Columns: {list(df.columns)}")

    ticker_col = _resolve_col(df, TICKER_COL_ALIASES)
    date_col = _resolve_col(df, DATE_COL_ALIASES)
    headline_col = _resolve_col(df, HEADLINE_COL_ALIASES)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df = df.dropna(subset=[date_col, headline_col])
    df[date_col] = df[date_col].dt.tz_localize(None)

    for ticker in TICKERS:
        subset = df[df[ticker_col].str.upper() == ticker].copy()
        subset = subset[[date_col, headline_col]].rename(
            columns={date_col: "date", headline_col: "headline"}
        )
        subset = subset.sort_values("date").drop_duplicates()
        # Restrict to 2018-2024
        subset = subset[
            (subset["date"] >= "2018-01-01") & (subset["date"] <= "2024-12-31")
        ]
        out_path = OUT_DIR / f"{ticker}_headlines.csv"
        subset.to_csv(out_path, index=False)
        print(f"  {ticker}: {len(subset):,} headlines → {out_path}")

    print("\nNews filtering complete.")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    filter_news()
