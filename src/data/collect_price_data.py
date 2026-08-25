"""Download 6 years of OHLCV data for all 5 tickers via yfinance."""

import yfinance as yf
import pandas as pd
from pathlib import Path

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]
START = "2018-01-01"
END = "2024-12-31"
OUT_DIR = Path("raw_data/prices")


def download_ticker(ticker: str) -> pd.DataFrame:
    print(f"Downloading {ticker} ({START} → {END})...")
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df.reset_index(inplace=True)
    # Flatten multi-level columns yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[1] == "" else c[0] for c in df.columns]
    out_path = OUT_DIR / f"{ticker}_2018_2024.csv"
    df.to_csv(out_path, index=False)
    print(f"  ✓ {len(df)} rows saved → {out_path}")
    return df


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for t in TICKERS:
        try:
            download_ticker(t)
        except Exception as e:
            print(f"  [ERROR] {t}: {e}")
    print("\nPrice data collection complete.")
