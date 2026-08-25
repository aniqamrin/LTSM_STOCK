"""
Compute technical indicators from raw price CSVs.

Input:  raw_data/prices/{TICKER}_2018_2024.csv
Output: processed_data/{TICKER}_features.csv

Feature columns:
  Date, Close, log_return, rsi_14, macd, macd_signal,
  bb_upper, bb_lower, sma_20, volume_log
"""

import pandas as pd
import numpy as np
from pathlib import Path

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]
IN_DIR = Path("raw_data/prices")
OUT_DIR = Path("processed_data")


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def compute_bollinger_bands(
    series: pd.Series, period: int = 20
) -> tuple[pd.Series, pd.Series]:
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return sma + (2 * std), sma - (2 * std)


def engineer_features(ticker: str) -> pd.DataFrame:
    in_path = IN_DIR / f"{ticker}_2018_2024.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Price file not found: {in_path}. Run collect_price_data.py first.")

    df = pd.read_csv(in_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Keep only rows with valid close prices; forward-fill gaps
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce").ffill()
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["rsi_14"] = compute_rsi(df["Close"])
    df["macd"], df["macd_signal"] = compute_macd(df["Close"])
    df["bb_upper"], df["bb_lower"] = compute_bollinger_bands(df["Close"])
    df["sma_20"] = df["Close"].rolling(window=20).mean()
    df["volume_log"] = np.log1p(df["Volume"])

    keep = [
        "Date", "Close", "log_return", "rsi_14",
        "macd", "macd_signal", "bb_upper", "bb_lower",
        "sma_20", "volume_log",
    ]
    df = df[keep].dropna().reset_index(drop=True)

    out_path = OUT_DIR / f"{ticker}_features.csv"
    df.to_csv(out_path, index=False)
    print(f"  {ticker}: {len(df)} rows → {out_path}")
    return df


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Engineering features...")
    for t in TICKERS:
        try:
            engineer_features(t)
        except Exception as e:
            print(f"  [ERROR] {t}: {e}")
    print("Feature engineering complete.")
