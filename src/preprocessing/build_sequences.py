"""
Merge price features with sentiment scores, scale, and build sliding-window sequences.

Input:  processed_data/{TICKER}_features.csv
        sentiment_scores/{TICKER}_sentiment.csv  (optional — filled with 0 if missing)
Output: processed_data/{TICKER}_sequences.npz
        processed_data/{TICKER}_scaler.pkl

Split: 70% train | 15% val | 15% test  (chronological, no shuffle)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]
LOOKBACK = 60
FEATURE_COLS = [
    "Close", "log_return", "rsi_14", "macd", "macd_signal",
    "bb_upper", "bb_lower", "sma_20", "volume_log", "sentiment_score",
]
PROC_DIR = Path("processed_data")
SENT_DIR = Path("sentiment_scores")


def build_sequences(ticker: str) -> dict:
    price_df = pd.read_csv(PROC_DIR / f"{ticker}_features.csv", parse_dates=["Date"])

    sent_path = SENT_DIR / f"{ticker}_sentiment.csv"
    if sent_path.exists():
        sent_df = pd.read_csv(sent_path, parse_dates=["date"])
        sent_df = sent_df.rename(columns={"date": "Date", "mean_score": "sentiment_score"})
        df = price_df.merge(sent_df[["Date", "sentiment_score"]], on="Date", how="left")
    else:
        print(f"  [WARN] {ticker}: no sentiment file, using 0.0")
        df = price_df.copy()
        df["sentiment_score"] = 0.0

    # Fill missing sentiment with 30-day rolling mean then 0
    df["sentiment_score"] = (
        df["sentiment_score"]
        .fillna(df["sentiment_score"].rolling(30, min_periods=1).mean())
        .fillna(0.0)
    )
    df = df.sort_values("Date").reset_index(drop=True)

    features = df[FEATURE_COLS].values.astype(np.float32)
    target = df["Close"].values.astype(np.float32)
    dates = df["Date"].values

    n = len(features)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    # Fit scaler on training rows only — prevents data leakage
    scaler = MinMaxScaler()
    scaler.fit(features[:train_end])
    features_scaled = scaler.transform(features)

    # Save scaler for use during live inference
    joblib.dump(scaler, PROC_DIR / f"{ticker}_scaler.pkl")

    X, y = [], []
    for i in range(LOOKBACK, n):
        X.append(features_scaled[i - LOOKBACK : i])
        y.append(target[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    seq_dates = dates[LOOKBACK:]

    # Shift split indices by LOOKBACK
    t_end = train_end - LOOKBACK
    v_end = val_end - LOOKBACK

    splits = {
        "X_train": X[:t_end],          "y_train": y[:t_end],
        "X_val":   X[t_end:v_end],     "y_val":   y[t_end:v_end],
        "X_test":  X[v_end:],          "y_test":  y[v_end:],
        "dates_train": seq_dates[:t_end],
        "dates_val":   seq_dates[t_end:v_end],
        "dates_test":  seq_dates[v_end:],
    }

    out_path = PROC_DIR / f"{ticker}_sequences.npz"
    np.savez(
        out_path,
        X_train=splits["X_train"], y_train=splits["y_train"],
        X_val=splits["X_val"],     y_val=splits["y_val"],
        X_test=splits["X_test"],   y_test=splits["y_test"],
        dates_train=splits["dates_train"],
        dates_val=splits["dates_val"],
        dates_test=splits["dates_test"],
    )

    print(
        f"  {ticker}: Train={len(splits['X_train'])} | "
        f"Val={len(splits['X_val'])} | Test={len(splits['X_test'])} → {out_path}"
    )
    return splits


if __name__ == "__main__":
    print("Building sequences...")
    for t in TICKERS:
        try:
            build_sequences(t)
        except Exception as e:
            print(f"  [ERROR] {t}: {e}")
    print("Sequence building complete.")
