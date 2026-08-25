"""
Train ARIMA baseline using pmdarima auto_arima.

Input:  raw_data/prices/{TICKER}_2018_2024.csv
Output: saved_models/{TICKER}_arima.pkl
        results/predictions/{TICKER}_test_predictions.csv  (column: arima_pred)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]
MODEL_DIR = Path("saved_models")
PRED_DIR = Path("results/predictions")


def train_ticker(ticker: str):
    print(f"\n{'='*60}\nARIMA | {ticker}\n{'='*60}")
    try:
        from pmdarima import auto_arima
    except ImportError:
        raise ImportError("Run: pip install pmdarima")

    df = pd.read_csv(f"raw_data/prices/{ticker}_2018_2024.csv", parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    prices = df["Close"].ffill().values.astype(float)

    n = len(prices)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_prices = prices[:train_end]
    test_prices = prices[val_end:]
    test_dates = df["Date"].values[val_end:]

    print("  Fitting auto_arima (this may take a few minutes)...")
    model = auto_arima(
        train_prices,
        start_p=1, start_q=1,
        max_p=5,   max_q=5,   max_d=2,
        seasonal=False,
        stepwise=True,
        error_action="ignore",
        suppress_warnings=True,
        information_criterion="aic",
    )
    print(f"  Best order: {model.order}")

    # Retrain on train+val, predict on test
    model.update(prices[train_end:val_end])
    y_pred = model.predict(n_periods=len(test_prices))
    y_test = test_prices

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / f"{ticker}_arima.pkl")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = PRED_DIR / f"{ticker}_test_predictions.csv"
    if pred_path.exists():
        existing = pd.read_csv(pred_path, parse_dates=["date"])
        existing["arima_pred"] = y_pred
        existing.to_csv(pred_path, index=False)
    else:
        pd.DataFrame({"date": test_dates, "actual": y_test, "arima_pred": y_pred}).to_csv(
            pred_path, index=False
        )

    print(f"  ✓ Saved → {MODEL_DIR / f'{ticker}_arima.pkl'}")
    return y_test, y_pred


if __name__ == "__main__":
    for t in TICKERS:
        try:
            train_ticker(t)
        except Exception as e:
            print(f"  [ERROR] {t}: {e}")
    print("\nARIMA training complete.")
