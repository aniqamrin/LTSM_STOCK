"""
Read all prediction CSVs, compute metrics for every model × ticker, and write results/metrics.json.
Run this after all four training scripts have finished.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from evaluate import compute_metrics

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]
MODEL_COLS = {
    "LSTM-Sentiment": "lstm_sentiment_pred",
    "LSTM-Only":      "lstm_only_pred",
    "ARIMA":          "arima_pred",
    "SVR":            "svr_pred",
}
PRED_DIR = Path("results/predictions")
OUT_PATH = Path("results/metrics.json")


def evaluate_all():
    all_metrics: list[dict] = []

    for ticker in TICKERS:
        pred_path = PRED_DIR / f"{ticker}_test_predictions.csv"
        if not pred_path.exists():
            print(f"[SKIP] {pred_path} not found")
            continue

        df = pd.read_csv(pred_path, parse_dates=["date"])
        y_true = df["actual"].values

        for model_name, col in MODEL_COLS.items():
            if col not in df.columns:
                print(f"  [SKIP] {ticker}/{model_name}: column '{col}' missing")
                continue
            y_pred = df[col].values
            # Drop rows where prediction is NaN
            mask = ~np.isnan(y_pred)
            metrics = compute_metrics(y_true[mask], y_pred[mask], ticker, model_name)
            all_metrics.append(metrics)
            print(
                f"  {ticker:5s} | {model_name:18s} | "
                f"RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  "
                f"MAPE={metrics['mape']:.2f}%  DirAcc={metrics['directional_accuracy']:.1f}%"
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n✓ Metrics saved → {OUT_PATH}")
    return all_metrics


if __name__ == "__main__":
    print("Computing metrics for all models × tickers...\n")
    evaluate_all()
