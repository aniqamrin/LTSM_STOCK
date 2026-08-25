"""
Shared evaluation helpers.
Import compute_metrics and save_predictions from your training scripts.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PRED_DIR = Path("results/predictions")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ticker: str,
    model_name: str,
) -> dict:
    """RMSE, MAE, MAPE, and Directional Accuracy."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100)

    actual_dir = np.diff(y_true) > 0
    pred_dir   = np.diff(y_pred) > 0
    dir_acc    = float(np.mean(actual_dir == pred_dir) * 100)

    return {
        "ticker": ticker,
        "model": model_name,
        "rmse": round(rmse, 4),
        "mae":  round(mae, 4),
        "mape": round(mape, 4),
        "directional_accuracy": round(dir_acc, 2),
    }


def save_predictions(
    ticker: str,
    model_name: str,
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
):
    col = model_name.lower().replace("-", "_").replace(" ", "_") + "_pred"
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{ticker}_test_predictions.csv"

    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        df[col] = y_pred
    else:
        df = pd.DataFrame({"date": dates, "actual": y_true, col: y_pred})

    df.to_csv(path, index=False)
