"""
Train SVR baseline using scikit-learn.
Features: the same 9 price+indicator features (no sentiment), flattened lookback window.

Output: saved_models/{TICKER}_svr.pkl
        results/predictions/{TICKER}_test_predictions.csv  (column: svr_pred)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]
MODEL_DIR = Path("saved_models")
PRED_DIR = Path("results/predictions")
N_FEATURES_NO_SENT = 9


def train_ticker(ticker: str):
    print(f"\n{'='*60}\nSVR | {ticker}\n{'='*60}")
    data = np.load(f"processed_data/{ticker}_sequences.npz", allow_pickle=True)

    # SVR needs 2D input: flatten (samples, lookback, features) → (samples, lookback*features)
    X_train = data["X_train"][:, :, :N_FEATURES_NO_SENT].reshape(len(data["X_train"]), -1)
    X_val   = data["X_val"][:, :, :N_FEATURES_NO_SENT].reshape(len(data["X_val"]), -1)
    X_test  = data["X_test"][:, :, :N_FEATURES_NO_SENT].reshape(len(data["X_test"]), -1)
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
    dates_test = data["dates_test"]

    # SVR is sensitive to scale; re-scale the flattened features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # Combine train+val for final fit
    X_fit = np.vstack([X_train_s, X_val_s])
    y_fit = np.concatenate([y_train, y_val])

    print(f"  Fitting SVR on {len(X_fit)} samples (this can take several minutes)...")
    model = SVR(kernel="rbf", C=100, gamma=0.001, epsilon=0.1)
    model.fit(X_fit, y_fit)

    y_pred = model.predict(X_test_s)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # Save both SVR and its scaler together
    joblib.dump({"svr": model, "scaler": scaler}, MODEL_DIR / f"{ticker}_svr.pkl")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = PRED_DIR / f"{ticker}_test_predictions.csv"
    if pred_path.exists():
        existing = pd.read_csv(pred_path, parse_dates=["date"])
        existing["svr_pred"] = y_pred
        existing.to_csv(pred_path, index=False)
    else:
        pd.DataFrame({"date": dates_test, "actual": y_test, "svr_pred": y_pred}).to_csv(
            pred_path, index=False
        )

    print(f"  ✓ Saved → {MODEL_DIR / f'{ticker}_svr.pkl'}")
    return y_test, y_pred


if __name__ == "__main__":
    for t in TICKERS:
        try:
            train_ticker(t)
        except Exception as e:
            print(f"  [ERROR] {t}: {e}")
    print("\nSVR training complete.")
