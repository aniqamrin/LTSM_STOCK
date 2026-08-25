"""
Train LSTM-Only baseline (same architecture, no sentiment feature).
Uses only the first 9 feature columns (drops sentiment_score).

Output: saved_models/{TICKER}_lstm_only.keras
        results/training_history/{TICKER}_lstm_only.json
        results/predictions/{TICKER}_test_predictions.csv  (column: lstm_only_pred)
"""

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]
LOOKBACK = 60
N_FEATURES_NO_SENT = 9  # drop last column (sentiment_score)
EPOCHS = 150
BATCH_SIZE = 32
MODEL_DIR = Path("saved_models")
HIST_DIR = Path("results/training_history")
PRED_DIR = Path("results/predictions")


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(
                128,
                return_sequences=True,
                input_shape=(LOOKBACK, N_FEATURES_NO_SENT),
                kernel_regularizer=tf.keras.regularizers.L2(1e-4),
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64, kernel_regularizer=tf.keras.regularizers.L2(1e-4)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_ticker(ticker: str):
    print(f"\n{'='*60}\nLSTM-Only | {ticker}\n{'='*60}")
    data = np.load(f"processed_data/{ticker}_sequences.npz", allow_pickle=True)

    # Strip the last feature column (sentiment_score index = 9)
    X_train = data["X_train"][:, :, :N_FEATURES_NO_SENT]
    X_val   = data["X_val"][:, :, :N_FEATURES_NO_SENT]
    X_test  = data["X_test"][:, :, :N_FEATURES_NO_SENT]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
    dates_test = data["dates_test"]

    model = build_model()
    model.summary()

    best_path = MODEL_DIR / f"{ticker}_lstm_only_best.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(best_path), save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-5),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / f"{ticker}_lstm_only.keras")

    HIST_DIR.mkdir(parents=True, exist_ok=True)
    with open(HIST_DIR / f"{ticker}_lstm_only.json", "w") as f:
        json.dump(history.history, f)

    y_pred = model.predict(X_test).flatten()

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = PRED_DIR / f"{ticker}_test_predictions.csv"
    if pred_path.exists():
        df = pd.read_csv(pred_path, parse_dates=["date"])
        df["lstm_only_pred"] = y_pred
        df.to_csv(pred_path, index=False)
    else:
        df = pd.DataFrame({"date": dates_test, "actual": y_test, "lstm_only_pred": y_pred})
        df.to_csv(pred_path, index=False)

    print(f"  ✓ Saved → {MODEL_DIR / f'{ticker}_lstm_only.keras'}")
    return y_test, y_pred, dates_test


if __name__ == "__main__":
    for t in TICKERS:
        try:
            train_ticker(t)
        except Exception as e:
            print(f"  [ERROR] {t}: {e}")
    print("\nLSTM-Only training complete.")
