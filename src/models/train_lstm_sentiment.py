"""
Train the hybrid LSTM + Sentiment model (the proposed model in the thesis).

Architecture (from Chapter 3.5):
  Input(60×10) → LSTM(128, return_seq) → Dropout(0.2)
               → LSTM(64) → Dropout(0.2)
               → Dense(32, ReLU) → Dense(1)

Output: saved_models/{TICKER}_lstm_sentiment.keras
        results/training_history/{TICKER}_lstm_sentiment.json
        results/predictions/{TICKER}_test_predictions.csv  (column: lstm_sentiment_pred)
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]
LOOKBACK = 60
N_FEATURES = 10
EPOCHS = 150
BATCH_SIZE = 32
MODEL_DIR = Path("saved_models")
HIST_DIR = Path("results/training_history")
PRED_DIR = Path("results/predictions")


def build_model(lookback: int = LOOKBACK, n_features: int = N_FEATURES) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(
                128,
                return_sequences=True,
                input_shape=(lookback, n_features),
                kernel_regularizer=tf.keras.regularizers.L2(1e-4),
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(
                64,
                kernel_regularizer=tf.keras.regularizers.L2(1e-4),
            ),
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
    print(f"\n{'='*60}\nLSTM-Sentiment | {ticker}\n{'='*60}")
    data = np.load(f"processed_data/{ticker}_sequences.npz", allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    dates_test = data["dates_test"]

    model = build_model()
    model.summary()

    best_path = MODEL_DIR / f"{ticker}_lstm_sentiment_best.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(best_path), save_best_only=True, monitor="val_loss"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-5
        ),
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
    model.save(MODEL_DIR / f"{ticker}_lstm_sentiment.keras")

    HIST_DIR.mkdir(parents=True, exist_ok=True)
    with open(HIST_DIR / f"{ticker}_lstm_sentiment.json", "w") as f:
        json.dump(history.history, f)

    y_pred = model.predict(X_test).flatten()

    # Persist predictions
    import pandas as pd
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = PRED_DIR / f"{ticker}_test_predictions.csv"
    df_pred = pd.DataFrame(
        {"date": dates_test, "actual": y_test, "lstm_sentiment_pred": y_pred}
    )
    if pred_path.exists():
        existing = pd.read_csv(pred_path, parse_dates=["date"])
        existing["lstm_sentiment_pred"] = y_pred
        existing.to_csv(pred_path, index=False)
    else:
        df_pred.to_csv(pred_path, index=False)

    print(f"  ✓ Model saved → {MODEL_DIR / f'{ticker}_lstm_sentiment.keras'}")
    return y_test, y_pred, dates_test


if __name__ == "__main__":
    for t in TICKERS:
        try:
            train_ticker(t)
        except Exception as e:
            print(f"  [ERROR] {t}: {e}")
    print("\nLSTM-Sentiment training complete.")
