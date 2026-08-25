"""
/api/predict/tomorrow/{ticker}

Fetches the latest ~6 months of price data from yfinance, engineers features,
builds a 60-day lookback sequence using the saved scaler, and runs the saved
LSTM-Sentiment model.

The saved model expects the same 10 features built during training.
Sentiment for the "live" window is approximated using the rolling mean of
the most recent 30 days from the pre-computed sentiment CSV.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Request

# Allow importing from src/preprocessing without installing as package
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "src" / "preprocessing"))
from engineer_features import (  # noqa: E402
    compute_bollinger_bands,
    compute_macd,
    compute_rsi,
)

router = APIRouter()

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]
LOOKBACK = 60
FEATURE_COLS = [
    "Close", "log_return", "rsi_14", "macd", "macd_signal",
    "bb_upper", "bb_lower", "sma_20", "volume_log", "sentiment_score",
]
MODEL_DIR = _root / "saved_models"
SCALER_DIR = _root / "processed_data"
SENTIMENT_DIR = _root / "sentiment_scores"


def load_models() -> dict:
    """Called at app startup. Returns {ticker: keras_model}."""
    models = {}
    try:
        import tensorflow as tf
    except ImportError:
        print("  [WARN] TensorFlow not installed — live predict disabled")
        return models

    for ticker in TICKERS:
        path = MODEL_DIR / f"{ticker}_lstm_sentiment.keras"
        if path.exists():
            models[ticker] = tf.keras.models.load_model(str(path))
            print(f"  ✓ model/{ticker}")
        else:
            print(f"  [missing] model/{ticker}")
    return models


def _build_live_features(ticker: str) -> pd.DataFrame:
    """Download ~6 months of data, engineer features, fill sentiment."""
    df = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"yfinance returned no data for {ticker}")
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce").ffill()
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    df = df.sort_values("Date").reset_index(drop=True)

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["rsi_14"] = compute_rsi(df["Close"])
    df["macd"], df["macd_signal"] = compute_macd(df["Close"])
    df["bb_upper"], df["bb_lower"] = compute_bollinger_bands(df["Close"])
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["volume_log"] = np.log1p(df["Volume"])

    # Approximate live sentiment with rolling mean from saved scores
    sent_path = SENTIMENT_DIR / f"{ticker}_sentiment.csv"
    if sent_path.exists():
        sent_df = pd.read_csv(sent_path, parse_dates=["date"])
        recent_mean = sent_df["mean_score"].tail(30).mean()
    else:
        recent_mean = 0.0
    df["sentiment_score"] = recent_mean

    df = df.dropna().reset_index(drop=True)
    return df


@router.post("/predict/tomorrow/{ticker}")
def predict_tomorrow(ticker: str, request: Request):
    ticker = ticker.upper()
    model = request.app.state.models.get(ticker)
    if model is None:
        raise HTTPException(503, f"Model for {ticker} not loaded. Run train_lstm_sentiment.py first.")

    scaler_path = SCALER_DIR / f"{ticker}_scaler.pkl"
    if not scaler_path.exists():
        raise HTTPException(503, "Scaler not found. Run build_sequences.py first.")
    scaler = joblib.load(str(scaler_path))

    try:
        df = _build_live_features(ticker)
    except Exception as e:
        raise HTTPException(500, f"Feature engineering failed: {e}")

    if len(df) < LOOKBACK:
        raise HTTPException(422, f"Not enough recent data ({len(df)} rows, need {LOOKBACK})")

    features = df[FEATURE_COLS].values[-LOOKBACK:].astype(np.float32)
    features_scaled = scaler.transform(features)
    X = features_scaled.reshape(1, LOOKBACK, len(FEATURE_COLS))

    predicted_price = float(model.predict(X, verbose=0)[0][0])
    last_close = float(df["Close"].iloc[-1])
    last_date = str(df["Date"].iloc[-1])[:10]

    direction = "UP" if predicted_price > last_close else "DOWN"
    pct_change = round((predicted_price - last_close) / last_close * 100, 2)

    return {
        "ticker": ticker,
        "last_date": last_date,
        "last_close": round(last_close, 2),
        "predicted_tomorrow": round(predicted_price, 2),
        "predicted_direction": direction,
        "predicted_change_pct": pct_change,
    }
