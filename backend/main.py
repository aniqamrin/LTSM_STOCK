"""
StockSentinel FastAPI backend.

At startup:
  - All pre-computed CSVs are loaded into app.state
  - Saved Keras models are loaded for live inference
  - No external calls during serving (except /predict/tomorrow which fetches yfinance)
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import metrics, predictions, sentiment, live_predict

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]

# Paths are relative to project root (run uvicorn from project root)
RESULTS_DIR = Path("results")
SENTIMENT_DIR = Path("sentiment_scores")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("── StockSentinel startup ──────────────────────────")

    # Load prediction CSVs
    app.state.predictions = {}
    for ticker in TICKERS:
        p = RESULTS_DIR / "predictions" / f"{ticker}_test_predictions.csv"
        if p.exists():
            app.state.predictions[ticker] = pd.read_csv(p, parse_dates=["date"])
            print(f"  ✓ predictions/{ticker}")
        else:
            print(f"  [missing] predictions/{ticker} — run pipeline first")

    # Load sentiment CSVs
    app.state.sentiment = {}
    for ticker in TICKERS:
        p = SENTIMENT_DIR / f"{ticker}_sentiment.csv"
        if p.exists():
            app.state.sentiment[ticker] = pd.read_csv(p, parse_dates=["date"])
            print(f"  ✓ sentiment/{ticker}")

    # Load metrics JSON
    metrics_path = RESULTS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            app.state.metrics = json.load(f)
        print(f"  ✓ metrics ({len(app.state.metrics)} records)")
    else:
        app.state.metrics = []
        print("  [missing] metrics.json — run evaluate_all.py first")

    # Load Keras models for live prediction
    app.state.models = live_predict.load_models()

    print("── Startup complete ───────────────────────────────")
    yield


app = FastAPI(title="StockSentinel API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",
        "https://*.vercel.app",    # Production
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions.router, prefix="/api")
app.include_router(metrics.router, prefix="/api")
app.include_router(sentiment.router, prefix="/api")
app.include_router(live_predict.router, prefix="/api")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "tickers_loaded": list(app.state.predictions.keys()),
        "models_loaded": list(app.state.models.keys()),
    }


@app.get("/api/tickers")
def list_tickers():
    return {"tickers": ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]}
