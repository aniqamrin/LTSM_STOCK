"""
FinBERT Sentiment Pipeline — run this on Google Colab T4 GPU (free).

Steps:
  1. Upload your raw_data/news/{TICKER}_headlines.csv files to Colab
  2. Runtime → Change runtime type → T4 GPU
  3. Run this script (or paste into a Colab cell)
  4. Download the output sentiment_scores/ folder
  5. Place it in your local project root

This script needs to run ONCE. After that, training loads the CSV in milliseconds.

Required packages (pre-installed on Colab):
  pip install transformers torch tqdm
"""

import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from pathlib import Path
from tqdm import tqdm

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]
NEWS_DIR = Path("raw_data/news")
OUT_DIR = Path("sentiment_scores")
BATCH_SIZE = 32  # reduce to 16 if you get CUDA OOM errors


def setup_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    print(f"Device: {'GPU (CUDA)' if device == 0 else 'CPU (slow — use GPU!)'}")
    # ProsusAI/finbert is ~500 MB, downloaded once and cached by HuggingFace
    pipe = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        return_all_scores=True,
        device=device,
        batch_size=BATCH_SIZE,
    )
    print("✓ FinBERT model loaded.")
    return pipe


def score_batch(pipe, texts: list[str]) -> list[float]:
    """Return sentiment score in [-1, +1] for each text."""
    # Truncate to 512 chars (FinBERT input limit)
    cleaned = [str(t)[:512] if pd.notna(t) else "" for t in texts]
    results = pipe(cleaned)
    scores = []
    for result in results:
        score_map = {r["label"].lower(): r["score"] for r in result}
        s = score_map.get("positive", 0.0) - score_map.get("negative", 0.0)
        scores.append(float(s))
    return scores


def process_ticker(pipe, ticker: str) -> pd.DataFrame:
    print(f"\n{'='*50}\n{ticker}")
    news_path = NEWS_DIR / f"{ticker}_headlines.csv"
    if not news_path.exists():
        print(f"  [SKIP] {news_path} not found.")
        return pd.DataFrame()

    df = pd.read_csv(news_path, parse_dates=["date"])
    df = df.dropna(subset=["headline"]).reset_index(drop=True)
    print(f"  Headlines: {len(df):,}")

    headlines = df["headline"].tolist()
    all_scores: list[float] = []

    for i in tqdm(range(0, len(headlines), BATCH_SIZE), desc=f"  {ticker}"):
        batch = headlines[i : i + BATCH_SIZE]
        all_scores.extend(score_batch(pipe, batch))

    df["sentiment_score"] = all_scores

    # Aggregate to trading-day level
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    daily = (
        df.groupby("date")["sentiment_score"]
        .agg(mean_score="mean", headline_count="count", std_score="std")
        .reset_index()
    )
    daily.columns = ["date", "mean_score", "headline_count", "std_score"]

    # Fill std NaN (single headline days)
    daily["std_score"] = daily["std_score"].fillna(0.0)

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{ticker}_sentiment.csv"
    daily.to_csv(out_path, index=False)
    print(f"  ✓ {len(daily)} trading days → {out_path}")

    coverage = len(daily)
    print(f"  Score range: [{daily['mean_score'].min():.3f}, {daily['mean_score'].max():.3f}]")
    print(f"  Mean score:   {daily['mean_score'].mean():.3f}")
    return daily


if __name__ == "__main__":
    pipe = setup_pipeline()
    for ticker in TICKERS:
        try:
            process_ticker(pipe, ticker)
        except Exception as e:
            print(f"  [ERROR] {ticker}: {e}")
    print(
        "\n✓ All done. Download the sentiment_scores/ folder and place it in your project root.\n"
        "You never need to run FinBERT again — training uses the saved CSVs."
    )
