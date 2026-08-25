"""
Master pipeline runner. Execute each phase in order:
  python run_pipeline.py --phase all         # run everything (except FinBERT - do that on Colab)
  python run_pipeline.py --phase data        # download prices + filter news
  python run_pipeline.py --phase preprocess  # feature engineering + sequence building
  python run_pipeline.py --phase train       # train all 4 models for all 5 tickers
  python run_pipeline.py --phase evaluate    # compute metrics + save predictions

Note: Phase 'sentiment' must be run separately on Google Colab (GPU required).
      See src/sentiment/run_finbert.py and the README for instructions.
"""

import argparse
import subprocess
import sys
from pathlib import Path

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]


def run(script: str):
    print(f"\n{'='*60}\nRunning: {script}\n{'='*60}")
    result = subprocess.run([sys.executable, script], check=True)
    return result


def phase_data():
    run("src/data/collect_price_data.py")
    # News filtering only runs if Kaggle data has been placed in raw_data/news/
    news_file = Path("raw_data/news/kaggle_financial_news.csv")
    if news_file.exists():
        run("src/data/filter_news.py")
    else:
        print(
            "\n[WARN] raw_data/news/kaggle_financial_news.csv not found.\n"
            "Download the Kaggle dataset first:\n"
            "  https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests\n"
            "Place it at raw_data/news/kaggle_financial_news.csv then re-run --phase data\n"
        )


def phase_preprocess():
    run("src/preprocessing/engineer_features.py")
    # Sentiment scores must exist before building sequences
    missing = [t for t in TICKERS if not Path(f"sentiment_scores/{t}_sentiment.csv").exists()]
    if missing:
        print(
            f"\n[WARN] Missing sentiment scores for: {missing}\n"
            "Run src/sentiment/run_finbert.py on Google Colab first (see README).\n"
            "Continuing with build_sequences.py — missing sentiment will be filled with zeros.\n"
        )
    run("src/preprocessing/build_sequences.py")


def phase_train():
    run("src/models/train_lstm_sentiment.py")
    run("src/models/train_lstm_only.py")
    run("src/models/train_arima.py")
    run("src/models/train_svr.py")


def phase_evaluate():
    run("src/evaluation/evaluate_all.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StockSentinel pipeline runner")
    parser.add_argument(
        "--phase",
        choices=["data", "preprocess", "train", "evaluate", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.phase in ("data", "all"):
        phase_data()
    if args.phase in ("preprocess", "all"):
        phase_preprocess()
    if args.phase in ("train", "all"):
        phase_train()
    if args.phase in ("evaluate", "all"):
        phase_evaluate()

    print("\n✓ Pipeline phase(s) complete.")
