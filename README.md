# StockSentinel
### LSTM + FinBERT Sentiment Analysis for Stock Price Prediction
**FYP — Aniq Amrin bin Azri, UiTM**

---

## Quick Start

```bash
# 1. Clone / open the project
cd LTSM_STOCK

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
python run_pipeline.py --phase data        # download prices + filter news
python run_pipeline.py --phase preprocess  # engineer features + build sequences
# (run FinBERT on Colab — see Phase 3 below)
python run_pipeline.py --phase train       # train all 4 models
python run_pipeline.py --phase evaluate    # compute metrics → results/metrics.json

# 5. Start the API
cd backend
uvicorn main:app --reload                  # http://localhost:8000

# 6. Start the dashboard (new terminal)
cd frontend
npm install
npm run dev                                # http://localhost:5173
```

---

## Pipeline Phases

### Phase 1 — Data Collection
```
python src/data/collect_price_data.py
```
Downloads 6 years of OHLCV data for AAPL, MSFT, TSLA, NVDA, GOOGL via yfinance.
Output: `raw_data/prices/{TICKER}_2018_2024.csv`

```
python src/data/filter_news.py
```
Filters the Kaggle dataset for your 5 tickers.
**First download:** https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests
Place the CSV at `raw_data/news/kaggle_financial_news.csv`
Output: `raw_data/news/{TICKER}_headlines.csv`

---

### Phase 2 — Preprocessing
```
python src/preprocessing/engineer_features.py
python src/preprocessing/build_sequences.py
```
Computes RSI, MACD, Bollinger Bands, SMA, log return, volume.
Builds 60-day sliding window sequences with 70/15/15 train/val/test split.
Output: `processed_data/{TICKER}_features.csv`, `{TICKER}_sequences.npz`, `{TICKER}_scaler.pkl`

---

### Phase 3 — FinBERT Sentiment (run on Google Colab T4 GPU)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Runtime → Change runtime type → **T4 GPU**
3. Upload your `raw_data/news/` folder
4. Upload `src/sentiment/run_finbert.py`
5. Run the script (~2–3 hours for all 5 tickers)
6. Download the output `sentiment_scores/` folder
7. Place it in your project root

**You only run this once.** After that, training loads the CSVs in milliseconds.
Output: `sentiment_scores/{TICKER}_sentiment.csv`

---

### Phase 4 — Model Training
```
python src/models/train_lstm_sentiment.py   # proposed model
python src/models/train_lstm_only.py        # baseline 1
python src/models/train_arima.py            # baseline 2
python src/models/train_svr.py              # baseline 3
```
Run on Google Colab T4 GPU (~20 min/ticker for LSTM models).
Output: `saved_models/`, `results/training_history/`, `results/predictions/`

---

### Phase 5 — Evaluation
```
python src/evaluation/evaluate_all.py
```
Computes RMSE, MAE, MAPE, Directional Accuracy for all models × tickers.
Output: `results/metrics.json`

---

## Web Dashboard

**Backend (FastAPI)**
```bash
cd backend
uvicorn main:app --reload
# API docs: http://localhost:8000/docs
```

| Endpoint | Description |
|----------|-------------|
| `GET /api/tickers` | List of 5 tickers |
| `GET /api/predictions/{ticker}` | Actual + predicted prices (test set) |
| `GET /api/sentiment/{ticker}` | Daily FinBERT sentiment scores |
| `GET /api/metrics` | All models × tickers × metrics |
| `POST /api/predict/tomorrow/{ticker}` | Live yfinance → model → next-day prediction |

**Frontend (React + Vite)**
```bash
cd frontend
npm run dev
# Dashboard: http://localhost:5173
```

Dashboard features:
- Ticker selector (AAPL / MSFT / TSLA / NVDA / GOOGL)
- Price chart: actual vs all 4 model predictions
- Sentiment timeline with headline count overlay
- Model comparison table (RMSE, MAE, MAPE, Dir Acc) — best values highlighted
- Directional accuracy gauge (LSTM-Sentiment)
- "Predict Tomorrow" button (live inference)

---

## Project Structure

```
LTSM_STOCK/
├── raw_data/
│   ├── prices/          ← yfinance CSVs
│   └── news/            ← Kaggle headlines per ticker
├── processed_data/      ← features, sequences, scalers
├── sentiment_scores/    ← FinBERT daily scores (from Colab)
├── saved_models/        ← .keras + .pkl model files
├── results/
│   ├── metrics.json
│   ├── training_history/
│   └── predictions/
├── src/
│   ├── data/            ← collect_price_data.py, filter_news.py
│   ├── preprocessing/   ← engineer_features.py, build_sequences.py
│   ├── sentiment/       ← run_finbert.py  (Colab)
│   ├── models/          ← train_lstm_sentiment.py, _only, _arima, _svr
│   └── evaluation/      ← evaluate.py, evaluate_all.py
├── backend/             ← FastAPI (main.py + routers/)
├── frontend/            ← React + Vite + Recharts
├── notebooks/           ← EDA, sentiment EDA, training, results viz
├── requirements.txt
└── run_pipeline.py
```

---

## LSTM Architecture

```
Input  (60, 10)
  → LSTM(128, return_sequences=True, L2=1e-4)
  → Dropout(0.2)
  → LSTM(64, L2=1e-4)
  → Dropout(0.2)
  → Dense(32, ReLU)
  → Dense(1)           ← predicted next-day close price

Optimizer: Adam(lr=0.001)
Loss:      MSE
Early stopping: patience=15, restore_best_weights=True
```

**10 input features per timestep:**
`Close, log_return, RSI-14, MACD, MACD-signal, BB-upper, BB-lower, SMA-20, volume_log, sentiment_score`

---

## Total Cost: RM 0

| Resource | Tool | Cost |
|----------|------|------|
| Price data | yfinance | Free |
| News data | Kaggle dataset | Free |
| Sentiment model | ProsusAI/finbert (HuggingFace) | Free |
| GPU training | Google Colab T4 | Free |
| ML framework | TensorFlow/Keras | Free |
| Backend | FastAPI | Free |
| Frontend | React + Vite + Recharts | Free |
| Frontend hosting | Vercel | Free |
| Backend hosting | Render.com | Free |

---

## Deployment

**Frontend → Vercel**
```bash
cd frontend && npm run build
# Push to GitHub → connect repo to vercel.com → auto-deploy
```

**Backend → Render.com**
```
# Connect GitHub repo → New Web Service
# Build command:  pip install -r backend/requirements.txt
# Start command:  cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT
```
Set environment variable `VITE_API_URL` in Vercel to your Render backend URL.

---

*For research purposes only. Not financial advice.*
