from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/sentiment/{ticker}")
def get_sentiment(ticker: str, request: Request):
    ticker = ticker.upper()
    df = request.app.state.sentiment.get(ticker)
    if df is None:
        raise HTTPException(404, f"No sentiment data for {ticker}. Run run_finbert.py first.")

    records = df.copy()
    records["date"] = records["date"].dt.strftime("%Y-%m-%d")
    records = records.where(records.notna(), other=None)
    return {"ticker": ticker, "data": records.to_dict(orient="records")}
