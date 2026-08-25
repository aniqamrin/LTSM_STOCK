from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/predictions/{ticker}")
def get_predictions(ticker: str, request: Request):
    ticker = ticker.upper()
    df = request.app.state.predictions.get(ticker)
    if df is None:
        raise HTTPException(404, f"No predictions for {ticker}. Run the pipeline first.")

    records = df.copy()
    records["date"] = records["date"].dt.strftime("%Y-%m-%d")
    # Replace NaN with None so JSON serialises cleanly
    records = records.where(records.notna(), other=None)
    return {"ticker": ticker, "data": records.to_dict(orient="records")}
