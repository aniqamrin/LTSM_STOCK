from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/metrics")
def get_all_metrics(request: Request):
    return {"metrics": request.app.state.metrics}


@router.get("/metrics/{ticker}")
def get_metrics_for_ticker(ticker: str, request: Request):
    ticker = ticker.upper()
    filtered = [m for m in request.app.state.metrics if m.get("ticker") == ticker]
    return {"ticker": ticker, "metrics": filtered}
