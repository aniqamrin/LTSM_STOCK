import { useState } from 'react'
import { api } from '../api/stockApi'

export default function PredictTomorrowCard({ ticker }) {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handlePredict = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await api.predictTomorrow(ticker)
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed. Ensure models are trained and backend is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-5 flex flex-col gap-4">
      <div>
        <h2 className="text-base font-semibold text-slate-800">Predict Tomorrow</h2>
        <p className="text-xs text-slate-500 mt-1">
          Fetches latest {ticker} data from yfinance and runs the saved LSTM-Sentiment model.
        </p>
      </div>

      <button
        onClick={handlePredict}
        disabled={loading}
        className="w-full py-2.5 rounded-xl bg-slate-900 text-white font-semibold text-sm
                   hover:bg-slate-700 active:scale-95 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? 'Predicting…' : `Predict ${ticker} Tomorrow`}
      </button>

      {error && (
        <div className="text-xs text-red-600 bg-red-50 border border-red-200 rounded-xl p-3">
          {error}
        </div>
      )}

      {result && (
        <div className="rounded-xl bg-slate-50 border border-slate-200 p-4 space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-xs text-slate-500">Last Close ({result.last_date})</span>
            <span className="font-semibold text-slate-800">${result.last_close?.toFixed(2)}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-slate-500">Predicted (tomorrow)</span>
            <span className="font-bold text-lg text-slate-900">${result.predicted_tomorrow?.toFixed(2)}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-slate-500">Direction</span>
            <span
              className={`font-bold text-sm px-2.5 py-1 rounded-full ${
                result.predicted_direction === 'UP'
                  ? 'bg-emerald-100 text-emerald-700'
                  : 'bg-red-100 text-red-700'
              }`}
            >
              {result.predicted_direction === 'UP' ? '▲' : '▼'} {result.predicted_direction}{' '}
              ({result.predicted_change_pct > 0 ? '+' : ''}{result.predicted_change_pct?.toFixed(2)}%)
            </span>
          </div>
        </div>
      )}

      <p className="text-xs text-slate-400">
        For research purposes only. Not financial advice.
      </p>
    </div>
  )
}
