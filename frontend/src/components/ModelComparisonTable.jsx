import { useState, useEffect } from 'react'
import { api } from '../api/stockApi'

const MODELS = ['LSTM-Sentiment', 'LSTM-Only', 'ARIMA', 'SVR']
const TICKERS = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL']
const METRICS = [
  { key: 'rmse',                  label: 'RMSE',    lower: true },
  { key: 'mae',                   label: 'MAE',     lower: true },
  { key: 'mape',                  label: 'MAPE %',  lower: true },
  { key: 'directional_accuracy',  label: 'Dir Acc %', lower: false },
]

function bestInColumn(rows, metricKey, lowerIsBetter) {
  const vals = rows.map(r => r[metricKey]).filter(v => v != null)
  return lowerIsBetter ? Math.min(...vals) : Math.max(...vals)
}

export default function ModelComparisonTable() {
  const [metrics, setMetrics] = useState([])
  const [activeTicker, setActiveTicker] = useState('AAPL')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.getAllMetrics()
      .then(res => { setMetrics(res.metrics || []); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-5">
        <div className="h-48 animate-pulse bg-slate-100 rounded-xl" />
      </div>
    )
  }

  if (!metrics.length) {
    return (
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-5">
        <p className="text-slate-400 text-sm text-center py-10">
          No metrics yet. Run evaluate_all.py after training all models.
        </p>
      </div>
    )
  }

  const rows = MODELS.map(model => {
    const entry = metrics.find(m => m.ticker === activeTicker && m.model === model)
    return { model, ...entry }
  })

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-5">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
        <h2 className="text-base font-semibold text-slate-800">Model Comparison</h2>
        <div className="flex gap-1 flex-wrap">
          {TICKERS.map(t => (
            <button
              key={t}
              onClick={() => setActiveTicker(t)}
              className={`px-3 py-1 text-xs rounded-full border transition-all ${
                activeTicker === t
                  ? 'bg-slate-800 text-white border-slate-800'
                  : 'bg-white text-slate-500 border-slate-300 hover:border-slate-500'
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200">
              <th className="text-left py-2 pr-4 text-slate-500 font-medium">Model</th>
              {METRICS.map(m => (
                <th key={m.key} className="text-right py-2 px-3 text-slate-500 font-medium">
                  {m.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(row => (
              <tr
                key={row.model}
                className={`border-b border-slate-100 ${
                  row.model === 'LSTM-Sentiment' ? 'bg-emerald-50' : ''
                }`}
              >
                <td className="py-2.5 pr-4 font-medium text-slate-800">
                  {row.model}
                  {row.model === 'LSTM-Sentiment' && (
                    <span className="ml-1.5 text-xs bg-emerald-100 text-emerald-700 px-1.5 py-0.5 rounded-full">
                      proposed
                    </span>
                  )}
                </td>
                {METRICS.map(m => {
                  const val = row[m.key]
                  const best = bestInColumn(rows, m.key, m.lower)
                  const isBest = val != null && val === best
                  return (
                    <td
                      key={m.key}
                      className={`text-right py-2.5 px-3 tabular-nums ${
                        isBest ? 'font-bold text-emerald-700' : 'text-slate-700'
                      }`}
                    >
                      {val != null ? val.toFixed(2) : '—'}
                      {isBest && <span className="ml-1 text-emerald-500 text-xs">★</span>}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-slate-400 mt-3">★ Best per column. LSTM-Sentiment is the proposed model.</p>
    </div>
  )
}
