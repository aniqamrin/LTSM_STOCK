import { useState, useEffect } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { api } from '../api/stockApi'

const MODEL_LINES = [
  { key: 'lstm_sentiment_pred', label: 'LSTM+Sentiment (Proposed)', color: '#10b981', dash: '6 2' },
  { key: 'lstm_only_pred',      label: 'LSTM-Only',                  color: '#ef4444', dash: '3 3' },
  { key: 'arima_pred',          label: 'ARIMA',                      color: '#f59e0b', dash: '4 4' },
  { key: 'svr_pred',            label: 'SVR',                        color: '#8b5cf6', dash: '2 4' },
]

export default function PriceChart({ ticker }) {
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [visibleModels, setVisibleModels] = useState(
    Object.fromEntries(MODEL_LINES.map(m => [m.key, true]))
  )

  useEffect(() => {
    setLoading(true)
    setError(null)
    api.getPredictions(ticker)
      .then(res => {
        const formatted = res.data.map(row => ({
          date: row.date,
          actual: row.actual != null ? +row.actual.toFixed(2) : null,
          lstm_sentiment_pred: row.lstm_sentiment_pred != null ? +row.lstm_sentiment_pred.toFixed(2) : null,
          lstm_only_pred:      row.lstm_only_pred != null ? +row.lstm_only_pred.toFixed(2) : null,
          arima_pred:          row.arima_pred != null ? +row.arima_pred.toFixed(2) : null,
          svr_pred:            row.svr_pred != null ? +row.svr_pred.toFixed(2) : null,
        }))
        setData(formatted)
        setLoading(false)
      })
      .catch(err => {
        setError(err.response?.data?.detail || 'No prediction data yet. Run the pipeline first.')
        setLoading(false)
      })
  }, [ticker])

  const toggleModel = key =>
    setVisibleModels(prev => ({ ...prev, [key]: !prev[key] }))

  if (loading) return <Card><Skeleton /></Card>
  if (error) return <Card><EmptyState message={error} /></Card>

  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
        <h2 className="text-base font-semibold text-slate-800">{ticker} — Actual vs Predicted (Test Set)</h2>
        <div className="flex flex-wrap gap-2">
          {MODEL_LINES.map(m => (
            <button
              key={m.key}
              onClick={() => toggleModel(m.key)}
              className={`text-xs px-2.5 py-1 rounded-full border transition-all ${
                visibleModels[m.key]
                  ? 'text-white border-transparent'
                  : 'bg-white text-slate-500 border-slate-300'
              }`}
              style={visibleModels[m.key] ? { backgroundColor: m.color, borderColor: m.color } : {}}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>
      <ResponsiveContainer width="100%" height={380}>
        <LineChart data={data} margin={{ right: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 10, fill: '#64748b' }}
            tickFormatter={d => d?.slice(5)}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fontSize: 10, fill: '#64748b' }}
            tickFormatter={v => `$${v}`}
            domain={['auto', 'auto']}
            width={60}
          />
          <Tooltip
            formatter={(v, name) => [v != null ? `$${v}` : '—', name]}
            contentStyle={{ fontSize: 12, borderRadius: 8 }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line
            type="monotone" dataKey="actual"
            stroke="#1e40af" strokeWidth={1.5} dot={false}
            name="Actual Price"
          />
          {MODEL_LINES.map(m =>
            visibleModels[m.key] ? (
              <Line
                key={m.key}
                type="monotone" dataKey={m.key}
                stroke={m.color} strokeWidth={m.key === 'lstm_sentiment_pred' ? 2 : 1.2}
                dot={false} strokeDasharray={m.dash}
                name={m.label}
              />
            ) : null
          )}
        </LineChart>
      </ResponsiveContainer>
    </Card>
  )
}

function Card({ children }) {
  return <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-5">{children}</div>
}
function Skeleton() {
  return <div className="h-96 animate-pulse bg-slate-100 rounded-xl" />
}
function EmptyState({ message }) {
  return (
    <div className="h-72 flex flex-col items-center justify-center text-slate-400 gap-2">
      <span className="text-3xl">📊</span>
      <p className="text-sm text-center max-w-xs">{message}</p>
    </div>
  )
}
