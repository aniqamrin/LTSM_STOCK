import { useState, useEffect } from 'react'
import {
  ComposedChart, Area, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { api } from '../api/stockApi'

export default function SentimentChart({ ticker }) {
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    api.getSentiment(ticker)
      .then(res => {
        // Sample every 5th day to avoid over-cluttering the chart
        const rows = res.data.filter((_, i) => i % 5 === 0).map(row => ({
          date: row.date,
          score: row.mean_score != null ? +row.mean_score.toFixed(3) : null,
          count: row.headline_count,
        }))
        setData(rows)
        setLoading(false)
      })
      .catch(err => {
        setError(err.response?.data?.detail || 'No sentiment data yet. Run run_finbert.py on Colab first.')
        setLoading(false)
      })
  }, [ticker])

  if (loading) return <Card><Skeleton /></Card>
  if (error) return <Card><EmptyState message={error} /></Card>

  return (
    <Card>
      <h2 className="text-base font-semibold text-slate-800 mb-4">
        {ticker} — Daily FinBERT Sentiment Score
      </h2>
      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={data} margin={{ right: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 10, fill: '#64748b' }}
            tickFormatter={d => d?.slice(0, 7)}
            interval="preserveStartEnd"
          />
          <YAxis
            yAxisId="score"
            domain={[-1, 1]}
            tick={{ fontSize: 10, fill: '#64748b' }}
            tickFormatter={v => v.toFixed(1)}
            width={40}
          />
          <YAxis
            yAxisId="count"
            orientation="right"
            tick={{ fontSize: 10, fill: '#94a3b8' }}
            width={40}
          />
          <Tooltip
            contentStyle={{ fontSize: 11, borderRadius: 8 }}
            formatter={(v, name) => [
              name === 'Sentiment Score' ? v?.toFixed(3) : v,
              name,
            ]}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <ReferenceLine yAxisId="score" y={0} stroke="#94a3b8" strokeDasharray="4 2" />
          <Bar
            yAxisId="count" dataKey="count" name="Headline Count"
            fill="#e2e8f0" opacity={0.6} radius={[2, 2, 0, 0]}
          />
          <Area
            yAxisId="score" type="monotone" dataKey="score" name="Sentiment Score"
            stroke="#10b981" fill="#d1fae5" strokeWidth={1.5} dot={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
      <p className="text-xs text-slate-400 mt-2">
        Score = P(positive) − P(negative) | Range [−1, +1] | Sampled every 5 trading days
      </p>
    </Card>
  )
}

function Card({ children }) {
  return <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-5">{children}</div>
}
function Skeleton() {
  return <div className="h-72 animate-pulse bg-slate-100 rounded-xl" />
}
function EmptyState({ message }) {
  return (
    <div className="h-60 flex flex-col items-center justify-center text-slate-400 gap-2">
      <span className="text-3xl">🗞️</span>
      <p className="text-sm text-center max-w-xs">{message}</p>
    </div>
  )
}
