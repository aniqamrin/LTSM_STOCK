export default function MetricsSummaryBar({ metrics, ticker }) {
  const entry = (metrics || []).find(m => m.ticker === ticker && m.model === 'LSTM-Sentiment')

  const stats = entry
    ? [
        { label: 'RMSE', value: `$${entry.rmse?.toFixed(2)}` },
        { label: 'MAE',  value: `$${entry.mae?.toFixed(2)}` },
        { label: 'MAPE', value: `${entry.mape?.toFixed(2)}%` },
        { label: 'Dir Acc', value: `${entry.directional_accuracy?.toFixed(1)}%` },
      ]
    : []

  if (!stats.length) return null

  return (
    <div className="bg-slate-800 rounded-2xl p-4 grid grid-cols-2 sm:grid-cols-4 gap-3">
      {stats.map(s => (
        <div key={s.label} className="text-center">
          <p className="text-slate-400 text-xs">{s.label}</p>
          <p className="text-white font-bold text-xl mt-0.5">{s.value}</p>
          <p className="text-slate-500 text-xs">LSTM+Sentiment</p>
        </div>
      ))}
    </div>
  )
}
