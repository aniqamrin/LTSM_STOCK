import { RadialBarChart, RadialBar, PolarAngleAxis, ResponsiveContainer } from 'recharts'

export default function DirectionalAccuracyCard({ metrics, ticker }) {
  const entry = (metrics || []).find(
    m => m.ticker === ticker && m.model === 'LSTM-Sentiment'
  )
  const accuracy = entry?.directional_accuracy ?? null

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-5 flex flex-col items-center justify-center">
      <h2 className="text-base font-semibold text-slate-800 mb-2 self-start w-full">
        Directional Accuracy
      </h2>
      <p className="text-xs text-slate-500 mb-4 self-start">LSTM-Sentiment | {ticker}</p>

      {accuracy != null ? (
        <>
          <ResponsiveContainer width={160} height={160}>
            <RadialBarChart
              cx="50%" cy="50%"
              innerRadius="70%" outerRadius="90%"
              startAngle={180} endAngle={0}
              data={[{ value: accuracy, fill: accuracy >= 60 ? '#10b981' : '#f59e0b' }]}
            >
              <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
              <RadialBar dataKey="value" cornerRadius={8} background={{ fill: '#f1f5f9' }} />
            </RadialBarChart>
          </ResponsiveContainer>
          <p className="text-4xl font-bold mt-[-1.5rem] text-slate-800">
            {accuracy.toFixed(1)}%
          </p>
          <p className="text-xs text-slate-500 mt-1">
            {accuracy >= 60 ? '✓ Above random baseline (50%)' : 'Below 60% — revisit features'}
          </p>
        </>
      ) : (
        <div className="flex flex-col items-center gap-2 py-8 text-slate-400">
          <span className="text-3xl">📈</span>
          <p className="text-xs text-center">No data yet.</p>
        </div>
      )}
    </div>
  )
}
