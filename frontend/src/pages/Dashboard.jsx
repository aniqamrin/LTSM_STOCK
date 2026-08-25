import { useState, useEffect } from 'react'
import PriceChart from '../components/PriceChart'
import SentimentChart from '../components/SentimentChart'
import ModelComparisonTable from '../components/ModelComparisonTable'
import DirectionalAccuracyCard from '../components/DirectionalAccuracyCard'
import PredictTomorrowCard from '../components/PredictTomorrowCard'
import MetricsSummaryBar from '../components/MetricsSummaryBar'
import { api } from '../api/stockApi'

export default function Dashboard({ ticker }) {
  const [metrics, setMetrics] = useState([])

  useEffect(() => {
    api.getAllMetrics()
      .then(res => setMetrics(res.metrics || []))
      .catch(() => {})
  }, [])

  return (
    <main className="max-w-7xl mx-auto px-4 py-6 space-y-5">
      {/* Key metrics bar */}
      <MetricsSummaryBar metrics={metrics} ticker={ticker} />

      {/* Main price chart — full width */}
      <PriceChart ticker={ticker} />

      {/* Sentiment chart + predict tomorrow side-by-side */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div className="lg:col-span-2">
          <SentimentChart ticker={ticker} />
        </div>
        <div className="flex flex-col gap-5">
          <DirectionalAccuracyCard metrics={metrics} ticker={ticker} />
          <PredictTomorrowCard ticker={ticker} />
        </div>
      </div>

      {/* Full model comparison table */}
      <ModelComparisonTable />
    </main>
  )
}
