import { useState } from 'react'
import Header from './components/Header'
import Dashboard from './pages/Dashboard'

export default function App() {
  const [ticker, setTicker] = useState('AAPL')

  return (
    <div className="min-h-screen bg-slate-100">
      <Header selectedTicker={ticker} onTickerChange={setTicker} />
      <Dashboard ticker={ticker} />
    </div>
  )
}
