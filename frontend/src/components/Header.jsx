const TICKERS = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL']

export default function Header({ selectedTicker, onTickerChange }) {
  return (
    <header className="bg-slate-900 text-white px-6 py-4 flex flex-col sm:flex-row items-center justify-between gap-3 shadow-lg">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 bg-emerald-500 rounded-lg flex items-center justify-center font-bold text-sm">SS</div>
        <div>
          <h1 className="text-lg font-bold leading-none">StockSentinel</h1>
          <p className="text-slate-400 text-xs">LSTM + FinBERT Sentiment Analysis</p>
        </div>
      </div>
      <div className="flex gap-2 flex-wrap justify-center">
        {TICKERS.map(t => (
          <button
            key={t}
            onClick={() => onTickerChange(t)}
            className={`px-4 py-1.5 rounded-full text-sm font-semibold transition-all ${
              selectedTicker === t
                ? 'bg-emerald-500 text-white shadow'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            {t}
          </button>
        ))}
      </div>
    </header>
  )
}
