import axios from 'axios'

const BASE = import.meta.env.VITE_API_URL || ''

export const api = {
  getTickers: () => axios.get(`${BASE}/api/tickers`).then(r => r.data),
  getPredictions: (ticker) => axios.get(`${BASE}/api/predictions/${ticker}`).then(r => r.data),
  getSentiment: (ticker) => axios.get(`${BASE}/api/sentiment/${ticker}`).then(r => r.data),
  getAllMetrics: () => axios.get(`${BASE}/api/metrics`).then(r => r.data),
  predictTomorrow: (ticker) =>
    axios.post(`${BASE}/api/predict/tomorrow/${ticker}`).then(r => r.data),
}
