export interface Instrument {
  id: number
  symbol: string
  display_name: string
  pip_size: number
  asset_class: string
  broker_id: string
}

export interface Candle {
  id: number
  instrument_id: number
  granularity: string
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface Signal {
  id: number
  instrument_id: number
  timestamp: string
  direction: 'BUY' | 'SELL' | 'HOLD'
  stop_method: string
  stop_pips: number
  tp_pips: number
  rr_ratio: number
  signal_source: string
  validated: boolean
}

export interface Trade {
  id: number
  instrument_id: number
  direction: 'BUY' | 'SELL'
  entry_price: number
  exit_price: number | null
  stop_price: number
  tp_price: number
  units: number
  risk_amount: number
  expected_pip_loss: number
  actual_pip_loss: number | null
  slippage: number | null
  rr_entry: number
  rr_actual: number | null
  signal_source: string
  stage: 'backtest' | 'sandbox' | 'live'
  outcome: 'win' | 'loss' | 'breakeven' | null
  exit_reason: 'tp_hit' | 'sl_hit' | 'trailing_stop' | 'time_exit' | null
  opened_at: string
  closed_at: string | null
}

export interface PortfolioSummary {
  balance: number
  unrealised_pnl: number
  open_positions: number
  equity_curve: { timestamp: string; balance: number }[]
}

export interface BacktestResult {
  id: number
  instrument_id: number
  strategy: string
  trade_count: number
  win_rate: number
  avg_rr: number
  max_drawdown: number
  sharpe: number
  expectancy: number
  passed: boolean
  run_at: string
}
