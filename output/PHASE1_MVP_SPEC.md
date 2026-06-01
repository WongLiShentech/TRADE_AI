# Phase 1 MVP — Product Requirements Document

**Platform:** Broker-agnostic swing trading platform  
**Phase 1 scope:** OANDA · Full forex universe · H4 / D1 timeframes  
**Author:** product-manager-agent  
**Date:** 2026-04-23  
**Status:** Approved for development  
**Last updated:** 2026-04-24

---

## Problem Statement

The user is a solo swing trader starting from $100 with the goal of scaling to $100,000. Manual trading is slow, inconsistent, and does not generate structured data for learning. The platform must validate a strategy in simulation before risking any real capital, then automate execution, then progressively replace rule-based signals with an ML model trained on real trade outcomes — all without changing a single line of code between stages.

---

## Users

| User | Description |
|---|---|
| Primary | Solo developer / sole trader — sole user of Phase 1 |

---

## Goals

1. Validate that a swing trading strategy is viable on real OANDA historical data before touching any account.
2. Run a fully automated bot on the OANDA practice account and collect real trade data.
3. Train an ML model on sandbox outcomes and progressively replace rule-based signals.
4. Promote to live trading via a single env var — zero code changes.
5. Enforce the risk framework in code at all times — no trade can bypass it.

---

## Three-Stage Deployment Model

This is an architectural constraint, not a feature. Every component must support all three stages via config only.

| Stage | Account type | Signal source | Purpose |
|---|---|---|---|
| 1 — Backtester | Historical candles | Rule-based (ATR / RSI / structure) | Validate strategy, confirm walk-forward viability |
| 2 — Sandbox | OANDA practice (live prices) | Rule-based → ML (progressive swap) | Collect real trade data, train ML model |
| 3 — Live | OANDA live | ML (primary) | Real capital, same code as Stage 2 |

Switch between Stage 2 and Stage 3: change `OANDA_BASE_URL` and `OANDA_STREAM_URL` to the live endpoints. Zero code changes.
Switch signal source: `SIGNAL_SOURCE=rule_based` → `SIGNAL_SOURCE=ml`.  
No code changes in either transition.

---

## Architectural Constraints

These are hard platform rules. Any design or implementation that violates them is rejected at review, not just flagged.

**AC-ARCH-1 — Stage switching is config-only, zero code changes.**
Promoting from Stage 1 → 2 → 3 requires changing env vars only. The application must be identical binary for all three stages. Any conditional code path that checks stage in business logic is a violation.

| Transition | Env var change | Code change |
|---|---|---|
| Stage 1 (backtest) → Stage 2 (sandbox) | `OANDA_BASE_URL=https://api-fxpractice.oanda.com` | None |
| Stage 2 (sandbox) → Stage 3 (live) | `OANDA_BASE_URL=https://api-fxtrade.oanda.com` | None |

**AC-ARCH-2 — Signal source is config-only, zero code changes.**
Rule-based and ML signal engines implement the same interface. Swapping between them requires one env var change. No `if SIGNAL_SOURCE == "ml"` branching in application logic — the factory resolves the correct engine at startup.

| Signal source | Env var | Code change |
|---|---|---|
| Rule-based (Phase 1A) | `SIGNAL_SOURCE=rule_based` | None |
| ML (Phase 1B) | `SIGNAL_SOURCE=ml` | None |

**AC-ARCH-3 — Both data pipelines are Phase 1A day-one deliverables.**
Historical candle ingestion (M2) and the live price stream (M3) are both built and shipped as part of Phase 1A. Neither is deferred to Phase 1B. The backtester consumes historical candles; sandbox execution consumes the live stream. Stage 2 (sandbox) is part of Phase 1A scope — therefore the live stream is required before Phase 1A is complete.

**AC-ARCH-4 — Rule-based signals are temporary by design.**
The rule-based signal engine (M5) exists solely to generate labelled trade data for the ML model. It is not the long-term signal source. The SignalEngine interface must be defined independently of either implementation so that M5 and S1 are interchangeable without touching any consuming code.

---

## Phase Structure

| Phase | Scope | Data sources required |
|---|---|---|
| 1A | Rule-based signals (ATR, RSI, structure levels). Backtester (Stage 1). Fully automated sandbox execution (Stage 2). Both data pipelines delivered. | Historical OANDA candles (M2) **and** live OANDA price stream (M3) — both day-one |
| 1B | ML model trained on accumulated sandbox trade outcomes. Signal source swapped via `SIGNAL_SOURCE=ml` — zero code changes. | Trade log DB (M10 sandbox records) |

---

## Requirements

### Must Have (Phase 1A — gates Stage 1 and Stage 2)

#### M1 — Instrument Discovery ✅ Completed: 2026-04-30
- Fetch full forex instrument list from OANDA API at runtime
- Store discovered instruments in the database
- Never hardcode any instrument name or symbol
- User selects active instruments from discovered list at runtime

#### M2 — Historical OANDA Candle Ingestion ✅ Completed: 2026-04-30
- Fetch OANDA historical candles for any discovered instrument
- Supported granularities: H4 and D1 (configurable)
- Store candles in the database
- Support incremental fetches (do not re-fetch what already exists)
- Candles used for: backtester, ATR computation, RSI computation, structure level detection

#### M3 — Live OANDA Price Stream ✅ Completed: 2026-05-13
- Connect to OANDA streaming API for any discovered instrument
- Deliver real-time price ticks to the signal engine
- Reconnect automatically on drop
- **Phase 1A day-one deliverable** — required for sandbox execution (Stage 2), which is in Phase 1A scope
- Not consumed during backtester runs (Stage 1 uses stored candles only), but the stream infrastructure is built and shipped alongside the backtester — it is not deferred

#### M4 — Technical Indicator Engine ✅ Completed: 2026-05-02
- ATR(14) on trading timeframe (H4 or D1) — Wilder smoothing, computed from stored candles
- RSI(14) on trading timeframe
- Structure level detection: swing highs and lows from recent N candles (N configurable)
- All indicators computed from stored candles (historical) and appended with each new candle (live)
- Indicators are inputs to the signal engine, not signals themselves

#### M5 — Rule-Based Signal Engine (Phase 1A signal source — temporary)
- **Temporary by design** — exists solely to generate labelled trade data for the ML model (Phase 1B). It is not the intended long-term signal source.
- Produces BUY / SELL / HOLD signal per instrument per candle close
- Inputs: ATR, RSI, structure levels (from M4)
- Each signal includes: direction, stop method, stop distance in pips, take-profit distance in pips, R:R ratio
- Implements the **SignalEngine interface** — the interface is defined first, independently of this implementation, so the ML engine (S1) is a drop-in replacement
- `SIGNAL_SOURCE=rule_based` routes through this engine; `SIGNAL_SOURCE=ml` routes through the ML engine — zero code changes required to swap
- All signals validated by RiskEngine before any action is taken
- **Phase 1A is technicals-only** — no fundamental data, no news sentiment, no calendar checks in M5. The news calendar lookup happens in M8 (trade classification only). News sentiment is Phase 1B. Fundamental data is Phase 2.
- Every signal record persists a **`signal_reasoning`** block: all indicator values at signal time (`atr_14`, `rsi_14`, `macd`, `structure_level`), `confluence_score` (integer 0–10, strict cap), `session` (asian / london / ny / overlap), `signal_source` (`rule_based` / `ml`), `stop_method` (`atr` / `structure` / `trailing`).
- **Direction-aware WikiFilter check**: before emitting a signal, the engine calls `WikiFilter.is_blocked(instrument, session, direction)`. If a (instrument, session, direction) tuple has reached `status: established` in the wiki with a `block_signals` rule, the signal is suppressed. The filter is rebuilt on every signal eval.

#### M6 — RiskEngine (platform enforcement layer) ✅ Completed: 2026-05-22
- Validates every signal before order placement or backtest entry
- Rejects (not warns) any signal where:
  - Calculated risk exceeds `MAX_RISK_PCT_PER_TRADE`
  - Stop distance exceeds `ATR_MULTIPLIER_MAX × ATR(14)` on trading timeframe
  - R:R ratio is below `MIN_RR_RATIO`
  - Calculated units fall below viable minimum (spread viability check)
- Computes position size using the locked triangle formula:
  - `pip_value_per_unit` — fetched from broker API, never hardcoded
  - `risk_amount = balance × RISK_PCT_PER_TRADE`
  - `units = floor(risk_amount / (stop_distance_pips × pip_value_per_unit))`
- Supports all stop methods: ATR-based, structure-based, trailing, time-based
- Trailing stop activates at `TRAILING_STOP_ACTIVATION` R:R (default 1:1)

#### M7 — Backtester (Stage 1 gate)
- Walk-forward validation on historical OANDA candles
- In-sample training window + out-of-sample test window (both configurable)
- Signal engine plugged into backtester — same signal engine used in live
- RiskEngine enforced on every backtest entry — no bypassing
- Outputs per run: total trades, win rate, average R:R, max drawdown, Sharpe ratio, expectancy
- Walk-forward must pass on out-of-sample data before Stage 2 is permitted (user confirms gate)
- Backtest results saved to database

#### M8 — Automated Order Execution
- **Fully autonomous — no manual approval step per trade.** Any signal that passes RiskEngine validation is executed immediately. There is no human-in-the-loop confirmation between signal and order placement.
- Places orders via BrokerClient abstraction — no direct broker API calls in application code
- Order types: market, limit, stop (configurable per signal)
- Attaches stop loss and take profit on every order — no naked positions
- Trailing stop management: monitors open positions and adjusts stop on each candle close
- Targets practice or live based on `OANDA_BASE_URL` — same code for both, zero code changes
- Enforces 2–4 trades per week maximum (configurable via `MAX_TRADES_PER_WEEK`)
- Does not enter new trades during major news window (configurable blackout periods)
- **Reads `bot_state.paused` before placing any new order.** When paused (circuit breaker tripped), refuses new entries while allowing open positions to honour their stops. Resumes only via manual `POST /api/v1/bot/resume`.
- **Auto-classifies every trade** via `services/trade_classifier.py`:
  - Queries the configured news calendar provider within ±`NEWS_CALENDAR_WINDOW_HOURS` of entry time. Default provider: `forexfactory`. Backup scaffolded: `finnhub`.
  - Computes spread anomaly: `spread_at_entry > SPREAD_ANOMALY_MULTIPLIER × instrument_typical_spread`.
  - Returns one of: `STRATEGY` / `NEWS` / `MANIPULATION` / `MANUAL` / `UNCERTAIN`, plus a `classification_confidence` (0.0–1.0). Confidence below `MIN_ML_TRAINING_CONFIDENCE` forces classification to `UNCERTAIN`.
  - Persists `signal_reasoning`, `news_events`, `auto_classification`, `classification_confidence`, `session`, `stop_method`, `confluence_score` on the Trade record.
  - **Never hardcodes** event types, currencies, or impact levels.

#### M9 — Portfolio Tracker
- Tracks account balance from `STARTING_BALANCE` configured in env
- Records equity curve over time
- Counts open positions, realised P&L, unrealised P&L
- Does not infer balance from broker — sources balance from broker API on each sync

#### M10 — Trade Log
- Persists every trade (backtest and live) with:
  - Instrument, direction, entry price, exit price, stop method, stop price
  - Expected pip loss (from signal), actual pip loss (from fill), slippage (difference)
  - Units, risk amount, R:R ratio at entry, actual outcome R:R
  - Signal source (rule_based or ml), stage (backtest / sandbox / live)
  - Outcome (win / loss / breakeven), exit reason (tp_hit / sl_hit / trailing_stop / time_exit)
  - Timestamp (open, close)
- **Every sandbox trade outcome is persisted to the database.** This includes filled trades, stopped-out trades, and trades closed at take-profit. No sandbox outcome is discarded.
- **Sandbox trade records (stage=sandbox) are the ML training dataset for Phase 1B.** The ML model (S1) queries this table directly — no separate data export step required.
- Trade log is queryable by stage, signal source, instrument, and date range to support both monitoring and ML feature extraction
- **Auto-emits a weekly markdown journal** every Sunday 00:00 UTC via APScheduler (`services/scheduler.py`):
  - Path: `{VAULT_PATH}/raw/TRADE_AI/journal/{ISO_year}-W{week}.md`
  - One section per trade — plain-language body, no jargon, includes `signal_reasoning`, classification, exit reason, news context.
  - Header: total trades, breakdown by classification (STRATEGY / NEWS / MANIPULATION / MANUAL / UNCERTAIN), net P&L, win rate on STRATEGY trades only.
  - Manual trigger: `POST /api/v1/journal/generate?week_offset=N`.
- **Followed by automated wiki ingestion** (`services/wiki_ingestor.py`):
  - Calls Anthropic API with the journal as input and existing wiki pages as context (with prompt caching).
  - Claude returns JSON ops (`create / update / append` per wiki page) which are applied with a path-traversal guard.
  - Strategy pages are tracked **per (instrument, session, direction)** — never combined.
  - Failures are logged to `wiki/TRADE_AI/log.md` as `failure | <reason>`. Trading is unaffected.
- **Followed by wiki promotion** (`services/wiki_promoter.py`): rewrites frontmatter `status` and `sample_size` based on STRATEGY trade counts per (instrument, session, direction). Status thresholds: 10 / 25 / 50 / 100. Pages cannot reach `established` until 100+ STRATEGY trades exist for that exact tuple.

#### M12 — Circuit Breaker (Part 4 — autonomous safety)
- Monitors recent STRATEGY trade outcomes and pauses the bot when thresholds breach:
  - `MIN_WIN_RATE_ALERT` over `CIRCUIT_BREAKER_WINDOW` recent trades (default 0.45 / 10 trades)
  - `MAX_DRAWDOWN_ALERT` over the same window (default 0.20)
  - `CIRCUIT_BREAKER_LOSSES` consecutive losses (default 7)
- Sets `bot_state.paused = True` and dispatches an alert via configured `ALERT_DELIVERY` channel (`log` default; `email` / `telegram` scaffolded).
- Runs daily at 00:00 UTC and after every closed trade.
- **No auto-resume.** Human must call `POST /api/v1/bot/resume`.

#### M11 — Signal Dashboard (UI)
- Functional React UI — no polish required
- Shows per instrument: current signal (BUY/SELL/HOLD), stop method, calculated units, R:R, ATR value
- Shows portfolio summary: balance, open positions, equity curve (Recharts line chart)
- Shows trade log table: last N trades, filterable by instrument and stage
- Shows backtester results: key metrics per run
- No authentication required in Phase 1 (sole user, local deployment)

---

### Should Have (Phase 1B — ML signal layer)

#### S1 — ML Signal Engine
- XGBoost classifier trained on trade log data (M10)
- Features: ATR, RSI, structure distance, hour of week, pip spread, recent win rate per instrument
- Produces same output schema as rule-based signal engine: direction, stop method, stop distance, R:R
- Model retrained incrementally as new sandbox trade data accumulates
- `SIGNAL_SOURCE=ml` routes signals through this engine instead of rule-based
- Model version and training date logged in database

#### S2 — Model Performance Tracking
- Tracks prediction accuracy, precision, recall per signal direction
- Compared against rule-based baseline on same out-of-sample period
- Stored in database, visible in dashboard

---

### Could Have (Phase 1 stretch)

#### C1 — News blackout auto-detection
- Fetch economic calendar from a configurable external source
- Automatically mark high-impact event windows as no-trade periods
- Currently: user manually configures blackout windows via env

#### C2 — Instrument correlation filter
- Reject simultaneous signals on highly correlated pairs (e.g. EUR/USD + GBP/USD same direction)
- Correlation threshold configurable via env

#### C3 — Telegram / email alert on signal or trade
- Push notification on signal detection or order fill
- Config-driven: webhook URL, no hardcoded integration

---

## Out of Scope (Phase 1)

- Phase 2 brokers (Alpaca, Binance) — scaffold only, no logic
- Multi-user auth, billing, user isolation
- Strategy marketplace
- MAS / regulatory compliance review
- Mobile UI
- Any instrument outside the forex universe (stocks, crypto, commodities)
- Manual order entry via UI (all orders are bot-placed)

---

## Acceptance Criteria

### AC-M1 — Instrument Discovery
- [ ] On startup, platform fetches full instrument list from broker API
- [ ] Instruments stored in database with symbol, display name, pip size
- [ ] No instrument symbol hardcoded anywhere in application code
- [ ] Switching broker does not require changing instrument discovery code

### AC-M2 — Historical Candle Ingestion
- [ ] Can fetch H4 and D1 candles for any instrument returned by M1
- [ ] Incremental fetch — skips already-stored candles
- [ ] Candle data queryable from database by instrument + granularity + date range

### AC-M3 — Live Price Stream
- [ ] Stream connects and delivers ticks for any selected instrument
- [ ] Auto-reconnects within 30 seconds of drop
- [ ] Stream infrastructure is built and deployable as part of Phase 1A — it is not deferred to Phase 1B
- [ ] Stream is not opened during backtester runs (Stage 1 uses stored candles) — this is a runtime behaviour, not a deferral of the feature
- [ ] Both practice (`OANDA_STREAM_URL=...fxpractice...`) and live (`OANDA_STREAM_URL=...fxtrade...`) use the same stream infrastructure — URL-only switch, no code change required

### AC-M4 — Technical Indicator Engine
- [ ] ATR(14) computed using Wilder smoothing on stored H4 or D1 candles
- [ ] RSI(14) computed on same timeframe
- [ ] Swing high/low structure levels detected from configurable lookback window
- [ ] Indicator values stored per candle in database

### AC-M5 — Rule-Based Signal Engine — complete 2026-05-13
- [x] Produces BUY/SELL for each active instrument on each candle close (HOLD = no signal persisted)
- [x] Signal includes entry, stop, target, confidence_score, score_breakdown, status, expires_at
- [x] `SIGNAL_SOURCE=rule_based` routes through `RuleBasedSignalEngine` via `get_signal_engine(settings)` factory
- [x] `SIGNAL_SOURCE=ml` will route through ML engine — zero code change required (factory dispatch)
- [x] APScheduler runs the pipeline at H4 closes (HH:01 every 4h UTC) and D1 close (21:02 UTC)
- [x] Universe trimmed to 10 majors+minors (EUR_USD, GBP_USD, USD_JPY, AUD_USD, NZD_USD, USD_CAD, USD_CHF, EUR_GBP, EUR_JPY, GBP_JPY)
- [x] Cooldown enforced — no new signal within `SIGNAL_COOLDOWN_BARS_AFTER_CLOSE × 4h` of any unresolved prior signal (closes GAP-2)
- [x] Pre-Friday-close cutoff enforced — no signals within `SIGNAL_NO_TRADE_HOURS_BEFORE_FRIDAY_CLOSE` of weekly close
- [x] Stale PENDING signals expire automatically (H4 → 4h, D1 → 24h)

### M5 Acceptance Criteria — Rule Definitions (closes GAP-8)

The rule-based engine evaluates each instrument at candle close against five conditions; a signal fires only when at least `SIGNAL_MIN_CONFLUENCE_SCORE` of the five hold for one direction (and the opposite direction does not also hit the threshold — ambiguity suppresses the signal).

**BUY conditions**
| # | Condition | Settings used |
|---|---|---|
| C1 | Trend up: D1 last close > D1 SMA(N) | `SIGNAL_TREND_SMA_PERIOD`, `SIGNAL_TREND_TIMEFRAME` |
| C2 | Pullback: H4 RSI(14) in [oversold, overbought] | `SIGNAL_RSI_OVERSOLD`, `SIGNAL_RSI_OVERBOUGHT` |
| C3 | Structure: latest H4 low within `buffer × ATR(14)` of any recent swing low (last 20 H4 indicator rows) | `SIGNAL_STRUCTURE_ATR_BUFFER`, `ATR_PERIOD` |
| C4 | Session: `classify_session(now_utc)` in allowed list | `SIGNAL_SESSION_FILTER` |
| C5 | Spread: `(ask - bid) / pip_size < max_pips` | `SIGNAL_MAX_SPREAD_PIPS` |

**SELL conditions** mirror BUY: trend down (D1 close < SMA), RSI in `[SIGNAL_RSI_OVERSOLD_SELL, SIGNAL_RSI_OVERBOUGHT_SELL]`, H4 high within ATR buffer of a recent swing high. Session and spread conditions are identical.

**Confluence threshold:** `SIGNAL_MIN_CONFLUENCE_SCORE = 3` in learning mode (current). Production target: 4. Documented in `decisions/signal-rules-v1.md`.

**Signal timeframe scope (Design A — locked 2026-05-20):** Only H4 fires signals. The D1 timeframe is used exclusively as the trend filter in condition C1; D1 never produces its own signal. Enforced two ways: (1) scheduler registers only `h4_candle_close_pipeline`; (2) `POST /signals/evaluate/{instrument}` validates `granularity` against `SIGNAL_GRANULARITIES` and returns HTTP 400 if not allowed. D1 candle + indicator ingestion is unaffected — those continue so the C1 trend filter has fresh data.

**Entry / stop / target**
- BUY: entry = `tick.ask`; stop = entry − `SIGNAL_STOP_ATR_MULTIPLIER × ATR(14)`; target = entry + `MIN_RR_RATIO × |entry − stop|`
- SELL: entry = `tick.bid`; stop = entry + `SIGNAL_STOP_ATR_MULTIPLIER × ATR(14)`; target = entry − `MIN_RR_RATIO × |entry − stop|`
- `SIGNAL_STOP_ATR_MULTIPLIER` (1.5) is the *signal* stop multiplier and is independent of `ATR_MULTIPLIER_MAX` (2.0) — the latter is the RiskEngine *rejection ceiling* applied in M6.

### AC-M6 — RiskEngine — complete 2026-05-22
- [x] Rejects (RiskValidationError) any signal exceeding `MAX_RISK_PCT_PER_TRADE` — reason `RISK_EXCEEDS_MAX`
- [x] Rejects any stop distance exceeding `ATR_MULTIPLIER_MAX × ATR(14)` — reason `STOP_TOO_WIDE`
- [x] Rejects any signal with R:R below `MIN_RR_RATIO` — reason `INSUFFICIENT_RR`
- [x] Rejects calculated units below `OANDA_MIN_UNITS` — reason `BELOW_MIN_ORDER_SIZE` (closes GAP-3)
- [x] Rejects same-direction signal on a correlated pair with an open signal — reason `CORRELATED_POSITION_EXISTS` (closes GAP-1)
- [x] Position size computed via locked triangle formula using broker API pip value (`PositionSizer`)
- [x] Zero hardcoded pip values — `OandaClient.get_pip_value()` fetches live from OANDA pricing API
- [x] Approved signals persist `status=APPROVED` + units; rejected signals persist `status=REJECTED` + `rejection_reason`
- [ ] Trailing stop activates at configured `TRAILING_STOP_ACTIVATION` threshold — deferred to M8 (trade lifecycle)
- [ ] RiskEngine unit tests cover all rejection cases — pending QA pass

### AC-M7 — Backtester
- [ ] Walk-forward runs on configurable in-sample / out-of-sample split
- [ ] RiskEngine enforced on every simulated entry
- [ ] Outputs: trade count, win rate, avg R:R, max drawdown, Sharpe, expectancy
- [ ] Results persisted to database
- [ ] Out-of-sample pass/fail gate logged — user must confirm before Stage 2

### AC-M8 — Automated Execution
- [ ] Bot executes orders autonomously — no per-trade human confirmation between signal and placement
- [ ] Orders placed via BrokerClient — no direct broker SDK calls in application logic
- [ ] Stop loss and take profit attached to every order
- [ ] `OANDA_BASE_URL` pointing to practice endpoint hits practice; pointing to live endpoint hits live — zero code change required
- [ ] `MAX_TRADES_PER_WEEK` enforced — bot refuses new entries when at limit
- [ ] Configured news blackout windows respected

### AC-M9 — Portfolio Tracker
- [ ] Balance synced from broker API on each cycle
- [ ] Equity curve persisted over time
- [ ] Open positions, realised and unrealised P&L displayed

### AC-M10 — Trade Log
- [ ] Every trade (backtest and live) persisted with all specified fields including outcome and exit reason
- [ ] Signal source field populated correctly (`rule_based` / `ml`)
- [ ] Stage field populated correctly (`backtest` / `sandbox` / `live`)
- [ ] Every sandbox trade outcome persisted regardless of result (win, loss, breakeven, stopped-out)
- [ ] Sandbox records (stage=sandbox) are directly queryable as ML training data — no transformation or export step required
- [ ] Log queryable by instrument, date range, stage, signal source, and outcome

### AC-M11 — Signal Dashboard
- [ ] Signal table shows current signal + stop + units + R:R per instrument
- [ ] Portfolio panel shows balance, open positions, equity curve chart
- [ ] Trade log table shows last N trades, filterable
- [ ] Backtester results panel shows last N runs with key metrics
- [ ] UI loads without error and reflects live database state

---

## Broker Impact

| Component | Broker dependency | Abstraction |
|---|---|---|
| Instrument discovery | BrokerClient.get_instruments() | Full abstraction |
| Historical candles | BrokerClient.get_candles() | Full abstraction |
| Live stream | BrokerClient.stream_prices() | Full abstraction |
| Pip value | BrokerClient.get_pip_value() | Full abstraction — never hardcoded |
| Order placement | BrokerClient.place_order() | Full abstraction |
| Account balance | BrokerClient.get_account() | Full abstraction |
| Account type (practice/live) | OANDA_BASE_URL + OANDA_STREAM_URL only | URL-only switch — no code change |

Switching from OANDA to any Phase 2 broker = change `BROKER` env var + implement BrokerClient for new broker. Zero application code changes.

---

## RiskEngine Impact

Every component that touches trade signals or order placement routes through `RiskEngine.validate()`:

| Component | RiskEngine interaction |
|---|---|
| Rule-based signal engine | Output validated before any action |
| ML signal engine | Output validated before any action |
| Backtester | Every simulated entry validated — same engine as live |
| Automated execution | Validated signal required before order placement |
| Position sizer | Called by RiskEngine — not called directly by application |

Any feature that bypasses RiskEngine is rejected at code review.

---

## Data Model (key entities)

| Entity | Key fields |
|---|---|
| Instrument | symbol, display_name, pip_size, asset_class, broker_id |
| Candle | instrument_id, granularity, timestamp, open, high, low, close, volume |
| Indicator | instrument_id, granularity, timestamp, atr14, rsi14, swing_high, swing_low |
| Signal | instrument_id, timestamp, direction, stop_method, stop_pips, tp_pips, rr_ratio, signal_source, validated (bool) |
| Trade | instrument_id, direction, entry_price, exit_price, stop_price, tp_price, units, risk_amount, expected_pip_loss, actual_pip_loss, slippage, rr_entry, rr_actual, signal_source, stage, opened_at, closed_at |
| BacktestRun | run_id, instrument_id, strategy, in_sample_start, in_sample_end, oos_start, oos_end, trade_count, win_rate, avg_rr, max_drawdown, sharpe, expectancy, passed (bool), run_at |
| EquityPoint | timestamp, balance, unrealised_pnl, stage |

---

## Definition of Done (Phase 1A)

Phase 1A is complete when:
1. Backtester runs walk-forward validation on OANDA historical candles, RiskEngine enforced, results saved to DB.
2. Walk-forward passes on out-of-sample data and user confirms Stage 2 gate.
3. Sandbox bot runs automated execution on OANDA practice account using live prices.
4. Trade log accumulates real entries with all required fields.
5. Signal dashboard renders current signals, portfolio state, and trade log without error.
6. No instrument symbols, pip values, or broker URLs hardcoded anywhere in application code.
7. Changing `OANDA_BASE_URL` and `OANDA_STREAM_URL` to live endpoints requires zero code changes and correctly targets live account.

Phase 1B is complete when:
1. XGBoost model trained on accumulated sandbox trade data.
2. `SIGNAL_SOURCE=ml` routes through ML engine without code changes.
3. ML signal performance tracked and visible in dashboard.
4. Model retrained on each configurable accumulation threshold.
