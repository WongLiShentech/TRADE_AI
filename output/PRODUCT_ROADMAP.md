# Product Roadmap

**Platform:** Broker-agnostic swing trading platform  
**Last updated:** 2026-05-22

---

## How to read this table

- **ICE score:** Impact × Confidence ÷ Effort (1–10 each). Higher = prioritise first.
- **Broker dep:** Which BrokerClient method this feature depends on.
- **RiskEngine dep:** Whether RiskEngine.validate() must be called.
- **Gate:** Feature that must ship before this one can start.

---

## Phase 1A — Personal Swing Trading Tool (OANDA, H4/D1, rule-based signals)

| # | Feature | ICE | Impact | Confidence | Effort | Broker dep | RiskEngine dep | Gate | Status |
|---|---|---|---|---|---|---|---|---|---|
| M1 | Instrument Discovery | 90 | 10 | 9 | 1 | get_instruments() | No | — | [x] 2026-04-30 |
| M2 | Historical Candle Ingestion | 81 | 9 | 9 | 1 | get_candles() | No | M1 | [x] 2026-04-30 |
| M3 | Live OANDA Price Stream | 56 | 8 | 7 | 1 | stream_prices() | No | M1 | [x] 2026-05-13 |
| M4 | Technical Indicator Engine (ATR, RSI, structure) | 64 | 8 | 8 | 1 | None (uses stored candles) | No | M2 | [x] 2026-05-02 |
| M6 | RiskEngine | 90 | 10 | 9 | 1 | get_pip_value() | Self | M4 | [x] 2026-05-22 |
| M5 | Rule-Based Signal Engine | 64 | 8 | 8 | 1 | None | Yes (M6) | M4, M6 | [x] 2026-05-13 |
| M7 | Backtester (walk-forward) | 72 | 9 | 8 | 1 | None (uses stored candles) | Yes (M6) | M5 | [ ] |
| M8 | Automated Order Execution | 56 | 8 | 7 | 1 | place_order() | Yes (M6) | M3, M5, M6, M7 gate pass | [ ] |
| M9 | Portfolio Tracker | 42 | 7 | 6 | 1 | get_account() | No | M8 | [ ] |
| M10 | Trade Log | 63 | 9 | 7 | 1 | None | No | M8 | [ ] |
| M11 | Signal Dashboard (UI) | 48 | 8 | 6 | 1 | None | No | M5, M9, M10 | [ ] |
| M12 | Circuit Breaker (autonomous safety) | 63 | 7 | 9 | 1 | None | No | M8 | [x] 2026-04-30 |

**Phase 1A gate:** Backtester walk-forward passes on out-of-sample data → user confirms → Stage 2 (sandbox) unlocked.

---

## Phase 1B — ML Signal Layer

### ML v1 — trained on backtest data (gates into sandbox)

Trained immediately after M7 completes, before sandbox starts. Sandbox runs under ML v1 from day one — rule-based engine is retired.

| # | Feature | ICE | Impact | Confidence | Effort | Broker dep | RiskEngine dep | Gate | Status |
|---|---|---|---|---|---|---|---|---|---|
| S1 | ML v1 Signal Engine (XGBoost, trained on backtest data) | 36 | 9 | 4 | 1 | None | Yes (M6) | M7 (backtest data) | [ ] |
| S2 | Model Performance Tracking | 24 | 6 | 4 | 1 | None | No | S1 | [ ] |
| S3 | News Sentiment Integration (Finnhub) | 20 | 5 | 4 | 1 | None (external API) | No | S1 | [ ] |

**Phase 1B gate (into sandbox):** ML v1 accuracy > rule-based baseline AND backtest gross win rate ≥ 55% on out-of-sample validation → user confirms → `SIGNAL_SOURCE=ml`, sandbox starts.

### ML v2 — trained on backtest + sandbox data (gates into live)

Trained after sufficient sandbox trades accumulate (weekly retraining cycle). Uses expanding window — backtest rows are never discarded.

| # | Feature | ICE | Impact | Confidence | Effort | Broker dep | RiskEngine dep | Gate | Status |
|---|---|---|---|---|---|---|---|---|---|
| S4 | ML v2 Signal Engine (backtest + sandbox data) | 36 | 9 | 4 | 1 | None | Yes (M6) | M10 (min sandbox trade count, user decides) | [ ] |

**Phase 1B gate (into live):** ML v2 beats ML v1 on out-of-sample validation → user confirms → Stage 3 (live) unlocked.

---

## Phase 1 Stretch

| # | Feature | ICE | Impact | Confidence | Effort | Broker dep | RiskEngine dep | Gate | Status |
|---|---|---|---|---|---|---|---|---|---|
| C1 | News blackout auto-detection | 14 | 7 | 2 | 1 | None (external calendar) | No | M8 | [ ] |
| C2 | Instrument correlation filter | 18 | 9 | 2 | 1 | None | No | M5 | [ ] |
| C3 | Telegram / email alerts | 12 | 6 | 2 | 1 | None | No | M8 | [ ] |

---

## Phase 2 — Multi-Asset Expansion

| # | Feature | ICE | Broker dep | RiskEngine dep | Gate | Status |
|---|---|---|---|---|---|---|
| P2-1 | Alpaca broker integration | — | BrokerClient impl | Yes | Phase 1A complete | [ ] |
| P2-2 | Binance broker integration | — | BrokerClient impl | Yes | Phase 1A complete | [ ] |
| P2-3 | Unified instrument selector across brokers | — | get_instruments() | No | P2-1 or P2-2 | [ ] |
| P2-4 | Asset class filter in UI | — | None | No | P2-3 | [ ] |
| P2-5 | Fundamental Data Layer (Finnhub + FRED) | — | None (external API) | No | P2-1 or P2-2 | [ ] |

---

## Phase 3 — Commercialisation

| # | Feature | Notes | Gate | Status |
|---|---|---|---|---|
| P3-1 | Multi-user auth + billing | MAS compliance review required | Phase 2 complete | [ ] |
| P3-2 | Strategy marketplace | MAS compliance review required | P3-1 | [ ] |
| P3-3 | Risk profiling per user | Per-user RiskEngine config | P3-1 | [ ] |
| P3-4 | API access | Rate limiting, auth | P3-1 | [ ] |
| P3-5 | MAS compliance review | Mandatory before commercialisation | All Phase 3 features | [ ] |

---

## Stage Promotion Checklist

### Stage 1 → Stage 2 (Backtester → Sandbox)
- [ ] Walk-forward validation passed on out-of-sample OANDA data
- [ ] RiskEngine rejection cases covered by unit tests
- [ ] No hardcoded instruments, pip values, or broker URLs in codebase
- [ ] `OANDA_BASE_URL` pointing to practice endpoint correctly targets practice — URL-only switch, zero code changes

### Stage 2 → Stage 3 (Sandbox → Live)
- [ ] Minimum N sandbox trades accumulated (user decides threshold)
- [ ] Win rate, avg R:R, max drawdown satisfy user-defined thresholds
- [ ] ML v2 trained on backtest + sandbox data and beats ML v1 on out-of-sample validation (mandatory — not optional)
- [ ] User explicitly confirms promotion
- [ ] `OANDA_BASE_URL` and `OANDA_STREAM_URL` pointing to live endpoints correctly target live — zero code changes
