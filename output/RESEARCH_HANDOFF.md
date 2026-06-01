# TRADE_AI — Research Handoff for Claude.ai

> **What I need from you:** I am building a personal algorithmic trading platform. I will describe exactly what I have built, what data I have, and what my schema looks like. I need you to use **web search** to independently research and propose improvements across two areas: (1) data quality and quantity for ML training, and (2) model selection and accuracy. Do not limit yourself to what I mention — I want you to genuinely vet my current state and surface anything I might be missing or doing sub-optimally. Give me concrete, prioritised, actionable recommendations with sources.

---

## What I Am Building

A broker-agnostic algorithmic swing trading platform called TRADE_AI. Personal use, starting from $200 SGD, targeting $100,000 SGD via compounding over several years. Phase 1 broker is OANDA (practice first, then live).

**Trading approach:**
- Swing trading on H4 and D1 forex timeframes
- 2–4 trades per week across forex majors and minors
- Fixed fractional position sizing: 1% risk per trade
- Stop loss: ATR-based (ATR14 × configurable multiplier)
- Take profit: minimum 2:1 R:R target, but **not always the exit point** — see trailing stop below
- **Trailing stop (Option B — partial lock):** When price reaches 50% of the way to TP, stop loss moves to breakeven. The remainder of the position trails toward TP. This means actual trade outcomes are a distribution: full 2R win, partial profit between 0R and 2R, breakeven (0R), or full -1R loss. Framing accuracy purely around a binary 2R win rate is incorrect.

**Tech stack:**
- Backend: FastAPI + SQLAlchemy + SQLite (dev) → PostgreSQL (prod)
- ML: Python — XGBoost planned, not yet implemented
- Broker: OANDA v20 REST API
- Config: Pydantic Settings, all environment-variable driven, zero defaults

---

## What Is Already Built

| Milestone | Status |
|---|---|
| Instrument Discovery (127 OANDA instruments) | Complete |
| Historical Candle Ingestion (H4 + D1, ~2 years) | Complete |
| Live Price Stream (OANDA WebSocket) | Complete |
| Indicator Engine (ATR14, RSI14, swing H/L) | Complete |
| Rule-Based Signal Engine (5-condition confluence) | Complete |
| Risk Engine (stop check, R:R, position sizing, correlation filter) | Complete |
| Circuit Breaker (win rate, drawdown, consecutive loss) | Complete |
| Backtester (M7) | **Not started** |
| ML Signal Engine | **Not started** |
| Order Execution | **Not started** |
| Trade Log (model exists, nothing populates it) | **Not started** |

---

## Exact Database Schema

### `instruments`
```
id            INTEGER PK
symbol        TEXT UNIQUE NOT NULL      -- e.g. EUR_USD
display_name  TEXT NOT NULL
pip_size      FLOAT NOT NULL            -- e.g. 0.0001
pip_location  INTEGER NOT NULL          -- e.g. -4
asset_class   TEXT NOT NULL             -- forex | cfd | metal
broker_id     TEXT NOT NULL
is_active     BOOLEAN NOT NULL
```

### `candles`
```
id              INTEGER PK
instrument_id   FK → instruments.id
granularity     TEXT NOT NULL           -- H4 | D
timestamp       DATETIME NOT NULL
open            FLOAT NOT NULL
high            FLOAT NOT NULL
low             FLOAT NOT NULL
close           FLOAT NOT NULL
volume          INTEGER NOT NULL
UNIQUE(instrument_id, granularity, timestamp)
```
~127 instruments × H4 + D granularities. ~4,380 H4 bars and ~730 D bars per instrument (~2 years). Additionally 5 SGD conversion pairs (USD_SGD, GBP_SGD, CAD_SGD, SGD_JPY, SGD_CHF) with 518 D bars each — stored to convert pip P&L to SGD for equity curve accuracy.

### `indicators`
```
id              INTEGER PK
instrument_id   FK → instruments.id
granularity     TEXT NOT NULL
timestamp       DATETIME NOT NULL
atr14           FLOAT NULL              -- Wilder-smoothed ATR(14)
rsi14           FLOAT NULL              -- Wilder-smoothed RSI(14)
swing_high      FLOAT NULL              -- local swing high if this bar qualifies
swing_low       FLOAT NULL              -- local swing low if this bar qualifies
UNIQUE(instrument_id, granularity, timestamp)
```

### `signals`
```
id                INTEGER PK
instrument_id     FK → instruments.id
granularity       TEXT NOT NULL         -- H4 | D
direction         TEXT NOT NULL         -- BUY | SELL
entry             FLOAT NOT NULL
stop              FLOAT NOT NULL
target            FLOAT NOT NULL
confidence_score  INTEGER NOT NULL      -- 0–5 (count of conditions met)
score_breakdown   JSON NOT NULL         -- {"trend": bool, "rsi": bool, "structure": bool, "session": bool, "spread": bool}
status            TEXT NOT NULL         -- PENDING | APPROVED | REJECTED | EXPIRED | EXECUTED
created_at        DATETIME NOT NULL
expires_at        DATETIME NOT NULL
rejection_reason  TEXT NULL             -- STOP_TOO_WIDE | INSUFFICIENT_RR | BELOW_MIN_ORDER_SIZE | CORRELATED_POSITION_EXISTS | NO_ATR_DATA
```

### `trades` (schema defined, not yet populated — M8 pending)
```
id                          INTEGER PK
instrument_id               FK → instruments.id
direction                   TEXT                    -- BUY | SELL
entry_price                 FLOAT
exit_price                  FLOAT NULL
stop_price                  FLOAT
tp_price                    FLOAT
units                       INTEGER
risk_amount                 FLOAT
expected_pip_loss           FLOAT
actual_pip_loss             FLOAT NULL
slippage                    FLOAT NULL
rr_entry                    FLOAT                   -- expected R:R at entry
rr_actual                   FLOAT NULL              -- actual R-multiple at exit
signal_source               TEXT                    -- rule_based | ml
stage                       TEXT                    -- backtest | sandbox | live
outcome                     TEXT NULL               -- win | loss | breakeven
exit_reason                 TEXT NULL               -- tp_hit | sl_hit | trailing_stop | time_exit
opened_at                   DATETIME
closed_at                   DATETIME NULL
signal_reasoning            JSON NULL               -- full indicator snapshot at signal time
confluence_score            INTEGER NULL
session                     TEXT NULL               -- asian | london | ny | overlap
stop_method                 TEXT NULL               -- atr | structure | trailing
news_events                 JSON NULL               -- calendar events near entry
auto_classification         TEXT NULL               -- STRATEGY | NEWS | MANIPULATION | MANUAL | UNCERTAIN
classification_confidence   FLOAT NULL
human_classification_override TEXT NULL
final_classification        TEXT NULL
```

### `backtest_runs` (schema defined, not yet populated — M7 pending)
```
id                INTEGER PK
instrument_id     FK → instruments.id
strategy          TEXT
in_sample_start   DATETIME
in_sample_end     DATETIME
oos_start         DATETIME
oos_end           DATETIME
trade_count       INTEGER
win_rate          FLOAT
avg_rr            FLOAT
max_drawdown      FLOAT
sharpe            FLOAT
expectancy        FLOAT                -- expected R per trade
passed            BOOLEAN
run_at            DATETIME
```
Note: this stores **aggregate stats per run only**. There is no per-trade backtest row table yet.

### `equity_points` (schema defined, not yet populated)
```
id              INTEGER PK
timestamp       DATETIME
balance         FLOAT
unrealised_pnl  FLOAT
stage           TEXT                  -- backtest | sandbox | live
```

---

## Current Rule-Based Signal Engine

Five boolean conditions scored per instrument at each H4 candle close:

| Condition | Logic |
|---|---|
| C1 Trend | D1 last close vs D1 SMA(50) — above = BUY bias, below = SELL bias |
| C2 RSI | H4 RSI(14) inside a configurable pullback band (e.g. 40–60 for BUY, 60–80 for SELL) |
| C3 Structure | H4 bar within ATR-buffer distance of a stored swing low (BUY) or swing high (SELL) |
| C4 Session | Current UTC session is in the configured allowed set (e.g. london, ny, overlap) |
| C5 Spread | Live spread in pips is below configured max |

Signal fires when score ≥ threshold (currently 3 of 5). If both BUY and SELL clear the threshold simultaneously → suppressed (ambiguous market).

**ML plan:** Replace or augment this engine with XGBoost using a meta-labeling approach — the rule-based conditions pick direction, XGBoost decides take/skip. XGBoost does not predict direction; it only filters false positives.

---

## Current Risk Parameters

```
RISK_PCT_PER_TRADE          = 0.01   (1% per trade)
MAX_RISK_PCT_PER_TRADE      = 0.05   (5% absolute ceiling — signals rejected above this)
ATR_MULTIPLIER_MAX          = 2.0    (stop rejected if > 2× ATR14)
MIN_RR_RATIO                = 2.0    (minimum 2:1 R:R required to emit signal)
TRAILING_STOP_ACTIVATION    = 1.0    (trailing stop activates at 1:1 R:R reached)
SIGNAL_STOP_ATR_MULTIPLIER  = configurable (stop = N × ATR14)
STARTING_BALANCE            = 200 SGD
ATR_PERIOD                  = 14
SIGNAL_MIN_CONFLUENCE_SCORE = 3 (of 5)
```

---

## Planned ML Architecture

- **Label design:** Each backtested trade labeled WIN (1) or NOT-WIN (0). NOT-WIN includes: stop hit, trailing stop fired, time exit. R-multiple stored as metadata on every row for expectancy analysis.
- **Label approach:** Meta-labeling (López de Prado). Rule-based picks direction, XGBoost filters.
- **Training data source:** M7 backtester will simulate signals against 2 years of stored H4/D1 candles and generate labeled rows.
- **Train-test split:** Chronological walk-forward only — never random shuffle. 14-bar embargo gap between in-sample and OOS windows.
- **Sample weights:** Backtest rows weight=1.0. Future sandbox/live rows weight=2.0 (real labels trusted more).
- **Model versioning:** ML v1 trains on backtest data only. ML v2 trains on backtest + sandbox data (expanding window, backtest rows never discarded).
- **Retraining:** Expanding window — add new rows, retrain from scratch each cycle.

---

## Estimated Training Data Size

When M7 (Backtester) runs, labeled row count depends on signal fire rate (unknown until backtester runs):

| Instruments included | H4 bars per instrument | Estimated signal rate | Estimated labeled rows |
|---|---|---|---|
| 10 majors only | ~4,380 | 5–15% | ~2,200 – 6,600 |
| 50 instruments | ~4,380 | 5–15% | ~11,000 – 33,000 |
| All 127 instruments | ~4,380 | 5–15% | ~28,000 – 83,000 |

---

## Python Dependencies (current — before ML)

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
sqlalchemy==2.0.36
alembic==1.13.3
pydantic==2.9.2
pydantic-settings==2.5.2
httpx==0.27.2
python-dotenv==1.0.1
APScheduler==3.10.4
anthropic>=0.40.0
```

No ML libraries yet. `pandas`, `scikit-learn`, `xgboost`, `shap` all need to be added.

---

## The Two Gaps I Need You to Deeply Research

---

### GAP 1 — Data Quality and Data Quantity

I have 2 years of OANDA H4/D1 OHLCV candles, ATR14, RSI14, and swing highs/lows. I have no macro context, no sentiment, no external data beyond price. My estimated labeled training row count is 3,000–83,000 depending on how many instruments I include in backtesting.

**I want you to independently research and answer:**

1. Given my actual schema, current features, and estimated row counts, what is the honest assessment of whether this data is sufficient for an ML model to learn a generalizable forex trading signal? What does the current academic and practitioner literature say about minimum data requirements for this type of problem?

2. What external data sources, feature types, or data engineering approaches have been shown to meaningfully improve ML model performance for H4/D1 forex swing trading? Do not limit your search to what I have mentioned — find what the industry and research community actually uses and recommends in 2024/2025.

3. Are there fundamental data quality issues I should be aware of with OANDA OHLCV data specifically, or with using mid prices (not bid/ask) for H4 bar construction in a trading ML model?

4. What feature engineering approaches (beyond raw OHLCV and the two indicators I have) are supported by evidence for forex swing trading ML? Look at both academic papers and documented practitioner implementations.

5. How should I handle the fact that my 127 instruments are not independent — many share USD, EUR, or risk-sentiment exposure? Does cross-instrument training help or hurt at my scale, and what is the recommended approach?

6. What is the right way to handle the train-test split and avoid look-ahead bias when joining any external time series (macro data, sentiment) to H4 candle bars? What are the most common mistakes practitioners make here?

7. Are there data augmentation or synthetic data techniques that are valid for financial time series at my scale, or do these techniques introduce bias that is worse than training on less data?

---

### GAP 2 — Model Selection and Accuracy

My plan is XGBoost in a meta-labeling framework. My target is a profitable expected value per trade — not a fixed win rate, since my trailing stop produces a variable R-multiple distribution (some trades exit early at 0R breakeven, some at partial profit, some at full 2R, some at -1R). The ML model's job is to filter out low-probability setups, not to hit a specific win rate target.

**I want you to independently research and answer:**

8. Given my problem — binary classification (take/skip) on H4 forex, variable R-multiple outcome distribution, ~3k–30k training rows, tabular features, meta-labeling framework — what model or ensemble of models does current research (2023–2025) recommend? Is XGBoost still the right choice, or have newer approaches shown consistent improvement at this scale?

9. Since my exits are not always at fixed 2R (trailing stop creates a distribution of outcomes), should I frame this as a binary classification problem at all, or would a regression model predicting expected R-multiple per trade be more appropriate? What does the literature say about classification vs regression framing for variable-exit trading systems?

10. The meta-labeling framework from López de Prado (2018) is the theoretical basis for my approach. What has the research community found about its real-world effectiveness since publication? Are there documented cases of measurable improvement in live trading, or is this primarily a theoretical framework that underperforms simpler baselines in practice?

11. What are the specific pitfalls of applying ML to financial time series at my scale (small sample, non-stationary, correlated instruments) that I need to guard against? What does current literature say about overfitting, regime change, and distribution shift in forex ML models?

12. What is an honest, evidence-based estimate of the performance ceiling for an ML-filtered H4 forex swing trading system with ATR-based stops and a trailing stop exit? Not the theoretical ceiling — what have documented live trading systems or rigorous academic backtests actually achieved in terms of expectancy or Sharpe ratio improvement over a rule-based baseline?

13. What model evaluation metrics should I use as the gate for promoting from backtesting to sandbox (real money, practice account)? Win rate alone is insufficient given my variable exits — what combination of metrics does the literature recommend for evaluating a variable-exit trading ML system?

14. What is the recommended retraining strategy for a live ML trading system at my scale — expanding window vs rolling window, frequency, and triggering conditions? What are the risks of retraining too frequently vs too infrequently given non-stationary forex markets?

---

## What I Want Back

For each of the 14 questions:
1. What the current best practice or evidence-based consensus is (cite your sources)
2. What the specific implication is for my system given my actual constraints
3. If applicable: concrete tools, libraries, data sources, or techniques I should use

Group your response by GAP 1 and GAP 2. End with a prioritised action list: what should I do first, second, and third to get the biggest improvement in model quality per unit of implementation effort.

---

*TRADE_AI | Phase 1A | OANDA | H4/D1 forex swing trading | XGBoost meta-labeling (planned) | $200 SGD starting capital*
