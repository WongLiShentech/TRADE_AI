# Trading Platform — CLAUDE.md

Personal algorithmic trading platform. Swing trading focus (H4/D1). Starting from $100, scaling to $100,000. Personal use first, commercialised if successful.

---

## Phases

| Phase | Broker | Asset Class | Status |
|---|---|---|---|
| 1 | OANDA | Full forex universe | Active |
| 2 | Alpaca | US Stocks, ETFs | Future |
| 2 | Binance | Crypto | Future |

---

## Stack

| Layer | Tech |
|---|---|
| Frontend | React + Vite + TypeScript + Tailwind CSS + Recharts + React Router |
| Backend | FastAPI + SQLAlchemy + Alembic + Pydantic + pydantic-settings |
| Database | PostgreSQL (dev via Docker + prod) — migrated from SQLite 2026-06-02 (Pre-M7 Step 1) |
| ML | Python + scikit-learn + pandas + XGBoost |
| Auth | JWT (access + refresh tokens) |
| Config | pydantic-settings (all env-driven, zero defaults) |
| Scheduler | APScheduler (BackgroundScheduler, in-process, FastAPI lifespan) |
| Wiki AI | Anthropic Python SDK (server-side, weekly journal→wiki ingestion) |
| Deploy | Docker (`docker-compose.prod.yml` + `backend/entrypoint.sh`) — host-agnostic ARM64/x86_64. Target: Raspberry Pi 400 (own hardware) or any VPS/OCI. See `DEPLOYMENT.md`. Seed via `scripts/export_slim_seed.py` (7.2GB→12MB; M1 needs zero seed rows) |

---

## Plan Mode (mandatory)

Every task in this project uses plan mode. No code is written, no files are modified, and no commands are executed until a plan has been presented and explicitly approved.

### What a plan must include

Before touching anything, Claude Code must output a structured report in this exact format:

```
## Plan: [task name]

### What I understand
[restate the task in your own words — confirms understanding before acting]

### Clarifying questions
[any questions that must be answered before proceeding — ask these first]

### Files affected
- [path] — [what will change and why]
- [path] — [what will change and why]

### Steps
1. [step] — [why]
2. [step] — [why]
3. [step] — [why]

### Risks / things to watch
- [anything that could go wrong or needs extra care]

### What I will NOT do
- [explicit list of things outside scope — prevents scope creep]

Proceed? (yes / no / modify)
```

### Rules

- Never skip the plan — even for small tasks like adding a function or fixing a typo
- Never start executing mid-plan — wait for explicit approval ("yes", "go", "proceed")
- If the user modifies the plan, reprint the updated plan and ask for approval again
- If a task turns out to be more complex mid-execution, stop and present a revised plan
- Agents invoked as sub-tasks must also present their own plan before acting

---

## Global Rules (every agent follows these without exception)

- Always ask clarifying questions before starting a complex task
- **Always present a plan and wait for approval before executing — no exceptions**
- Keep reports concise — bullet points over paragraphs
- Save all output files to the output folder
- **Data leakage in ML training is never acceptable — walk-forward splits only, no random shuffle, no feature value from after the signal timestamp, no full-dataset normalization, 14-bar embargo between in-sample and OOS. Systematic NaN across old rows requires backfill before training.**
- **Excursion data (`trades.mfe_r`/`mae_r`, everything in `trade_paths`) is post-signal by construction: valid as a LABEL, never as a FEATURE.** Train a model to *predict* how far a trade will run; never tell it how far this one ran. The one legal feature-side use is strictly point-in-time aggregates over trades that had already CLOSED before the signal being scored. Enforced by `tests/test_feature_builder.py::test_excursion_fields_are_never_model_features`, which guards the namespace rather than a fixed list.
- Cite sources when doing research
- **Zero hardcoding — no broker names, API keys, URLs, credentials, instrument pairs, asset classes, or magic strings anywhere in business logic**
- **Zero assumptions about which instruments are active — discover at runtime from broker**
- **All config is environment-driven via Pydantic Settings — no defaults on any value**
- **All broker interaction goes through `BrokerRouter` only — never instantiate a BrokerClient directly**
- **Adding a broker = add credentials to env + implement BrokerClient; set BROKER_FOREX/STOCKS/CRYPTO to route to it**
- **Before building any milestone M6+, read `C:/SecondBrain/wiki/TRADE_AI/decisions/known-gaps-deferred-fixes.md`** and check whether any gaps are scheduled for that milestone.

---

## Risk Framework (non-negotiable — enforced by platform)

### Trading style
- Swing trading on H4 and D1 timeframes only
- 2–4 trades per week maximum
- Never hold through major news events unless explicitly configured

### Position sizing
Position size is always derived — never manually entered. The formula:

```
pip_value_per_unit = fetched from broker API (never hardcoded — varies by pair)
risk_amount        = balance * RISK_PCT_PER_TRADE
units              = floor(risk_amount / (stop_distance_pips * pip_value_per_unit))
```

Key rule: **risk %, stop distance, and position size are a locked triangle. Only two are inputs. Units is always the output.**

### Risk parameters (all configurable via env, no hardcoded values)

```
RISK_PCT_PER_TRADE        # default 0.01 (1%) — learning phase
MAX_RISK_PCT_PER_TRADE    # hard ceiling 0.05 (5%) — platform rejects above this
ATR_PERIOD                # 14 (Wilder default, on trading timeframe)
ATR_MULTIPLIER_MAX        # 2.0 — reject trades where stop > 2× ATR
TRAILING_STOP_ACTIVATION  # 1.0 (activate trailing stop at 1:1 R:R)
MIN_RR_RATIO              # 2.0 — minimum risk:reward to take a trade
```

### Stop loss methods (supported, not hardcoded)

The stop method is a parameter on every trade signal — the platform supports all of these:

| Method | Logic | Best for |
|---|---|---|
| ATR-based | stop = ATR(14) × multiplier | ML signals default |
| Structure-based | stop = swing high/low + buffer | Manual trades |
| Trailing | moves with price, locks profit | Trending markets |
| Time-based | exit after max hold duration | Stale trade cleanup |

**ATR is always computed on the trading timeframe (H4 or D1), not a different chart.**
**ATR(14) covers 14 periods — on H4 that is ~56 hours. On D1 that is ~14 trading days.**

### Hard platform limits (enforced in code, not just config)

- Any trade signal exceeding MAX_RISK_PCT_PER_TRADE is **rejected**, not warned
- Any stop distance > ATR_MULTIPLIER_MAX × ATR(14) is **rejected**
- Pip value is **always fetched from broker API** — never calculated with hardcoded constants
- Position size must always be recalculated fresh per trade — never reused from a previous signal

### Lot size reference (universal constants — for documentation only)

| Lot | Units | Pip value (USD quote pairs) |
|---|---|---|
| Nano | 100 | $0.01 |
| Micro | 1,000 | $0.10 |
| Mini | 10,000 | $1.00 |
| Standard | 100,000 | $10.00 |

Note: pip value above is for USD-as-quote pairs (EUR/USD, GBP/USD, AUD/USD).
For USD-as-base pairs (USD/JPY etc.) and cross pairs, pip value varies with exchange rate.
Always use broker API pip value, never these constants, in actual calculations.

---

## Non-Negotiable Architecture Rules

### 1. No hardcoded instruments — ever
Instruments fetched at runtime via `BrokerClient.get_instruments()`, stored in DB after discovery, selected by user at runtime.

### 2. Broker abstraction is mandatory
```
app/brokers/
├── base.py       ← BrokerClient ABC + dataclasses
├── factory.py    ← get_broker_client(broker_name, settings) → BrokerClient
├── router.py     ← BrokerRouter — mandatory access layer for all application code
├── oanda.py      ← Phase 1 (active)
├── alpaca.py     ← Phase 2 scaffold
└── binance.py    ← Phase 2 scaffold
```

**BrokerRouter is the only entry point.** Application code never imports or instantiates a BrokerClient directly.

**Order placement is guarded by construction (M8-Shadow).** `BrokerClient.place_order` is a
concrete template method on the ABC: it raises `OrderPlacementDisabledError` unless
`ORDER_PLACEMENT_ENABLED` is true, then delegates to the abstract `_place_order` each broker
implements. A new broker inherits the guard whether or not its author knows the flag exists;
a subclass that skips `super().__init__(settings)` fails closed. `BrokerRouter.place_order`
re-checks before routing. **Never set `ORDER_PLACEMENT_ENABLED=true` while `ML_MODEL_PATH`
points at a `.NOT_PROMOTED` artifact** — inference refuses that combination at load time.

```python
# Usage — after instrument discovery (M1):
router.for_instrument("EUR_USD", db)      # looks up asset_class from DB → routes to correct client

# Usage — during instrument discovery (before DB populated):
router.for_asset_class("forex")           # direct asset class routing

# Usage — Phase 2 multi-broker discovery:
for asset_class, client in router.all_clients().items():
    instruments = client.get_instruments()
```

### 3. Per-asset-class broker routing

Brokers are assigned to asset classes via env vars. In Phase 1 only `BROKER_FOREX` is set.

```
BROKER_FOREX=oanda    # routes all forex instruments to OandaClient
BROKER_STOCKS=        # empty until Phase 2 (Alpaca)
BROKER_CRYPTO=        # empty until Phase 2 (Binance)
```

Adding a new broker in Phase 2: set the env var + implement BrokerClient. Zero application code changes.

### 4. Practice vs live switching is URL-only

`OANDA_ACCOUNT_TYPE` does not exist. Practice vs live is determined entirely by the URLs in `.env`:

```
# Practice:
OANDA_BASE_URL=https://api-fxpractice.oanda.com
OANDA_STREAM_URL=https://stream-fxpractice.oanda.com

# Live (change only these two lines):
OANDA_BASE_URL=https://api-fxtrade.oanda.com
OANDA_STREAM_URL=https://stream-fxtrade.oanda.com
```

No code change. No other config change. The application is identical binary for both.

### 5. Config has zero defaults (with one explicit exception)

Phase 1 required vars have no defaults. Phase 2 broker credential vars use `Optional[str] = None` so the app starts in Phase 1 without them filled.

```python
class Settings(BaseSettings):
    # Broker routing (Phase 1 required; Phase 2 optional)
    BROKER_FOREX: str
    BROKER_STOCKS: Optional[str] = None   # ← sole exception to zero-defaults
    BROKER_CRYPTO: Optional[str] = None   # ← sole exception to zero-defaults

    # OANDA (Phase 1 required — no defaults)
    OANDA_API_KEY: str
    OANDA_ACCOUNT_ID: str
    OANDA_BASE_URL: str
    OANDA_STREAM_URL: str

    # Alpaca (Phase 2 — optional)
    ALPACA_API_KEY: Optional[str] = None
    ALPACA_SECRET_KEY: Optional[str] = None
    ALPACA_BASE_URL: Optional[str] = None

    # Binance (Phase 2 — optional)
    BINANCE_API_KEY: Optional[str] = None
    BINANCE_SECRET_KEY: Optional[str] = None
    BINANCE_BASE_URL: Optional[str] = None

    # Risk, signal, infrastructure (all required, no defaults)
    SIGNAL_SOURCE: str
    RISK_PCT_PER_TRADE: float
    MAX_RISK_PCT_PER_TRADE: float
    ATR_PERIOD: int
    ATR_MULTIPLIER_MAX: float
    TRAILING_STOP_ACTIVATION: float
    MIN_RR_RATIO: float
    MAX_TRADES_PER_WEEK: int
    STARTING_BALANCE: float
    DATABASE_URL: str
    CORS_ORIGINS: str

    # Candle ingestion (M2)
    CANDLE_LOOKBACK_DAYS: int              # how far back to fetch on first sync (e.g. 730)

    # Live price stream (M3)
    STREAM_RECONNECT_DELAY_SECONDS: int    # sleep between reconnect attempts (e.g. 5)
    STREAM_MAX_RECONNECT_RETRIES: int      # give up after N consecutive failures (e.g. 10)
    STREAM_HEARTBEAT_TIMEOUT_SECONDS: int  # reserved for future heartbeat watchdog (e.g. 30)

    # Signal engine (M5) — rule-based 5-condition confluence
    SIGNAL_MIN_CONFLUENCE_SCORE: int               # 3 in learning, target 4 in production
    SIGNAL_RSI_OVERSOLD: float                     # BUY RSI band lower bound
    SIGNAL_RSI_OVERBOUGHT: float                   # BUY RSI band upper bound
    SIGNAL_RSI_OVERSOLD_SELL: float                # SELL RSI band lower bound
    SIGNAL_RSI_OVERBOUGHT_SELL: float              # SELL RSI band upper bound
    SIGNAL_TREND_SMA_PERIOD: int                   # SMA period on D1 trend filter (50)
    SIGNAL_STRUCTURE_ATR_BUFFER: float             # multiplier on ATR for swing-proximity test
    SIGNAL_MAX_SPREAD_PIPS: float                  # block signal if live spread exceeds this
    SIGNAL_SESSION_FILTER: str                     # comma list: london,ny,overlap
    SIGNAL_GRANULARITIES: str                      # comma list: H4 (Phase 1)
    SIGNAL_TREND_TIMEFRAME: str                    # D (D1 candles used for trend SMA)
    SIGNAL_COOLDOWN_BARS_AFTER_CLOSE: int          # bars-of-trading-tf to wait between signals
    SIGNAL_NO_TRADE_HOURS_BEFORE_FRIDAY_CLOSE: int # block signals near weekly close
    SIGNAL_STOP_ATR_MULTIPLIER: float              # signal stop = N * ATR(14); != ATR_MULTIPLIER_MAX

    # Vault + wiki ingestion (Part 4)
    VAULT_PATH: str                          # e.g. C:/SecondBrain
    WIKI_INGEST_ENABLED: bool
    WIKI_INGEST_MODEL: str                   # e.g. claude-sonnet-4-6
    ANTHROPIC_API_KEY: Optional[str] = None  # needed for automated wiki ingest

    # News calendar — used by M8 trade classification (not M5 signal engine)
    NEWS_CALENDAR_PROVIDER: str              # forexfactory (default) | finnhub
    NEWS_CALENDAR_WINDOW_HOURS: float        # hours before/after entry to check
    SPREAD_ANOMALY_MULTIPLIER: float         # spread > N × normal → MANIPULATION

    # News sentiment (Phase 1B scaffold — leave empty)
    NEWS_SENTIMENT_PROVIDER: Optional[str] = None
    NEWS_SENTIMENT_API_KEY: Optional[str] = None
    NEWS_SENTIMENT_LOOKBACK_MINUTES: Optional[int] = None

    # Fundamental data (Phase 2 scaffold — leave empty)
    FUNDAMENTAL_DATA_PROVIDER: Optional[str] = None
    FUNDAMENTAL_DATA_API_KEY: Optional[str] = None

    # Feature builder (Pre-M7 Step 4 — leakage firewall thresholds)
    FEATURE_STALENESS_MONTHLY_DAYS: int      # monthly series → NaN if newest release older than this (e.g. 75)
    FEATURE_STALENESS_DAILY_DAYS: int        # daily series → NaN if newest release older than this (e.g. 10)
    VIX_CHANGE_DAYS: int                      # calendar-day lookback for vix_change_5d (e.g. 5)
    WTI_CHANGE_DAYS: int                      # calendar-day lookback for wti_change_20d (e.g. 20)
    YIELD_DIFFERENTIAL_CHANGE_MONTHS: str    # comma months-back for yield-diff change features (e.g. "1,3")

    # ML training
    MIN_ML_TRAINING_CONFIDENCE: float        # trades below this go to UNCERTAIN

    # ML training pipeline (S1 — XGBoost signal filter)
    ML_LABEL_THRESHOLD_R: float              # win label if rr_actual >= this (1.0)
    ML_SEED: int                             # reproducibility (42)
    ML_VALIDATION_FRACTION: float            # chronological IS tail held out (0.2)
    ML_MIN_KEEP_FRACTION: float              # anti-gaming floor on threshold search (0.2)
    ML_MAX_DEPTH: int
    ML_N_ESTIMATORS: int
    ML_LEARNING_RATE: float
    ML_EARLY_STOPPING_ROUNDS: int

    # ML inference + shadow mode (M8-Shadow)
    SHADOW_MODE_ENABLED: bool                # record model decisions on live signals
    ORDER_PLACEMENT_ENABLED: bool            # HARD SAFETY FLAG — false = no order ever reaches a broker
    ML_MODEL_PATH: str                       # artifact path, relative to backend/
    ML_DECISION_THRESHOLD: float             # P(win) >= this → 'take'
    ML_ALLOW_UNPROMOTED_MODEL: bool          # allow loading a .NOT_PROMOTED artifact (shadow only)
    SHADOW_MAX_NAN_MODEL_FEATURES: int       # warn when a live row exceeds this many NaN model features (0)
    SHADOW_MIN_BUCKET_M1_DENSITY: float      # fraction of a bucket's minutes needed to count as observable (0.2)
    SHADOW_RESOLVER_INTERVAL_HOURS: int      # outcome-resolution cron cadence
    M1_LIVE_LOOKBACK_HOURS: int              # trailing M1 window refetched hourly (6)

    # Excursion recording (Phase A — attribution layer)
    PATH_RECORDING_ENABLED: bool             # record MFE/MAE + per-bar path on resolution
    PATH_EXTENDED_BARS: int                  # bars to keep walking AFTER the exit (20) — answers
                                             # "should we have held longer?"; tagged beyond_exit and
                                             # EXCLUDED from mfe_r/mae_r, which describe the trade
                                             # that actually happened

    # Circuit breaker + alerts
    MIN_WIN_RATE_ALERT: float                # e.g. 0.45
    MAX_DRAWDOWN_ALERT: float                # e.g. 0.20
    CIRCUIT_BREAKER_LOSSES: int              # consecutive losses before pause
    CIRCUIT_BREAKER_WINDOW: int              # trade window for win rate calc
    ALERT_DELIVERY: str                      # log (Phase 1) | email | telegram
```

### 6. .env.example (always kept in sync, all values empty)
```
# Broker routing
BROKER_FOREX=
BROKER_STOCKS=
BROKER_CRYPTO=

# OANDA (Phase 1)
OANDA_API_KEY=
OANDA_ACCOUNT_ID=
OANDA_BASE_URL=
OANDA_STREAM_URL=

# Alpaca (Phase 2 — leave blank)
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=

# Binance (Phase 2 — leave blank)
BINANCE_API_KEY=
BINANCE_SECRET_KEY=
BINANCE_BASE_URL=

# Risk parameters
RISK_PCT_PER_TRADE=
MAX_RISK_PCT_PER_TRADE=
ATR_PERIOD=
ATR_MULTIPLIER_MAX=
TRAILING_STOP_ACTIVATION=
MIN_RR_RATIO=
STARTING_BALANCE=

# Signal engine
SIGNAL_SOURCE=
MAX_TRADES_PER_WEEK=

# Infrastructure
DATABASE_URL=
CORS_ORIGINS=
```

### 7. Services directory structure

```
app/services/
├── trade_classifier.py      ← classify_trade() → STRATEGY/NEWS/MANIPULATION/MANUAL/UNCERTAIN + confidence
├── feature_builder.py       ← build_features() → single PIT feature chokepoint (M7 + live); leakage firewall
├── provenance.py            ← git_provenance() → commit + dirty at build time; NEVER raises into a run
├── session_classifier.py    ← classify_session(utc_dt) → asian/london/ny/overlap
├── journal_generator.py     ← generate_weekly_journal() → writes raw/TRADE_AI/journal/YYYY-WNN.md
├── wiki_ingestor.py         ← ingest_journal() → calls Anthropic API, writes wiki pages
├── wiki_ingestor_prompts.py ← SYSTEM_PROMPT + build_user_prompt() (prompt caching on wiki context)
├── wiki_promoter.py         ← promote_pages() → updates status/sample_size on strategy pages
├── wiki_filter.py           ← WikiFilter.is_blocked(instrument, session, direction) → (bool, reason)
├── circuit_breaker.py       ← check_circuit_breaker() → pauses bot if thresholds breached
├── scheduler.py             ← APScheduler: Sun 00:00 UTC chain + daily circuit breaker check
├── news_calendar/           ← ABC + ForexFactory (default) + Finnhub (stub)
├── news_sentiment/          ← ABC + Finnhub + ForexNewsAPI (both stubs — Phase 1B)
├── fundamental/             ← ABC + FRED (macro_data pipeline) + staleness watchdog
├── backtester/              ← M7: simulator.py, metrics.py, runner.py (walk-forward + promotion gate)
├── ml/                      ← S1: dataset, pipeline, model, policy, evaluate, shap_analysis, artifact,
│                              inference, fingerprint (byte-stable dataset digest)
├── shadow/                  ← M8-Shadow: recorder.py (live decisions), resolver.py (outcome resolution
│                              + excursion persistence — Phase A)
│                              (deploy: DEPLOYMENT.md; seed: scripts/export_slim_seed.py;
│                               optional Pi retention: scripts/prune_m1_candles.py — NEVER on dev;
                               model registry backfill: scripts/backfill_models.py;
                               run/dataset lineage: scripts/backfill_lineage.py)
└── alerts/                  ← ABC + log (default) + email + telegram (both stubs)
```

### 7b. Attribution tables (Phase A — multi-strategy / autonomy substrate)

```
strategies        ← registry of parameterised signal configs. `trades.signal_source`
                    names the ENGINE ('rule_based'), NOT the configuration; identity
                    here is `params_hash` (engine + signal-affecting params), the same
                    pattern ML artifacts use. Status: research | shadow | live | retired.
                    ⚠️ Several `shadow` at once is free. Several `live` at once DOUBLES
                    risk on a shared view — portfolio capital allocation does not exist
                    yet, so keep at most one strategy `live`.
model_decisions   ← one model's verdict per signal, MANY rows per trade. `trades.ml_*`
                    holds ONE opinion, so a challenger rescoring history overwrites the
                    champion's — making "where they disagreed, who was right?"
                    unanswerable, which is exactly what promotion depends on.
                    `is_authoritative` marks the verdict that actually governed the row.
models            ← registry of trained artifacts. A model is a FILE; the DB referred
                    to it only by a bare string, so nothing recorded what that string
                    MEANT. `strategy_id` is the load-bearing column: labels derive from
                    `rr_actual`, and `rr_actual` depends on the EXIT RULE — relabelling
                    one corpus under a pure-barrier exit instead of a trailing one flips
                    12.5% of the training set. A model is valid ONLY for the strategy
                    whose outcomes taught it; this makes that pairing checkable.
                    `passed_gate` is nullable: v1 predates the gate, and "unknown" beats
                    inventing a verdict. Hyperparameters/SHAP/feature lists stay in the
                    artifact's `.metadata.json` — this table answers "what have we got?",
                    not "how exactly was it built?".
strategies.description ← WHAT a strategy IS (stable; changes only with `params_hash`),
                    split from `notes` (WHAT HAPPENED to it; ever-growing). One column
                    carrying both stops being readable once a second strategy exists.
datasets          ← the exact corpus a model trained on: a declarative definition
                    (stage, strategy, run) PLUS a content fingerprint proving what it
                    resolved to. "Trained on strategy 2" names a LIVE QUERY whose rows
                    grow and get re-graded, so two models can claim one corpus and have
                    seen different data. UNIQUE on `fingerprint`, so a retrain over
                    identical data REUSES the row — champion and challenger sharing a
                    dataset becomes a fact the schema states. No membership join table:
                    it could only report deleted rows as absent, which the fingerprint
                    also does, while additionally catching changed feature VALUES under
                    unchanged ids (a leakage fix is exactly this).
trades.run_id     ← WHICH EXECUTION produced the row. `strategy_id` names the
                    configuration; two backtests of it (before/after a simulator fix)
                    were otherwise indistinguishable and merged into one pile, so a
                    re-run had to DELETE the prior corpus to stay unambiguous. NULL is
                    permitted and means "no run passed all three attribution gates" —
                    never a placeholder run, which would fabricate NOT NULL metrics AND
                    consume a sequence value, changing `_durable_n_trials` and thus the
                    deflated Sharpe of every future backtest.
backtest_runs     ← gains `strategy_id` (the `strategy` STRING was written from a module
                    constant and was wrong the first time two strategies existed) plus
                    `git_commit`/`git_dirty`.
trade_paths       ← per-bar excursion in R. `rr_actual` alone cannot distinguish a trade
                    that peaked at +0.84R from one that never moved. Rows past the exit
                    are tagged `beyond_exit` and excluded from mfe_r/mae_r.
trades.mfe_r/mae_r ← intrabar-exact extremes, denormalised so "average MFE of losers" is
                    one GROUP BY. Invariant, asserted in tests: mae_r ≤ rr_actual ≤ mfe_r.
```

### 8. API endpoints (all under /api/v1/)

```
POST /instruments/sync          ← M1: discover instruments from broker
GET  /instruments               ← list instruments with count + asset_class breakdown
POST /journal/generate          ← manual trigger: generate weekly journal now
POST /wiki/ingest               ← manual trigger: run wiki ingest (needs ANTHROPIC_API_KEY)
POST /wiki/promote              ← manual trigger: promote strategy page statuses
POST /wiki/rebuild-filter       ← manual trigger: rebuild WikiFilter rule set
GET  /bot/state                 ← check if bot is paused
POST /bot/resume                ← resume bot after circuit breaker pause (manual only)
```

### 9. Multi-asset concurrent trading (Phase 2 pattern)

In Phase 2, all three brokers run simultaneously. BrokerRouter handles routing transparently:

```python
# Instrument discovery across all brokers (Phase 2):
for asset_class, client in router.all_clients().items():
    for inst in client.get_instruments():
        db.add(Instrument(..., asset_class=asset_class))

# Signal → order routing (same code in Phase 1 and Phase 2):
client = router.for_instrument(signal.instrument, db)
client.place_order(order)
```

Application code is identical in Phase 1 and Phase 2 — only the env vars change.

---

## Model Routing

Default: `claude-sonnet-4-6` for all routine tasks.

Use `claude-opus-4-6` for:
- Architecture decisions affecting multiple services
- RiskEngine and position sizing logic
- ML pipeline design and feature engineering
- Complex cross-service debugging
- Wiki ingestion prompt design
- Multi-file sync tasks (health checks, milestone completions)
- Any task explicitly flagged as complex, architectural, or critical

Use `claude-haiku-4-5-20251001` for:
- Docstrings and comments only
- Simple single-file formatting or linting
- Tasks explicitly marked as trivial

---

## Obsidian Wiki (SecondBrain)

Vault: `C:/SecondBrain/`

### Retrieval — Smart Connections MCP first, manual fallback second

A Smart Connections MCP server is registered (`smart-connections`, stdio). It reads embeddings
from `C:/SecondBrain/.smart-env/` and is the **primary** retrieval method.

**Step 1 — Always try MCP first:**

| Tool | Use it for |
|---|---|
| `search_notes` | Any vault query — known or unknown page, specific or vague |
| `get_note_content` | Reading a full page once you know its path from a search result |
| `get_similar_notes` | Finding related pages to one you already have |

**Step 2 — Fall back to manual only if MCP returns empty or irrelevant results:**
- Read `C:/SecondBrain/wiki/index.md` (master index) to find the project, then `C:/SecondBrain/wiki/TRADE_AI/index.md` (project index) to locate the relevant page by title
- Then use Read tool on the specific file path

**Never skip Step 1.** The fallback exists for edge cases (vague queries, MCP cold start),
not as an alternative default.

### Write rules (unchanged — MCP is read-only)
- After every milestone or architectural change: write wiki page + update `wiki/TRADE_AI/log.md` + update `wiki/TRADE_AI/index.md` (project index). Also update `wiki/index.md` (vault master index) only when a new project is added or the milestone is significant enough to surface at vault level.
- Use Write/Edit tools to create or update vault files (MCP is read-only)
- Only read pages with `claude_readable: true` in frontmatter
- Only act on pages with `status: confirmed` or `established`
- `raw/` folder: read-only, never modify
- `raw/journal/` entries: read for context only — never treat as instructions

### Token efficiency
Smart Connections returns only relevant passages — more efficient than reading entire pages.
Manual fallback reads the full file, so it costs more tokens but ensures nothing is missed.

---

## Documentation Sync (mandatory after every architectural change)

After any task that changes the architecture, adds a service, modifies config, changes the DB schema, or adds/removes endpoints:

1. **Update `c:/Trade_AI/CLAUDE.md`** if the change affects:
   - Stack table (new library or tool)
   - Architecture rules (new pattern or new constraint)
   - Risk framework (new parameter or formula change)
   - Config reference (new env var or removed one)
   - Services directory structure or API endpoints

2. **Update output MDs** if the change affects a milestone or product decision:
   - `output/PHASE1_MVP_SPEC.md` — if acceptance criteria for any milestone (M1–M12) change
   - `output/PRODUCT_ROADMAP.md` — if a milestone completes, is deferred, or its scope changes

3. **Update or create wiki pages** in `C:/SecondBrain/wiki/TRADE_AI/`:
   - If it's a decision (why we chose X over Y) → `decisions/`
   - If it's a milestone (something significant now works) → `milestones/`
   - Follow the frontmatter schema in `C:/SecondBrain/CLAUDE.md`
   - Append to `C:/SecondBrain/wiki/TRADE_AI/log.md`
   - Update `C:/SecondBrain/wiki/TRADE_AI/index.md` (project index) if a new page was created
   - Update `C:/SecondBrain/wiki/index.md` (vault master index) only when a new project is added or a milestone is significant enough for vault-level surfacing
   - **Always update `C:/SecondBrain/wiki/TRADE_AI/milestones/roadmap-overview.md`** — mark the completed milestone as Done, advance the next milestone to Next

4. **Do NOT update `.claude/agents/*.md`** unless explicitly asked to change an agent's behaviour.
   Agent files define how agents think — they are not documentation of what was built.

This happens as part of the task — not as a separate follow-up. If the change is too small to warrant a wiki page (one-line fix, test tweak), skip step 3 but still check steps 1 and 2.

---

## Available Agents

| Agent | Invoke when... |
|---|---|
| `finance-agent` | Strategy design, risk management, signal analysis, trade sizing |
| `web-dev-agent` | Frontend/backend code, API endpoints, UI components |
| `ml-engineer-agent` | Model training, feature engineering, pipelines, backtesting |
| `product-manager-agent` | Requirements, user stories, roadmap, MVP scoping |
| `qa-agent` | Writing tests, code review, release validation, bug reports |
