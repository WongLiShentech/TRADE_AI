from datetime import datetime
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Per-asset-class broker routing
    BROKER_FOREX: str
    BROKER_STOCKS: Optional[str] = None
    BROKER_CRYPTO: Optional[str] = None

    # OANDA credentials (Phase 1 — required)
    # Practice vs live switching: change OANDA_BASE_URL and OANDA_STREAM_URL only.
    OANDA_API_KEY: str
    OANDA_ACCOUNT_ID: str
    OANDA_BASE_URL: str
    OANDA_STREAM_URL: str

    # Alpaca credentials (Phase 2 — optional)
    ALPACA_API_KEY: Optional[str] = None
    ALPACA_SECRET_KEY: Optional[str] = None
    ALPACA_BASE_URL: Optional[str] = None

    # Binance credentials (Phase 2 — optional)
    BINANCE_API_KEY: Optional[str] = None
    BINANCE_SECRET_KEY: Optional[str] = None
    BINANCE_BASE_URL: Optional[str] = None

    # Signal engine
    SIGNAL_SOURCE: str
    SIGNAL_MIN_CONFLUENCE_SCORE: int
    SIGNAL_RSI_OVERSOLD: float
    SIGNAL_RSI_OVERBOUGHT: float
    SIGNAL_RSI_OVERSOLD_SELL: float
    SIGNAL_RSI_OVERBOUGHT_SELL: float
    SIGNAL_TREND_SMA_PERIOD: int
    SIGNAL_STRUCTURE_ATR_BUFFER: float
    SIGNAL_CONDITIONS: str  # comma list naming which registered conditions this strategy evaluates
    SIGNAL_MAX_SPREAD_PIPS: float
    SIGNAL_SESSION_FILTER: str               # comma-separated, parsed in engine
    SIGNAL_GRANULARITIES: str                # comma-separated
    SIGNAL_TREND_TIMEFRAME: str
    SIGNAL_COOLDOWN_BARS_AFTER_CLOSE: int
    SIGNAL_NO_TRADE_HOURS_BEFORE_FRIDAY_CLOSE: int
    SIGNAL_STOP_ATR_MULTIPLIER: float

    # Risk parameters (all required, zero defaults)
    RISK_PCT_PER_TRADE: float
    MAX_RISK_PCT_PER_TRADE: float
    ATR_PERIOD: int
    ATR_MULTIPLIER_MAX: float
    TRAILING_STOP_ACTIVATION: float
    MIN_RR_RATIO: float
    MAX_TRADES_PER_WEEK: int
    STARTING_BALANCE: float
    OANDA_MIN_UNITS: int

    # Infrastructure (required)
    DATABASE_URL: str
    CORS_ORIGINS: str

    # Candle ingestion (M2)
    CANDLE_LOOKBACK_DAYS: int  # how far back to fetch on first sync per instrument
    # Live M1 Bid/Ask ingestion (M8-Shadow Phase 3 / GAP-13). Trailing window in HOURS
    # re-fetched by the hourly M1 job: [now - M1_LIVE_LOOKBACK_HOURS, now]. Must exceed
    # the job's cadence so consecutive runs OVERLAP (short outages self-heal); the cost
    # is bounded and constant per run regardless of how long the platform was down.
    M1_LIVE_LOOKBACK_HOURS: int

    # Live price stream (M3)
    STREAM_RECONNECT_DELAY_SECONDS: int
    STREAM_MAX_RECONNECT_RETRIES: int
    STREAM_HEARTBEAT_TIMEOUT_SECONDS: int

    # Indicator engine (M4)
    SWING_LOOKBACK_PERIODS: int  # candles either side for swing high/low detection (CENTRED — see models/indicator.py)
    SIGNAL_DONCHIAN_PERIOD: int  # trailing bars for the Donchian channel (causal; backs c3_structure)

    # Vault (Part 1) — server-side wiki ingestion writes journal here
    VAULT_PATH: str

    # News calendar (Part 2 — used by M8 trade classification only)
    NEWS_CALENDAR_PROVIDER: str
    NEWS_CALENDAR_WINDOW_HOURS: float
    SPREAD_ANOMALY_MULTIPLIER: float
    NEWS_REFRESH_FORWARD_DAYS: int                     # forward window for the live news-refresh job (e.g. 14)
    FINNHUB_API_KEY: Optional[str] = None              # free Finnhub key — broad news features (verify-free sub-step)

    # News sentiment (Phase 1B scaffold — empty until needed)
    NEWS_SENTIMENT_PROVIDER: Optional[str] = None
    NEWS_SENTIMENT_API_KEY: Optional[str] = None
    NEWS_SENTIMENT_LOOKBACK_MINUTES: Optional[int] = None

    # Fundamental data (FRED macro_data pipeline — Pre-M7 Step 2)
    FUNDAMENTAL_DATA_PROVIDER: Optional[str] = None   # "fred"
    FUNDAMENTAL_DATA_API_KEY: Optional[str] = None    # FRED API key
    MACRO_LOOKBACK_DAYS: int                           # how far back to backfill macro_data (e.g. 900)
    # Live fundamentals orchestration (Cron A weekly + Cron B intraday)
    FUNDAMENTAL_INTRADAY_REFRESH_HOURS: int            # Cron B cadence in hours (e.g. 4)
    FUNDAMENTAL_REFRESH_LOOKBACK_DAYS: int             # recent macro window the live refresh re-pulls (e.g. 60)
    FUNDAMENTAL_STALENESS_ALERT_HOURS: float           # warn if newest spine refresh older than this (e.g. 26)

    # Feature builder (Pre-M7 Step 4) — leakage firewall thresholds, zero hardcoding
    FEATURE_STALENESS_MONTHLY_DAYS: int    # monthly series → NaN if newest release older than this (e.g. 75)
    FEATURE_STALENESS_DAILY_DAYS: int      # daily series → NaN if newest release older than this (e.g. 10)
    VIX_CHANGE_DAYS: int                   # calendar-day lookback for vix_change_5d (e.g. 5)
    WTI_CHANGE_DAYS: int                   # calendar-day lookback for wti_change_20d (e.g. 20)
    YIELD_DIFFERENTIAL_CHANGE_MONTHS: str  # comma list of months-back for yield-diff change features (e.g. "1,3")

    # Backtester (M7) — triple-barrier simulator, walk-forward CV, promotion gate.
    # Zero hardcoding: every simulator/gate number is env-driven (no inline constants).
    SIGNAL_MAX_HOLD_BARS: int                    # time exit, in trading-TF bars (e.g. 10 ≈ 40h on H4)
    BACKTEST_TRAILING_LOCK_PCT: float            # partial fires at this fraction of entry→target (0.5 = 1R when target is 2R)
    BACKTEST_TRAILING_DISTANCE_ATR_MULT: float   # trail distance = N × ATR14-at-signal (frozen for the trade; e.g. 1.0)
    BACKTEST_EMBARGO_BARS: int                    # walk-forward IS→OOS embargo, in trading-TF bars (e.g. 14) — consumed by Part B runner
    BACKTEST_PROMOTION_PROFIT_FACTOR_MIN: float  # promotion gate: min profit factor per OOS fold (e.g. 1.3)
    BACKTEST_PROMOTION_EXPECTANCY_MIN: float     # promotion gate: min expectancy (mean R) per OOS fold (e.g. 0.15)
    BACKTEST_PROMOTION_MAX_DD_MAX: float         # promotion gate: max drawdown ceiling per OOS fold (e.g. 0.25)
    BACKTEST_PROMOTION_MIN_OOS_TRADES: int       # promotion gate: min trade count per OOS fold (e.g. 50)
    BACKTEST_RESPECT_WEEKLY_CAP: bool            # false → backtest ignores MAX_TRADES_PER_WEEK (a LIVE throttle) so the training set captures every rule-valid trade
    EXECUTION_WINDOW_START: datetime             # Bid/Ask execution window start — NEVER inferred from min(timestamp); Part B enforces
    EXECUTION_WINDOW_END: datetime               # Bid/Ask execution window end — NEVER inferred from max(timestamp); Part B enforces
    MID_WARMUP_START: datetime                   # Mid-price warmup window start — indicators need history left of the execution window (Cycle-2 leftward extension); NEVER inferred from min(timestamp)
    BACKTEST_FOLD_BOUNDS: str                    # comma list T1,T2,T3 as fractions of the execution window for the 3-fold expanding walk-forward (Part B consumes)

    # Anthropic (Part 4 — server-side wiki ingestion)
    ANTHROPIC_API_KEY: Optional[str] = None
    WIKI_INGEST_ENABLED: bool
    WIKI_INGEST_MODEL: str

    # ML training threshold
    MIN_ML_TRAINING_CONFIDENCE: float

    # ML training pipeline (S1) — XGBoost signal filter. Zero hardcoding: every
    # label rule / split fraction / hyperparameter is env-driven (no inline consts).
    ML_LABEL_THRESHOLD_R: float          # win label if rr_actual >= this (e.g. 1.0)
    ML_SEED: int                         # global RNG seed for reproducibility (e.g. 42)
    ML_VALIDATION_FRACTION: float        # chronological tail of each IS fold held out for early stop + threshold (e.g. 0.2)
    ML_MIN_KEEP_FRACTION: float          # filter must keep >= this fraction of OOS trades (anti-gaming, e.g. 0.2)
    ML_MAX_DEPTH: int                    # XGBoost max_depth (e.g. 4)
    ML_N_ESTIMATORS: int                 # XGBoost n_estimators upper bound (e.g. 400)
    ML_LEARNING_RATE: float              # XGBoost learning_rate (e.g. 0.05)
    ML_EARLY_STOPPING_ROUNDS: int        # early-stopping patience on the validation tail (e.g. 50)

    # ML inference + shadow mode (M8-Shadow). Shadow mode scores real live signals
    # with the serialized S1 artifact and LOGS the decision — it places no orders.
    SHADOW_MODE_ENABLED: bool            # true → live path scores each signal and writes a stage='shadow' row
    # HARD SAFETY FLAG. false → the platform may never send an order to a broker,
    # whatever any signal/model says. Flipping this to true IS the sandbox
    # transition and is a deliberate, reviewed step: it must NEVER be flipped while
    # a NOT_PROMOTED artifact is loaded (inference refuses to load one when this is
    # true — see app/services/ml/inference.py safety guards).
    ORDER_PLACEMENT_ENABLED: bool
    ML_MODEL_PATH: str                   # joblib artifact path; relative paths resolve against backend/
    ML_DECISION_THRESHOLD: float         # P(win) >= this → 'take', else 'skip' (artifact's deployment_threshold)
    # Escape hatch for shadowing a candidate that FAILED the walk-forward promotion
    # gate. Inference refuses to load a .NOT_PROMOTED artifact unless this is true
    # AND ORDER_PLACEMENT_ENABLED is false — that pair is what makes observing a
    # rejected model safe by construction.
    ML_ALLOW_UNPROMOTED_MODEL: bool
    # CHALLENGER artifacts — comma-separated paths, empty for none. Each is scored on
    # every live signal alongside the champion (ML_MODEL_PATH) and its verdict is
    # written to `model_decisions` with is_authoritative=False.
    #
    # A challenger NEVER influences behaviour: it does not touch trades.ml_*, does not
    # change take/skip, and cannot cause an order. That is the entire point — running
    # a candidate beside the incumbent on identical live signals is the only way to
    # answer "where did they disagree, and who was right?", which is what promotion
    # turns on, and it has to be free of consequence to be worth doing.
    #
    # Defaulted to "" rather than required, because a deployment with no challenger is
    # the normal case and every other ML_ setting being mandatory would make adding
    # this a breaking config change for an optional feature.
    ML_CHALLENGER_MODEL_PATHS: str = ""
    # Shadow outcome resolver (Phase 3) cadence, in hours between runs. The resolver
    # is idempotent and cheap when the queue is empty, so a low value only costs a
    # bounded query; it never re-touches an already-resolved row.
    SHADOW_RESOLVER_INTERVAL_HOURS: int
    # Max NaN model-core features tolerated on a live shadow row before it is WARNED
    # about. The M7 training corpus has FEATURE_KEYS_MODEL 100% populated, so a live
    # row with holes is not comparable to it. The row is still recorded (a hole in the
    # corpus is worse than a flagged row) and the count is stored in
    # signal_reasoning['shadow']['nan_model_features'] so such rows are excludable.
    # 0 = warn on ANY NaN model feature.
    SHADOW_MAX_NAN_MODEL_FEATURES: int
    # Minimum fraction of a trading-TF bucket's minutes that must be present as
    # COMPLETE Bid+Ask M1 bars for that bucket to count toward the resolver's
    # observability test. Guards against a bucket with a handful of minutes counting
    # the same as a full one and letting the simulator walk past the true SL/TP touch.
    SHADOW_MIN_BUCKET_M1_DENSITY: float

    # Execution venue (M8-Sandbox). THREE states, not a boolean, because "no
    # orders", "practice orders" and "real money" are genuinely different and a
    # boolean can only express two.
    #   observe  — record decisions, never contact a broker  (current)
    #   sandbox  — real order tickets against the PRACTICE account
    #   live     — real money; NOT IMPLEMENTED and blocked in brokers/base.py
    # Orders additionally require ORDER_PLACEMENT_ENABLED=true, and `sandbox` is
    # cross-checked against OANDA_BASE_URL containing 'fxpractice' — a mode saying
    # "practice" while the URL points at live is the one misconfiguration that
    # silently risks real money.
    EXECUTION_MODE: str

    # Excursion / path recording (Phase A — attribution layer)
    # Record where each trade TRAVELLED (MFE/MAE + per-bar R), not merely where it
    # ended. Off by default in the sense that a caller must ask for it; when a
    # caller does, these two govern it.
    PATH_RECORDING_ENABLED: bool
    # How many signal-TF bars to keep walking AFTER the trade closed, to answer
    # "should we have held longer?" — a question the closed trade's own record can
    # never answer. Those bars are tagged beyond_exit and are excluded from mfe_r/
    # mae_r, which describe the trade that actually happened. 0 disables the
    # lookahead entirely (path stops at the exit bar).
    PATH_EXTENDED_BARS: int

    # Circuit breaker + alerts (Part 4)
    MIN_WIN_RATE_ALERT: float
    MAX_DRAWDOWN_ALERT: float
    CIRCUIT_BREAKER_LOSSES: int
    CIRCUIT_BREAKER_WINDOW: int
    ALERT_DELIVERY: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
