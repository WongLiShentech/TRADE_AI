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

    # Live price stream (M3)
    STREAM_RECONNECT_DELAY_SECONDS: int
    STREAM_MAX_RECONNECT_RETRIES: int
    STREAM_HEARTBEAT_TIMEOUT_SECONDS: int

    # Indicator engine (M4)
    SWING_LOOKBACK_PERIODS: int  # candles either side for swing high/low detection

    # Vault (Part 1) — server-side wiki ingestion writes journal here
    VAULT_PATH: str

    # News calendar (Part 2 — used by M8 trade classification only)
    NEWS_CALENDAR_PROVIDER: str
    NEWS_CALENDAR_WINDOW_HOURS: float
    SPREAD_ANOMALY_MULTIPLIER: float

    # News sentiment (Phase 1B scaffold — empty until needed)
    NEWS_SENTIMENT_PROVIDER: Optional[str] = None
    NEWS_SENTIMENT_API_KEY: Optional[str] = None
    NEWS_SENTIMENT_LOOKBACK_MINUTES: Optional[int] = None

    # Fundamental data (Phase 2 scaffold — empty until needed)
    FUNDAMENTAL_DATA_PROVIDER: Optional[str] = None
    FUNDAMENTAL_DATA_API_KEY: Optional[str] = None

    # Anthropic (Part 4 — server-side wiki ingestion)
    ANTHROPIC_API_KEY: Optional[str] = None
    WIKI_INGEST_ENABLED: bool
    WIKI_INGEST_MODEL: str

    # ML training threshold
    MIN_ML_TRAINING_CONFIDENCE: float

    # Circuit breaker + alerts (Part 4)
    MIN_WIN_RATE_ALERT: float
    MAX_DRAWDOWN_ALERT: float
    CIRCUIT_BREAKER_LOSSES: int
    CIRCUIT_BREAKER_WINDOW: int
    ALERT_DELIVERY: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
