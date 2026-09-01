from datetime import datetime

from sqlalchemy import DateTime, Float, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class MacroData(Base):
    """
    Bitemporal-CAPABLE macro / fundamental time-series store (yields, policy rates,
    VIX, and later CPI/unemployment). The schema supports multiple vintages per
    (series, ref_period) — one row per `release_time` — via UNIQUE(series, ref_period,
    release_time).

    Two ingestion paths share this table (branched in refresh_macro on
    MacroSeries.revised):
      - NON-revised market data (yields/policy/VIX): ONE row per (series, ref_period)
        with a SYNTHETIC `release_time` = ref_period + publish_lag.
      - REVISED indicators (CPI/core-CPI/unemployment/retail-sales/HICP): FULL
        BITEMPORAL — MULTIPLE rows per (series, ref_period), one per vintage, each
        with its TRUE `release_time` from FRED output_type=1 realtime_start (first
        release + one row per revision). No schema change — these just add rows.

    feature_builder reads point-in-time with ONE rule for both backtest and live:
    rows with `release_time <= signal_time` FIRST, then the latest `ref_period`
    among survivors, then argmax(release_time) within it. Gate leakage on
    `release_time`, NEVER on `ref_period` (publish lag matters).
    """

    __tablename__ = "macro_data"
    __table_args__ = (
        UniqueConstraint(
            "series", "ref_period", "release_time",
            name="uq_macro_series_period_vintage",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    series: Mapped[str] = mapped_column(String, nullable=False, index=True)   # internal name, e.g. "US_10Y_YIELD"
    ref_period: Mapped[str] = mapped_column(String, nullable=False)           # period the value describes, ISO date e.g. "2025-04-01"
    release_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)  # when THIS vintage became public
    value: Mapped[float] = mapped_column(Float, nullable=False)               # value as of that vintage
