from datetime import datetime

from sqlalchemy import Boolean, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class NewsCalendarEvent(Base):
    """
    Economic-calendar event (schedule-only in Phase 1: time/currency/impact/title).

    Mono-temporal but MUTABLE: exactly one row per logical event, always reflecting
    the latest known schedule. Identity is `event_key` (a stable hash of
    source + currency + normalized title + reference date) — NOT the timestamp,
    because the timestamp is the attribute that mutates on a reschedule. Refreshes
    UPSERT on `event_key`, so a moved event overwrites its own `timestamp` with no
    duplicate/orphan row.

    Distinct from `trades.news_events` (a per-trade JSON audit snapshot). This is
    the queryable calendar that feature_builder reads point-in-time and the M8
    Tier-1 hard gate checks against.
    """

    __tablename__ = "news_calendar_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_key: Mapped[str] = mapped_column(String, nullable=False, unique=True)   # sha1(source, currency, title_norm, ref_date)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)  # UTC fire time — MUTABLE
    currency: Mapped[str] = mapped_column(String, nullable=False, index=True)     # USD | EUR | ...
    impact: Mapped[str] = mapped_column(String, nullable=False)                   # low | medium | high
    title: Mapped[str] = mapped_column(String, nullable=False)
    is_tier1: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    all_day: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)  # tentative/null-time → block whole day (e.g. BoJ)
    source: Mapped[str] = mapped_column(String, nullable=False, index=True)       # cb_spine | fred_release | forexfactory | finnhub
    fetched_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)        # last refresh time (staleness audit)
