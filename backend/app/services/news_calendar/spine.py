"""
Tier-1 news-spine refresh — generate deterministic Tier-1 events from the registry
(app/domain/news_spine.py) + FRED release dates and UPSERT into news_calendar_events.

Shared by the one-shot backfill (`scripts/ingest_news_calendar.py`) and the scheduled
live refresh (scheduler). Idempotent: keyed on `event_key` (stable hash of
source+currency+title+date, NOT timestamp), so an intra-day reschedule overwrites the
same row. UTC conversion is DST-correct via zoneinfo.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.config import Settings
from app.domain.news_spine import SPINE, SpineEvent
from app.models.instrument import Instrument
from app.models.news_calendar_event import NewsCalendarEvent
from app.services.fundamental.factory import get_fundamental_data


def _active_currencies(db: Session) -> set[str]:
    """Currencies used by at least one ACTIVE instrument, discovered at runtime from
    the instruments table (symbols like 'EUR_USD'). The spine gates on this so a bank's
    events ingest only when a live pair uses that currency — adding a pair auto-includes
    its bank's events with zero code change (see feedback-modular-extensible)."""
    out: set[str] = set()
    for (symbol,) in db.query(Instrument.symbol).filter(Instrument.is_active.is_(True)).all():
        out.update(symbol.split("_"))
    return out


def _event_key(source: str, currency: str, title: str, date_iso: str) -> str:
    norm = title.strip().lower()
    return hashlib.sha1(f"{source}|{currency}|{norm}|{date_iso}".encode()).hexdigest()


def _to_utc_naive(date_iso: str, hhmm: str, tz: str) -> datetime:
    h, m = (int(x) for x in hhmm.split(":"))
    local = datetime(int(date_iso[:4]), int(date_iso[5:7]), int(date_iso[8:10]), h, m, tzinfo=ZoneInfo(tz))
    return local.astimezone(timezone.utc).replace(tzinfo=None)


def _dates_for(ev: SpineEvent, provider, start: str, end: str) -> list[str]:
    if ev.fred_release_id is not None:
        return provider.get_release_dates(ev.fred_release_id, start=start, end=end)
    return [d for d in ev.static_dates if start <= d <= end]


def refresh_spine(db: Session, settings: Settings, start: str, end: str) -> dict:
    """Upsert all spine events in [start, end] (ISO dates). Returns per-event counts."""
    provider = get_fundamental_data(settings)
    fetched = datetime.now(timezone.utc).replace(tzinfo=None)
    active = _active_currencies(db)
    summary: dict[str, int] = {}

    for ev in SPINE:
        if ev.currency not in active:
            summary[ev.title] = -2  # skipped: no active pair uses this currency
            continue
        source = "fred_release" if ev.fred_release_id is not None else "cb_spine"
        try:
            dates = _dates_for(ev, provider, start, end)
        except Exception as exc:  # noqa: BLE001 — never abort the whole refresh on one event
            summary[ev.title] = -1
            db.rollback()
            continue
        if not dates:
            summary[ev.title] = 0
            continue

        rows = []
        for d in dates:
            ts = (
                datetime(int(d[:4]), int(d[5:7]), int(d[8:10]), 0, 0)
                if ev.all_day
                else _to_utc_naive(d, ev.announce_local, ev.tz)
            )
            rows.append({
                "event_key": _event_key(source, ev.currency, ev.title, d),
                "timestamp": ts,
                "currency": ev.currency,
                "impact": ev.impact,
                "title": ev.title,
                "is_tier1": ev.is_tier1,
                "all_day": ev.all_day,
                "source": source,
                "fetched_at": fetched,
            })

        stmt = pg_insert(NewsCalendarEvent).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["event_key"],
            set_={
                "timestamp": stmt.excluded.timestamp,
                "impact": stmt.excluded.impact,
                "title": stmt.excluded.title,
                "is_tier1": stmt.excluded.is_tier1,
                "all_day": stmt.excluded.all_day,
                "fetched_at": stmt.excluded.fetched_at,
            },
        )
        db.execute(stmt)
        db.commit()
        summary[ev.title] = len(rows)

    return summary


def spine_staleness_hours(db: Session) -> float | None:
    """Hours since the spine was last refreshed (max fetched_at). None if empty.
    Used by the M8 gate / a watchdog to fail-closed when news data is stale."""
    row = (
        db.query(NewsCalendarEvent.fetched_at)
        .filter(NewsCalendarEvent.source.in_(("cb_spine", "fred_release")))
        .order_by(NewsCalendarEvent.fetched_at.desc())
        .first()
    )
    if row is None or row[0] is None:
        return None
    return (datetime.now(timezone.utc).replace(tzinfo=None) - row[0]).total_seconds() / 3600.0
