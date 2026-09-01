"""
Pre-M7 Step 3 — one-shot backfill of the Tier-1 news spine into news_calendar_events
(2yr back + forward window). Thin wrapper around
app.services.news_calendar.spine.refresh_spine (shared with the scheduled live refresh).

Requires FUNDAMENTAL_DATA_PROVIDER=fred + FUNDAMENTAL_DATA_API_KEY. Run from backend/:
    python scripts/ingest_news_calendar.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings
from app.database import SessionLocal
from app.services.news_calendar.spine import refresh_spine


def ingest() -> None:
    s = get_settings()
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=s.CANDLE_LOOKBACK_DAYS + 180)).strftime("%Y-%m-%d")
    end = (now + timedelta(days=s.NEWS_REFRESH_FORWARD_DAYS)).strftime("%Y-%m-%d")
    print(f"[ingest_news_calendar] window {start} -> {end}", flush=True)

    db = SessionLocal()
    try:
        summary = refresh_spine(db, s, start, end)
        for title, n in summary.items():
            print(f"  {title}: {n}", flush=True)
        total = sum(v for v in summary.values() if v > 0)
        print(f"[ingest_news_calendar] done — {total} events upserted", flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    ingest()
