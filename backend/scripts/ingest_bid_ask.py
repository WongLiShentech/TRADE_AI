"""
Pre-M7 Step 1.6 — Candle standardization (GAP-11).

Re-ingests candles for ALL active instruments over ONE fixed [A, B] window
(A = B - CANDLE_LOOKBACK_DAYS, B = run time) so every instrument / timeframe /
price_type shares identical coverage. Full-coverage (not incremental) and
idempotent (ON CONFLICT DO NOTHING), so it is safe to re-run.

Matrix — the minimal-but-complete set the backtest consumes:
    H4 -> Mid (signal/indicators), Bid, Ask (entry & spread realism)
    D  -> Mid (trend filter + indicators)
    M1 -> Bid, Ask (barrier checks + microstructure; spread = Ask - Bid)
M1 Mid is intentionally skipped (not consumed; would add ~5M unused rows).

Run from backend/:
    python scripts/ingest_bid_ask.py

Long-running (~hours, ~10.5M rows). Designed to run in the background; prints
progress per (instrument, granularity, price_type) and never aborts the whole
job on a single window failure.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Ensure backend/ is on sys.path so `app.*` imports resolve when run as a script.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.brokers.router import get_broker_router
from app.config import get_settings
from app.database import SessionLocal
from app.models.instrument import Instrument
from app.services import candle_service

# granularity -> price types to ingest
MATRIX: dict[str, list[str]] = {
    "H4": ["M", "B", "A"],
    "D": ["M"],
    "M1": ["B", "A"],
}

# Sub-window size per granularity (days) — bounds per-request memory. get_candles
# paginates internally; this just caps how much we hold + commit at a time.
SUBWINDOW_DAYS: dict[str, int] = {"H4": 365, "D": 365, "M1": 30}


def ingest() -> None:
    settings = get_settings()
    router = get_broker_router()
    db = SessionLocal()
    try:
        b = datetime.now(timezone.utc)
        a = b - timedelta(days=settings.CANDLE_LOOKBACK_DAYS)
        active = db.query(Instrument).filter_by(is_active=True).order_by(Instrument.symbol).all()
        print(
            f"[ingest_bid_ask] window {a.isoformat()} -> {b.isoformat()} "
            f"({settings.CANDLE_LOOKBACK_DAYS}d) across {len(active)} active pairs",
            flush=True,
        )

        grand_total = 0
        for inst in active:
            for granularity, price_types in MATRIX.items():
                win = timedelta(days=SUBWINDOW_DAYS[granularity])
                for price_type in price_types:
                    inserted = 0
                    sub_start = a
                    while sub_start < b:
                        sub_end = min(sub_start + win, b)
                        try:
                            inserted += candle_service.fetch_and_store_window(
                                inst.symbol, granularity, sub_start, sub_end,
                                db, settings, router, price_type=price_type,
                            )
                        except Exception as exc:  # noqa: BLE001 — never abort the whole job
                            print(
                                f"  ! {inst.symbol} {granularity} {price_type} "
                                f"[{sub_start.date()}..{sub_end.date()}] error: {exc}",
                                flush=True,
                            )
                        sub_start = sub_end
                        time.sleep(0.05)  # be polite to the OANDA API
                    grand_total += inserted
                    print(f"  {inst.symbol} {granularity} {price_type}: {inserted} rows", flush=True)

        print(f"[ingest_bid_ask] done — {grand_total} rows inserted", flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    ingest()
