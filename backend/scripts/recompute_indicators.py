"""
One-shot FULL RECOMPUTE of the indicators table for all active instruments.

Fixes BUG-NEW-1 (ragged left edge) + F2 (stale right edge), both caused by the
incremental short-circuit in indicator_service.compute_and_store. Mid candles are
already uniform across pairs (Step 1.6), so a clean full recompute from the earliest
Mid candle naturally yields identical per-pair windows.

Indicators are Mid-ONLY and computed on H4/D1 only (never M1). Uses the
full_recompute=True override which delete-then-inserts cleanly on the natural key.

Run from backend/:
    python scripts/recompute_indicators.py
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings
from app.database import SessionLocal
from app.models.instrument import Instrument
from app.services.indicator_service import compute_and_store

GRANULARITIES = ["H4", "D"]  # D1 = 'D'; M1 intentionally excluded


def recompute() -> None:
    s = get_settings()
    db = SessionLocal()
    try:
        instruments = (
            db.query(Instrument).filter_by(is_active=True).order_by(Instrument.id).all()
        )
        print(
            f"[recompute_indicators] {len(instruments)} active instruments "
            f"× {GRANULARITIES} (full_recompute=True)",
            flush=True,
        )
        total = 0
        for inst in instruments:
            for gran in GRANULARITIES:
                n = compute_and_store(
                    instrument_symbol=inst.symbol,
                    granularity=gran,
                    db=db,
                    settings=s,
                    full_recompute=True,
                )
                total += n
                print(f"  {inst.symbol:<10} {gran:<3} -> {n} rows", flush=True)
        print(f"[recompute_indicators] done — {total} indicator rows written", flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    recompute()
