"""
One-shot BACKFILL of macro_data from FRED. Thin wrapper around
app.services.fundamental.refresh.refresh_macro (shared with the live Cron A/B jobs);
picks up new MACRO_SERIES registry entries automatically.

Idempotent: ON CONFLICT (series, ref_period, release_time) DO NOTHING — a revision
lands as a new release_time row, never an in-place overwrite (point-in-time firewall).

Two release_time paths (branched per series on MacroSeries.revised):
  - NON-revised market data (yields/policy/VIX): SYNTHETIC release_time = ref_period
    + publish_lag_days (45 monthly / 1 daily), FRED output_type=1 (latest).
  - REVISED indicators (CPI/core-CPI/unemployment/retail-sales/EU-HICP/UK-CPI):
    FULL BITEMPORAL — EVERY vintage from FRED output_type=1 over the full realtime
    window, each row's TRUE release_time = realtime_start (first release + one row per
    later revision). Genuine vintages — leakage-safe with no synthetic lag.

Requires FUNDAMENTAL_DATA_PROVIDER=fred + FUNDAMENTAL_DATA_API_KEY. Run from backend/:
    python scripts/ingest_macro.py
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings
from app.database import SessionLocal
from app.services.fundamental.refresh import refresh_macro


def ingest() -> None:
    s = get_settings()
    print(
        f"[ingest_macro] backfill provider={s.FUNDAMENTAL_DATA_PROVIDER} "
        f"lookback={s.MACRO_LOOKBACK_DAYS}d",
        flush=True,
    )
    db = SessionLocal()
    try:
        summary = refresh_macro(db, s, s.MACRO_LOOKBACK_DAYS)
        for name, n in summary.items():
            mark = "!" if n < 0 else ("-" if n == 0 else " ")
            print(f"  {mark} {name}: {n}", flush=True)
        total = sum(v for v in summary.values() if v > 0)
        failed = [k for k, v in summary.items() if v < 0]
        print(f"[ingest_macro] done — {total} rows; failures={failed or 'none'}", flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    ingest()
