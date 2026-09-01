"""
Cycle-2 Phase A — LEFTWARD candle extension (2 years → 5 years).

Extends the existing candle history to the LEFT only. The 2024-06→2026-06 execution
window and the 2024-01→ Mid warmup already in the DB are NOT re-fetched: for each
(instrument, granularity, price_type) this fetches ONLY the missing left stretch
[target_start, seam_end), where:

    seam_end   = the CURRENT earliest stored timestamp for that key (captured ONCE,
                 persisted to the progress file so partial resumes never move it), and
    target_start:
        Mid (M)      -> settings.MID_WARMUP_START        (e.g. 2021-01-01)
        Bid/Ask (B,A)-> settings.EXECUTION_WINDOW_START  (e.g. 2021-06-01)

Matrix (mirrors scripts/ingest_bid_ask.py — the minimal set the backtest consumes):
    H4 -> M (signal/indicators), B, A (entry & spread realism)
    D  -> M (trend filter + indicators)
    M1 -> B, A (barrier checks + microstructure; spread = Ask - Bid)

Idempotent + resumable: fetch_and_store_window inserts ON CONFLICT DO NOTHING, so the
single seam candle at seam_end de-dupes and any re-run skips already-stored rows. A JSON
progress file (output/extend_candles_progress.json) records seam_end + next_start + done
per key, so an interrupted M1 job resumes exactly where it stopped with no gap. Zero
hardcoding: the whole window comes from env (MID_WARMUP_START, EXECUTION_WINDOW_START).

Long-running (~hours; M1 is millions of new rows). Prints per-chunk progress and never
aborts the whole job on a single window failure.

Run from backend/:
    python scripts/extend_candles_5y.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure backend/ is on sys.path so `app.*` imports resolve when run as a script.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import func

from app.brokers.router import get_broker_router
from app.config import Settings, get_settings
from app.database import SessionLocal
from app.models.candle import Candle
from app.models.instrument import Instrument
from app.services import candle_service

# granularity -> price types to extend (identical matrix to ingest_bid_ask.py)
MATRIX: dict[str, list[str]] = {
    "H4": ["M", "B", "A"],
    "D": ["M"],
    "M1": ["B", "A"],
}

# Sub-window size per granularity (days) — bounds per-request memory / commit size.
SUBWINDOW_DAYS: dict[str, int] = {"H4": 365, "D": 365, "M1": 30}

# Progress file lives in the repo output folder (project rule: all output → output/).
PROGRESS_PATH = BACKEND_ROOT.parent / "output" / "extend_candles_progress.json"


def _target_start(price_type: str, settings: Settings) -> datetime:
    """Left boundary for a price_type: Mid uses the warmup start, Bid/Ask the
    execution-window start. Both are env-driven — never inferred from the data."""
    raw = settings.MID_WARMUP_START if price_type == "M" else settings.EXECUTION_WINDOW_START
    return raw.replace(tzinfo=None) if raw.tzinfo is not None else raw


def _load_progress() -> dict:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {}


def _save_progress(progress: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2, default=str))


def _existing_min(db, instrument_id: int, granularity: str, price_type: str) -> datetime | None:
    return (
        db.query(func.min(Candle.timestamp))
        .filter_by(instrument_id=instrument_id, granularity=granularity, price_type=price_type)
        .scalar()
    )


def extend() -> None:
    settings = get_settings()
    router = get_broker_router()
    db = SessionLocal()
    progress = _load_progress()
    try:
        active = (
            db.query(Instrument).filter_by(is_active=True).order_by(Instrument.symbol).all()
        )
        print(
            f"[extend_candles_5y] leftward extension across {len(active)} active pairs; "
            f"mid_start={_target_start('M', settings).date()} "
            f"exec_start={_target_start('B', settings).date()}",
            flush=True,
        )

        grand_total = 0
        for inst in active:
            for granularity, price_types in MATRIX.items():
                win = timedelta(days=SUBWINDOW_DAYS[granularity])
                for price_type in price_types:
                    key = f"{inst.symbol}|{granularity}|{price_type}"
                    start = _target_start(price_type, settings)

                    entry = progress.get(key)
                    if entry is None:
                        seam_min = _existing_min(db, inst.id, granularity, price_type)
                        # seam_end = current earliest stored candle (fetch strictly left of it;
                        # the boundary candle itself de-dupes). If none stored, fall back to
                        # the execution-window end so a brand-new key still gets a full window.
                        seam_end = seam_min or settings.EXECUTION_WINDOW_END.replace(tzinfo=None)
                        entry = {
                            "start": start.isoformat(),
                            "seam_end": seam_end.isoformat(),
                            "next_start": start.isoformat(),
                            "done": False,
                            "inserted": 0,
                        }
                        progress[key] = entry
                        _save_progress(progress)

                    if entry["done"]:
                        print(f"  = {key}: already done ({entry['inserted']} rows)", flush=True)
                        grand_total += entry["inserted"]
                        continue

                    seam_end = datetime.fromisoformat(entry["seam_end"])
                    sub_start = datetime.fromisoformat(entry["next_start"])
                    if sub_start >= seam_end:
                        entry["done"] = True
                        _save_progress(progress)
                        print(f"  = {key}: nothing to fetch (seam already reached)", flush=True)
                        continue

                    print(
                        f"  > {key}: {sub_start.date()} -> {seam_end.date()} "
                        f"(sub={SUBWINDOW_DAYS[granularity]}d)",
                        flush=True,
                    )
                    while sub_start < seam_end:
                        sub_end = min(sub_start + win, seam_end)
                        try:
                            n = candle_service.fetch_and_store_window(
                                inst.symbol, granularity, sub_start, sub_end,
                                db, settings, router, price_type=price_type,
                            )
                            entry["inserted"] += n
                            grand_total += n
                        except Exception as exc:  # noqa: BLE001 — never abort the whole job
                            db.rollback()
                            print(
                                f"    ! {key} [{sub_start.date()}..{sub_end.date()}] error: {exc}",
                                flush=True,
                            )
                        entry["next_start"] = sub_end.isoformat()
                        _save_progress(progress)
                        sub_start = sub_end
                        time.sleep(0.05)  # be polite to the OANDA API

                    entry["done"] = True
                    _save_progress(progress)
                    print(f"  {key}: +{entry['inserted']} rows", flush=True)

        print(f"[extend_candles_5y] done — {grand_total} rows inserted (cumulative)", flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    extend()
