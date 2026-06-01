"""
Trim active instrument universe to the 10 majors+minors used in M5.

Idempotent — running it multiple times keeps the same 10 active and the rest inactive.
Universe list is hardcoded here ONLY because this is a one-shot ops script, not
business logic. The application itself reads is_active from the DB at runtime.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure backend/ is on sys.path so `app.*` imports resolve when run as a script.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.database import SessionLocal
from app.models.instrument import Instrument

UNIVERSE = {
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "AUD_USD",
    "NZD_USD",
    "USD_CAD",
    "USD_CHF",
    "EUR_GBP",
    "EUR_JPY",
    "GBP_JPY",
}


def trim() -> dict:
    db = SessionLocal()
    try:
        all_inst = db.query(Instrument).all()
        active_now, deactivated_now = 0, 0
        missing: list[str] = []
        for inst in all_inst:
            should_be_active = inst.symbol in UNIVERSE
            if inst.is_active != should_be_active:
                inst.is_active = should_be_active
            if should_be_active:
                active_now += 1
            else:
                deactivated_now += 1
        for symbol in UNIVERSE:
            if not any(i.symbol == symbol for i in all_inst):
                missing.append(symbol)
        db.commit()
        return {
            "active": active_now,
            "inactive": deactivated_now,
            "missing_from_db": missing,
        }
    finally:
        db.close()


if __name__ == "__main__":
    result = trim()
    print(result)
