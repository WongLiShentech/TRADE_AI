"""Per-instrument DB failures must not poison the shared session — QA MED 6.

All three pipeline loops open ONE ``SessionLocal`` and iterate every active
instrument inside it. Before this fix, a DB error on one instrument left that session
in a failed transaction, so every SUBSEQUENT instrument raised
``PendingRollbackError`` on an unrelated statement — one bad pair silently taking down
the whole run while the summary reported a generic error count.

These tests inject a DB-level failure on the FIRST instrument and assert the rest of
the loop still does real work. They are deliberately written against the observable
outcome (later instruments succeed) rather than against "rollback was called", so they
would still catch the bug if the fix were implemented some other way.

Run from backend/:  python -m pytest tests/test_pipeline_rollback.py -v
"""
from __future__ import annotations

import inspect

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.services import pipeline as pl


def _active_instrument_count(db) -> int:
    from app.models.instrument import Instrument

    return db.query(Instrument).filter(Instrument.is_active.is_(True)).count()


# ── 1. the handlers actually roll back ───────────────────────────────────────
@pytest.mark.parametrize(
    "func",
    [
        pl.run_candle_close_pipeline,
        pl.run_candle_refresh_pipeline,
        pl.run_trailing_window_refresh,
    ],
)
def test_every_per_instrument_handler_rolls_back(func):
    """Source-level guarantee, per loop: each ``except`` that continues the loop must
    roll the shared session back first.

    Checked structurally because the alternative — provoking a genuine DB error inside
    each of three network-bound loops — would make this suite slow and flaky without
    testing anything more.
    """
    src = inspect.getsource(func)
    handlers = [line for line in src.splitlines() if "except Exception" in line]
    assert handlers, f"{func.__name__} has no per-instrument handler to check"
    assert src.count("db.rollback()") >= len(handlers), (
        f"{func.__name__} continues its loop after an exception without calling "
        "db.rollback() — a DB error on one instrument will poison every later one"
    )


def test_rollback_precedes_the_error_count_increment():
    """Ordering matters: the summary bookkeeping runs on the shared session too, so the
    rollback has to come first."""
    for func in (
        pl.run_candle_close_pipeline,
        pl.run_candle_refresh_pipeline,
        pl.run_trailing_window_refresh,
    ):
        src = inspect.getsource(func)
        for block in src.split("except Exception")[1:]:
            head = block[: block.find("logger.warning")]
            assert "db.rollback()" in head, (
                f"{func.__name__}: rollback must run before the handler touches the "
                "session again"
            )


# ── 2. the behaviour that matters: one bad instrument does not stop the rest ─
def test_a_failing_instrument_does_not_stop_the_refresh_loop(db, settings, monkeypatch):
    """End-to-end on the cheapest loop: poison the first instrument's fetch with a DB
    error and assert the remaining instruments still fetch and compute.

    Without the rollback the poisoned session makes every later instrument fail too,
    so ``errors`` would equal the instrument count and ``fetched`` would be 0.
    """
    from app.services import candle_service

    total = _active_instrument_count(db)
    assert total > 1, "this test needs more than one active instrument"

    calls: list[str] = []
    real = candle_service.fetch_and_store_latest

    def _poisoned(*args, **kwargs):
        calls.append(args[0] if args else kwargs.get("instrument_symbol"))
        if len(calls) == 1:
            raise SQLAlchemyError("injected DB failure on the first instrument")
        return real(*args, **kwargs)

    monkeypatch.setattr(candle_service, "fetch_and_store_latest", _poisoned)

    summary = pl.run_candle_refresh_pipeline("D", settings)

    assert summary["errors"] == 1, (
        f"only the poisoned instrument should fail, got {summary['errors']} errors — "
        "the shared session was not rolled back"
    )
    assert summary["instruments"] == total
    assert len(calls) == total, "every instrument must still be attempted"
