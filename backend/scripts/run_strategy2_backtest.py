"""Register `rule_based_v2_fixed` and backtest it — the pure-barrier exit variant.

Same C1-C5 entry as `rule_based_v1`, same stop, same 2R target, same 10-bar time
exit. The ONLY difference is that the partial-exit-plus-trailing-stop is disabled,
so a trade leaves by stop, target, or time and nothing else.

Mechanism: `BACKTEST_TRAILING_LOCK_PCT = 1.0` places the partial trigger exactly AT
the target, and `_run_core` tests `tp_touch` BEFORE `partial_touch`, so the target
always wins and the partial can never fire. A config value, not a code fork —
a fork would introduce a second variable and ruin the comparison.

`BACKTEST_TRAILING_LOCK_PCT` is already one of `_IDENTITY_PARAMS`, so this earns a
distinct `params_hash` on its own: identical configuration ⇒ identical id, any
difference ⇒ a new strategy. Nothing here special-cases it.

Run from ``backend/``:
    python scripts/run_strategy2_backtest.py [--dry-run] [--force]
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

# MUST precede get_settings(): the runner reads the lock pct off the Settings object,
# and Settings is built once and cached.
os.environ["BACKTEST_TRAILING_LOCK_PCT"] = "1.0"

from app.config import get_settings                                   # noqa: E402
from app.database import SessionLocal                                 # noqa: E402
from app.models import Strategy, Trade                                # noqa: E402
from app.services.backtester.runner import run_backtest               # noqa: E402
from scripts.backfill_attribution import _params_hash, _IDENTITY_PARAMS, _ENGINE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("strategy2")

_NAME = "rule_based_v2_fixed"
_DESCRIPTION = (
    "H4 pullback-in-trend across 10 FX majors. 5-condition confluence, 3-of-5 "
    "required. ATR(14) stop, 2R target, 10-bar time exit. NO partial exit and NO "
    "trailing stop — a trade leaves by stop, target, or time only. Entry logic is "
    "byte-identical to rule_based_v1; the exit is the sole difference."
)
_NOTES = (
    "Created 2026-09-08 from the exit-attribution experiment. Over 5,472 paired "
    "trades (identical entries under both exit rules) the pure barrier is worth "
    "+0.037R per trade, t=4.77, +203R total. The trail contributed EXACTLY 0.000R "
    "on 1,878 losing trades — a trade heading for its stop never reaches +0.5R, so "
    "the protection never arms — while costing 0.488R on every trade that reached "
    "target, because the partial banked 1.5R instead of 2.0R."
)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    dry = "--dry-run" in sys.argv
    force = "--force" in sys.argv

    settings = get_settings()
    assert float(settings.BACKTEST_TRAILING_LOCK_PCT) == 1.0, settings.BACKTEST_TRAILING_LOCK_PCT
    db = SessionLocal()
    try:
        # ── identity: copy strategy 1's params, change only the exit ──────────
        s1 = db.query(Strategy).filter_by(name="rule_based_v1").first()
        if s1 is None:
            logger.error("rule_based_v1 not registered — run backfill_attribution first")
            sys.exit(1)

        params = dict(s1.params)
        params["BACKTEST_TRAILING_LOCK_PCT"] = 1.0
        missing = [k for k in _IDENTITY_PARAMS if k not in params]
        if missing:
            logger.warning("identity params absent from strategy 1: %s", missing)
        phash = _params_hash(_ENGINE, params)

        if phash == s1.params_hash:
            logger.error("hash collides with strategy 1 — the exit change did not register")
            sys.exit(1)

        s2 = db.query(Strategy).filter_by(params_hash=phash).first()
        if s2 is None:
            logger.info("registering %s  hash=%s", _NAME, phash)
            for k in _IDENTITY_PARAMS:
                mark = "  <-- CHANGED" if params.get(k) != s1.params.get(k) else ""
                logger.info("    %-42s %s%s", k, params.get(k, "(absent)"), mark)
            if dry:
                logger.info("[dry-run] would register, then backtest — nothing written")
                return
            s2 = Strategy(
                name=_NAME, engine=_ENGINE, params_hash=phash, params=params,
                status="research", description=_DESCRIPTION, notes=_NOTES,
            )
            db.add(s2)
            db.commit()
            logger.info("  -> strategy id=%s", s2.id)
        else:
            logger.info("already registered: id=%s %s", s2.id, s2.name)

        existing = db.query(Trade).filter(Trade.strategy_id == s2.id).count()
        if existing and not force:
            logger.error(
                "strategy %s already has %d trades — re-running would DOUBLE the corpus. "
                "Pass --force to delete them and rebuild.", s2.id, existing,
            )
            sys.exit(1)
        if existing and force:
            # Delete by strategy_id, never by stage: a partial run must be removed
            # whole, and strategy 1's rows must not be touched.
            logger.info("--force: deleting %d existing rows for strategy %s", existing, s2.id)
            db.query(Trade).filter(Trade.strategy_id == s2.id).delete(synchronize_session=False)
            db.commit()

        if dry:
            logger.info("[dry-run] would backtest into strategy id=%s", s2.id)
            return

        logger.info("running pure-barrier backtest -> strategy_id=%s ...", s2.id)
        t0 = time.time()
        result = run_backtest(db, settings, strategy_id=s2.id)
        logger.info("backtest finished in %.0fs", time.time() - t0)

        n1 = db.query(Trade).filter(Trade.stage == "backtest", Trade.strategy_id == s1.id).count()
        n2 = db.query(Trade).filter(Trade.stage == "backtest", Trade.strategy_id == s2.id).count()
        orphan = db.query(Trade).filter(Trade.stage == "backtest", Trade.strategy_id.is_(None)).count()
        logger.info("CORPUS: strategy %s = %d rows | strategy %s = %d rows | unattributed = %d",
                    s1.id, n1, s2.id, n2, orphan)
        if orphan:
            logger.error("UNATTRIBUTED backtest rows exist — the runner failed to stamp them")
            sys.exit(1)

        out = Path(__file__).resolve().parents[2] / "output" / "strategy2_backtest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "strategy_id": s2.id, "params_hash": phash,
            "rows_strategy_1": n1, "rows_strategy_2": n2,
            "result": str(result),
        }, indent=2, default=str), encoding="utf-8")
        logger.info("wrote %s", out)
    finally:
        db.close()


if __name__ == "__main__":
    main()
