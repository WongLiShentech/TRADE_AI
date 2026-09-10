"""Backfill the Phase A attribution layer onto rows that predate it.

The 20260902_attribution migration added three tables and four columns, all empty.
This script fills them for existing rows. It is READ-ONLY on every pre-existing
value: ``rr_actual``, ``outcome``, ``exit_reason``, ``exit_price`` and
``closed_at`` are never written here (the one exception is step 3, which resolves
rows that are still PENDING and therefore have no outcome to overwrite).

Five steps, each independently skippable and each resumable
----------------------------------------------------------
1. REGISTER the current strategy in ``strategies``. Identity is a digest over the
   engine name plus the signal-affecting parameters, read from the backtest run's
   own recorded params — not from the live .env, which could since have drifted.
2. ATTRIBUTE every existing trade to it. Today that is a single strategy, so every
   row gets the same id; the value is the audit trail, and the fact that a future
   parameter change produces a DIFFERENT hash and therefore cannot silently pool
   with these rows.
3. RESOLVE stranded shadow rows. Rows recorded on the dev machine have their M1
   here and nowhere else; left alone, the server's closure cap will eventually
   force-resolve them against minute data it does not have, producing confident
   garbage. Resolving them here is the only chance to do it honestly.
4. BACKFILL ``model_decisions`` from the existing ``trades.ml_*`` columns, marked
   authoritative — they ARE what governed those rows.
5. WALK each resolved trade again to record MFE/MAE and its per-bar path.

Verification (step 5)
---------------------
Every re-simulated ``rr_actual`` must equal the stored one. The simulator is
deterministic given the same M1, and path recording is passive, so any difference
means the walk is no longer reproducing history — at which point the paths are
describing a trade that did not happen. A single mismatch is reported; a mismatch
RATE above ``_MAX_MISMATCH_RATE`` aborts, because that is systematic.

Usage
-----
    python scripts/backfill_attribution.py                 # all steps
    python scripts/backfill_attribution.py --steps 1,2     # just registration
    python scripts/backfill_attribution.py --limit 100     # smoke test
    python scripts/backfill_attribution.py --dry-run       # report, write nothing

Resumability: step 5 skips trades that already have ``mfe_r`` set, so an
interrupted run continues where it stopped. Re-running is safe throughout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime
import math
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure backend/ is on sys.path so `app.*` imports resolve when run as a script.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import func

from app.config import get_settings
from app.database import SessionLocal
from app.models import BacktestRun, ModelDecision, Strategy, Trade, TradePath
from app.services.backtester.simulator import simulate
from app.services.shadow.resolver import resolve_pending
from app.services.strategy_registry import ENGINE, IDENTITY_PARAMS, params_hash

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("backfill")

# Parameters that define WHAT THE STRATEGY IS. Deliberately excludes gate
# thresholds, fold bounds and window bounds: those govern how a backtest is
# EVALUATED, not what signals it produces, and folding them into the identity
# would make an unchanged strategy look new every time a gate was retuned.
_IDENTITY_PARAMS = IDENTITY_PARAMS

_STRATEGY_NAME = "rule_based_v1"
_ENGINE = ENGINE
_ATR_KEY = "atr14"
_COMMIT_EVERY = 100
# Above this fraction of rr_actual mismatches, stop: isolated differences can come
# from a genuine M1 gap that has since been backfilled, but a systematic rate means
# the walk itself has changed and every path it produces is describing fiction.
_MAX_MISMATCH_RATE = 0.01
_RR_TOLERANCE = 1e-6


# Re-exported so existing callers (backfill_lineage, run_strategy2_backtest) keep
# working, but the DEFINITION now lives in app/services/strategy_registry.py — the
# live path needs it too, and two copies of an identity rule eventually disagree.
_params_hash = params_hash


def _atr_of(trade: Trade) -> Optional[float]:
    """ATR(14) at signal time. Mirrors resolver._atr_of, tolerating either layout.

    Shadow rows store the feature dict flat; some backtest rows nest it under
    ``features``. Returns None rather than a default — inventing an ATR invents the
    trailing-stop distance, and therefore invents an exit.
    """
    reasoning = trade.signal_reasoning or {}
    value = reasoning.get(_ATR_KEY)
    if value is None and isinstance(reasoning.get("features"), dict):
        value = reasoning["features"].get(_ATR_KEY)
    if value is None:
        return None
    try:
        atr = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(atr) or math.isinf(atr) or atr <= 0.0:
        return None
    return atr


# ── step 1 ───────────────────────────────────────────────────────────────────
def step_register(db, dry_run: bool) -> Optional[Strategy]:
    """Create (or find) the strategy row for the configuration that produced history.

    Params come from the BACKTEST RUN's own record, not the live .env: the run is
    the authority on what produced those 7,074 rows, and reading the env would
    silently attribute them to whatever the config happens to say today.
    """
    run = db.query(BacktestRun).order_by(BacktestRun.run_at.desc()).first()
    if run is None or not run.fold_breakdown:
        logger.error("no backtest_runs row with fold_breakdown — cannot derive params")
        return None

    all_params = (run.fold_breakdown or {}).get("params", {})
    params = {k: all_params[k] for k in _IDENTITY_PARAMS if k in all_params}
    missing = [k for k in _IDENTITY_PARAMS if k not in all_params]
    if missing:
        logger.warning("  identity params absent from the run record: %s", missing)

    phash = _params_hash(_ENGINE, params)
    existing = db.query(Strategy).filter_by(params_hash=phash).first()
    if existing:
        logger.info("  strategy already registered: id=%s %s", existing.id, existing.name)
        return existing

    logger.info("  registering %s  engine=%s  hash=%s", _STRATEGY_NAME, _ENGINE, phash)
    for k in _IDENTITY_PARAMS:
        logger.info("      %-45s %s", k, params.get(k, "(absent)"))
    if dry_run:
        return None

    strategy = Strategy(
        name=_STRATEGY_NAME,
        engine=_ENGINE,
        params_hash=phash,
        params=params,
        status="shadow",
        notes=(
            "H4 pullback-in-trend, 5-condition confluence, 3-of-5 required. "
            "M7 baseline PF 1.41; failed the walk-forward gate on fold-1 drawdown "
            "(29% vs 25% ceiling). Filtered live by the s1_xgb_v2 artifact."
        ),
    )
    db.add(strategy)
    db.commit()
    logger.info("  -> strategy id=%s", strategy.id)
    return strategy


# Rows opened at or after this instant are attributed BY THE WRITER, not by this
# backfill. Anything still NULL after it indicates a writer that forgot to stamp —
# which should be fixed at the writer, never papered over here.
_ATTRIBUTION_CUTOFF = datetime(2026, 9, 8)


# ── step 2 ───────────────────────────────────────────────────────────────────
def step_attribute(db, strategy: Strategy, dry_run: bool) -> int:
    """Point every unattributed trade at the strategy that produced it.

    Bounded by ``opened_at`` on purpose. This step was written when exactly one
    strategy existed, so "unattributed" and "strategy 1" meant the same thing.
    They no longer do: the backtest runner now stamps ``strategy_id`` as it writes,
    and any row this step still finds NULL is either historical or a bug. Claiming
    every NULL row for strategy 1 would silently mislabel a second strategy's
    corpus — the exact failure this attribution layer exists to prevent.
    """
    q = db.query(Trade).filter(
        Trade.strategy_id.is_(None),
        Trade.opened_at < _ATTRIBUTION_CUTOFF,
    )
    n = q.count()
    logger.info("  %d unattributed trades -> strategy id=%s", n, strategy.id)
    if dry_run or n == 0:
        return n
    q.update({Trade.strategy_id: strategy.id}, synchronize_session=False)
    db.commit()
    return n


# ── step 3 ───────────────────────────────────────────────────────────────────
def step_resolve_stranded(db, settings, dry_run: bool) -> dict:
    """Resolve pending shadow rows while their M1 still exists on THIS machine.

    The dev box holds five years of M1; the server holds only what it has gathered
    since deployment. A row recorded here and left pending will eventually trip the
    server's closure cap and be force-resolved against minute data that does not
    exist there — flagged ambiguous, but still a fabricated label. This is the only
    place it can be resolved honestly.
    """
    pending = (
        db.query(func.count(Trade.id))
        .filter(Trade.stage == "shadow", Trade.rr_actual.is_(None))
        .scalar()
    )
    logger.info("  %d pending shadow rows", pending)
    if dry_run or not pending:
        return {"pending": pending}
    summary = resolve_pending(db, settings)
    logger.info("  %s", summary)
    return summary


# ── step 4 ───────────────────────────────────────────────────────────────────
def step_model_decisions(db, settings, dry_run: bool) -> int:
    """Lift the one-opinion-per-trade ml_* columns into the many-opinions table.

    Marked ``is_authoritative`` because these ARE the verdicts that governed those
    rows. A challenger rescoring the same history later adds rows alongside; only
    one per trade may ever carry the flag.
    """
    rows = (
        db.query(Trade)
        .filter(Trade.ml_decision.isnot(None), Trade.ml_probability.isnot(None))
        .all()
    )
    existing = {
        (t, m)
        for t, m in db.query(ModelDecision.trade_id, ModelDecision.model_id).all()
    }
    todo = [t for t in rows if (t.id, t.ml_model_id) not in existing]
    logger.info("  %d scored trades, %d without a model_decisions row", len(rows), len(todo))
    if dry_run or not todo:
        return len(todo)

    threshold = float(settings.ML_DECISION_THRESHOLD)
    db.bulk_save_objects(
        [
            ModelDecision(
                trade_id=t.id,
                model_id=t.ml_model_id or "unknown",
                probability=float(t.ml_probability),
                decision=t.ml_decision,
                # The threshold in force when the row was written. Recorded per
                # decision because the same probability under a revised threshold
                # is a different verdict, and that difference is otherwise invisible.
                threshold=threshold,
                nan_features=((t.signal_reasoning or {}).get("shadow") or {}).get(
                    "nan_model_features"
                ),
                is_authoritative=True,
            )
            for t in todo
        ]
    )
    db.commit()
    return len(todo)


# ── step 5 ───────────────────────────────────────────────────────────────────
def step_paths(db, settings, limit: Optional[int], dry_run: bool) -> dict:
    """Re-walk resolved trades to record MFE/MAE and the per-bar path."""
    q = (
        db.query(Trade)
        .filter(Trade.rr_actual.isnot(None), Trade.mfe_r.is_(None))
        .order_by(Trade.id)
    )
    if limit:
        q = q.limit(limit)
    trades = q.all()
    total = len(trades)
    logger.info("  %d resolved trades without a path", total)
    if dry_run or not total:
        return {"total": total}

    stats = {"total": total, "done": 0, "no_atr": 0, "mismatch": 0, "truncated": 0, "failed": 0}
    mismatches: list[tuple[int, float, float]] = []
    started = time.time()

    for i, t in enumerate(trades, 1):
        atr = _atr_of(t)
        if atr is None:
            stats["no_atr"] += 1
            continue
        gran = ((t.signal_reasoning or {}).get("shadow") or {}).get("granularity") or "H4"
        try:
            r = simulate(
                db, settings, t.instrument_id, t.direction,
                float(t.entry_price), float(t.stop_price), float(t.tp_price),
                t.opened_at, atr, gran, record_path=True,
            )
        except Exception as exc:                                   # noqa: BLE001
            logger.warning("  trade %s failed: %s", t.id, exc)
            stats["failed"] += 1
            continue

        # THE verification. Deterministic simulator + passive recorder ⇒ identical
        # result. A difference means the walk is no longer reproducing history.
        if abs(float(r.rr_actual) - float(t.rr_actual)) > _RR_TOLERANCE:
            stats["mismatch"] += 1
            mismatches.append((t.id, float(t.rr_actual), float(r.rr_actual)))

        if r.path:
            db.query(TradePath).filter(TradePath.trade_id == t.id).delete(
                synchronize_session=False
            )
            db.bulk_save_objects(
                [
                    TradePath(
                        trade_id=t.id, bar=p.bar, r_close=p.r_close, r_best=p.r_best,
                        r_worst=p.r_worst, mfe_r=p.mfe_r, mae_r=p.mae_r,
                        beyond_exit=p.beyond_exit, degraded=p.degraded,
                    )
                    for p in r.path
                ]
            )
            t.mfe_r = None if r.mfe_r is None else float(r.mfe_r)
            t.mae_r = None if r.mae_r is None else float(r.mae_r)
            t.path_truncated = bool(r.path_truncated)
            if r.path_truncated:
                stats["truncated"] += 1
            stats["done"] += 1

        if i % _COMMIT_EVERY == 0:
            db.commit()
            rate = i / max(time.time() - started, 1e-9)
            eta = (total - i) / max(rate, 1e-9) / 60.0
            logger.info(
                "  %d/%d  (%.1f/s, ~%.0f min left)  mismatches=%d",
                i, total, rate, eta, stats["mismatch"],
            )
            # Abort on a SYSTEMATIC divergence, not an isolated one.
            if i >= 200 and stats["mismatch"] / i > _MAX_MISMATCH_RATE:
                db.commit()
                logger.error(
                    "ABORTING: %d/%d rr_actual mismatches (>%.1f%%). The walk is not "
                    "reproducing history; paths would describe trades that did not "
                    "happen. First few: %s",
                    stats["mismatch"], i, _MAX_MISMATCH_RATE * 100, mismatches[:5],
                )
                stats["aborted"] = True
                return stats

    db.commit()
    if mismatches:
        logger.warning("  %d rr_actual mismatches, e.g. %s", len(mismatches), mismatches[:5])
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", default="1,2,3,4,5", help="comma list of steps to run")
    ap.add_argument("--limit", type=int, default=None, help="cap step 5 (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args(argv)
    steps = {s.strip() for s in args.steps.split(",") if s.strip()}

    settings = get_settings()
    db = SessionLocal()
    try:
        strategy = None
        if "1" in steps:
            logger.info("STEP 1 — register strategy")
            strategy = step_register(db, args.dry_run)
        if "2" in steps:
            logger.info("STEP 2 — attribute trades")
            strategy = strategy or db.query(Strategy).filter_by(name=_STRATEGY_NAME).first()
            if strategy is None:
                logger.error("  no strategy registered — run step 1 first")
            else:
                step_attribute(db, strategy, args.dry_run)
        if "3" in steps:
            logger.info("STEP 3 — resolve stranded shadow rows")
            step_resolve_stranded(db, settings, args.dry_run)
        if "4" in steps:
            logger.info("STEP 4 — backfill model_decisions")
            step_model_decisions(db, settings, args.dry_run)
        if "5" in steps:
            logger.info("STEP 5 — walk trades for MFE/MAE + paths")
            stats = step_paths(db, settings, args.limit, args.dry_run)
            logger.info("  %s", stats)
            if stats.get("aborted"):
                return 1
        logger.info("DONE")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
