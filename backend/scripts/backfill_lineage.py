"""Populate run and dataset lineage for history that predates it.

Idempotent and machine-agnostic: safe to run twice, and safe on both the local
research database (two strategies, two runs) and the server (one strategy, one run).
Everything it writes is DERIVED FROM RECORDED EVIDENCE. Where the evidence is gone it
writes nothing and says so, rather than inventing a plausible value.

Run from ``backend/``::

    python scripts/backfill_lineage.py --dry-run   # report only
    python scripts/backfill_lineage.py            # apply

Inferred, never observed — and gated
------------------------------------
No process recorded which run produced which trade; that is the gap being closed.
So every ``run_id`` written here is an INFERENCE, and it is only written when three
independent checks agree:

1. exactly one run resolves to that strategy (two candidates ⇒ the rows could belong
   to either, and guessing is what this layer exists to prevent);
2. the run's own ``fold_breakdown['totals']['trades_simulated']`` equals the row count
   — the run's contemporaneous record of its own output;
3. every row's ``opened_at`` falls inside the window the run actually walked.

The gates are applied identically to every strategy, so the column never mixes
certainty levels: a stamp means "three checks agreed", uniformly, whether the run
happened in July or this afternoon. Where they disagree, nothing is written and the
disagreement is printed.

What it will NOT do
-------------------
It will not invent a run for rows whose parent cannot be identified. A synthetic
placeholder was considered and rejected on three grounds: it would have to fabricate
fourteen NOT NULL measurements including ``passed`` — a promotion verdict nobody ever
reached; ``GET /backtester/runs`` would serve it as genuine; and, decisively, it would
consume a ``backtest_runs_id_seq`` value, permanently changing ``_durable_n_trials``
and therefore the deflated Sharpe applied to every future backtest. A bookkeeping
decision must not move a statistical result.

Such rows lose little: ``strategy_id`` already attributes them, and that is the
attribute labels actually depend on.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import func, select, update

from app.config import get_settings
from app.database import SessionLocal
from app.models import BacktestRun, Strategy, Trade, TrainingDataset
from app.services.feature_builder import FEATURE_KEYS_MODEL
from app.services.ml.dataset import load_dataset
from app.services.ml.fingerprint import feature_keys_hash
from scripts.backfill_attribution import _ENGINE, _IDENTITY_PARAMS, _params_hash

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("lineage")

_STAGE = "backtest"


# ── step 1: which strategy did each run actually execute? ────────────────────
def _derive_strategy(run: BacktestRun, by_hash: dict[str, Strategy]) -> Strategy | None:
    """Identify a run's strategy from the params IT RECORDED, not from its label.

    ``backtest_runs.strategy`` cannot be trusted: it was written from a module
    constant, so run 7 executed rule_based_v2_fixed and recorded rule_based_v1. The
    run's own ``fold_breakdown['params']`` snapshot is the authority — recomputing the
    identity hash over it is the same mechanism ``backfill_attribution.step_register``
    uses to mint a strategy in the first place, so the two cannot disagree.
    """
    params_all = (run.fold_breakdown or {}).get("params") or {}
    if not params_all:
        return None
    params = {k: params_all[k] for k in _IDENTITY_PARAMS if k in params_all}
    if len(params) != len(_IDENTITY_PARAMS):
        missing = [k for k in _IDENTITY_PARAMS if k not in params_all]
        logger.warning("  run %s: params snapshot missing %s — cannot derive", run.id, missing)
        return None
    return by_hash.get(_params_hash(_ENGINE, params))


def step_runs(db, dry: bool) -> tuple[int, dict[int, int]]:
    """Returns (changes, {run_id: strategy_id}).

    The mapping is returned rather than re-read from the DB by step 2, because under
    ``--dry-run`` nothing is written — and a dry run that cannot see its own step 1
    would report "0 rows would be stamped" and then stamp thousands for real. A dry
    run that under-reports is worse than none: it is a rehearsal of a different play.
    """
    logger.info("[1] attributing backtest_runs to strategies")
    by_hash = {s.params_hash: s for s in db.execute(select(Strategy)).scalars()}
    changed = 0
    mapping: dict[int, int] = {}

    for run in db.execute(select(BacktestRun).order_by(BacktestRun.id)).scalars():
        derived = _derive_strategy(run, by_hash)
        if derived is None:
            logger.warning("  run %s: strategy NOT derivable — left NULL", run.id)
            continue

        mapping[run.id] = derived.id
        if run.strategy_id == derived.id and run.strategy == derived.name:
            logger.info("  run %s: already correct (%s)", run.id, derived.name)
            continue

        if run.strategy != derived.name:
            # A transcription error, not a re-measurement — but record that a
            # correction happened rather than quietly overwriting the evidence.
            logger.info(
                "  run %s: CORRECTING strategy %r -> %r (id=%s)",
                run.id, run.strategy, derived.name, derived.id,
            )
            if not dry:
                fb = dict(run.fold_breakdown or {})
                corrections = list(fb.get("_corrections") or [])
                corrections.append({
                    "field": "strategy",
                    "was": run.strategy,
                    "now": derived.name,
                    "reason": "runner._STRATEGY was a module constant and could not "
                              "know which strategy the run executed",
                    "at": datetime.now(timezone.utc).isoformat(),
                })
                fb["_corrections"] = corrections
                fb["strategy"] = derived.name
                fb["strategy_id"] = derived.id
                run.fold_breakdown = fb
        else:
            logger.info("  run %s: linking to strategy id=%s (%s)", run.id, derived.id, derived.name)

        if not dry:
            run.strategy = derived.name
            run.strategy_id = derived.id
        changed += 1

    if not dry:
        db.commit()
    return changed, mapping


# ── step 2: stamp run_id where — and only where — it is provable ─────────────
def step_trades(db, dry: bool, mapping: dict[int, int]) -> int:
    """Attribute trades to the run that produced them, gated on three checks.

    Every stamp here is INFERRED from recorded evidence, never observed — the process
    that wrote the rows did not record which run it belonged to, which is the gap this
    backfill exists to close. The gates are the evidence, and they are applied
    identically to every strategy so that no column mixes certainty levels: a stamp
    means "three independent checks agreed", uniformly.
    """
    logger.info("[2] attributing trades to runs")
    changed = 0

    runs = list(db.execute(select(BacktestRun).order_by(BacktestRun.id)).scalars())
    # Prefer the mapping step 1 derived (it is authoritative and survives --dry-run)
    # and fall back to whatever is already persisted.
    resolved = {r.id: mapping.get(r.id, r.strategy_id) for r in runs}

    for strategy_id in sorted({v for v in resolved.values() if v is not None}):
        candidates = [r for r in runs if resolved.get(r.id) == strategy_id]
        rows = db.execute(
            select(func.count(Trade.id), func.min(Trade.opened_at), func.max(Trade.opened_at))
            .where(Trade.stage == _STAGE, Trade.strategy_id == strategy_id)
        ).one()
        n_rows, t_min, t_max = rows
        if not n_rows:
            continue

        unstamped = db.execute(
            select(func.count(Trade.id)).where(
                Trade.stage == _STAGE,
                Trade.strategy_id == strategy_id,
                Trade.run_id.is_(None),
            )
        ).scalar_one()
        if not unstamped:
            logger.info("  strategy %s: all %d rows already stamped", strategy_id, n_rows)
            continue

        # GATE 1 — exactly one candidate run. Two runs of one strategy means the rows
        # could belong to either, and guessing is precisely what this layer exists to
        # stop.
        if len(candidates) != 1:
            logger.warning(
                "  strategy %s: %d candidate runs (%s) — cannot attribute %d rows "
                "unambiguously, left NULL",
                strategy_id, len(candidates), [r.id for r in candidates], unstamped,
            )
            continue
        run = candidates[0]

        # GATE 2 — the run's own record of how many trades it simulated must match.
        recorded = ((run.fold_breakdown or {}).get("totals") or {}).get("trades_simulated")
        if recorded != n_rows:
            logger.warning(
                "  strategy %s: run %s recorded trades_simulated=%s but %d rows exist "
                "— refusing to attribute",
                strategy_id, run.id, recorded, n_rows,
            )
            continue

        # GATE 3 — every row must fall inside the window the run actually walked.
        if t_min < run.in_sample_start or t_max > run.oos_end:
            logger.warning(
                "  strategy %s: rows span %s..%s, outside run %s window %s..%s "
                "— refusing to attribute",
                strategy_id, t_min, t_max, run.id, run.in_sample_start, run.oos_end,
            )
            continue

        logger.info("  strategy %s: stamping run_id=%s on %d rows", strategy_id, run.id, unstamped)
        if not dry:
            db.execute(
                update(Trade)
                .where(
                    Trade.stage == _STAGE,
                    Trade.strategy_id == strategy_id,
                    Trade.run_id.is_(None),
                )
                .values(run_id=run.id)
            )
        changed += unstamped

    if not dry:
        db.commit()

    # Report — never silently attribute — what is left unresolvable. Skipped under
    # --dry-run, where nothing was written and this would contradict the lines above.
    if not dry:
        orphans = db.execute(
            select(Trade.strategy_id, func.count(Trade.id))
            .where(Trade.stage == _STAGE, Trade.run_id.is_(None))
            .group_by(Trade.strategy_id)
        ).all()
        for sid, n in orphans:
            logger.info(
                "  strategy %s: %d rows remain run_id=NULL — no run passed all three "
                "gates for them (see module docstring)", sid, n,
            )
    return changed


# ── step 3: register the corpora as datasets ─────────────────────────────────
def step_datasets(db, dry: bool) -> int:
    logger.info("[3] registering datasets")
    settings = get_settings()
    fk_hash = feature_keys_hash(list(FEATURE_KEYS_MODEL))
    changed = 0

    strategy_ids = [
        sid for (sid,) in db.execute(
            select(Trade.strategy_id)
            .where(Trade.stage == _STAGE, Trade.strategy_id.is_not(None))
            .group_by(Trade.strategy_id)
            .order_by(Trade.strategy_id)
        ).all()
    ]

    for sid in strategy_ids:
        ds = load_dataset(db, settings.ML_LABEL_THRESHOLD_R, strategy_id=sid)
        fp = ds.fingerprint()
        existing = db.execute(
            select(TrainingDataset).where(TrainingDataset.fingerprint == fp)
        ).scalar_one_or_none()
        if existing is not None:
            logger.info("  strategy %s: dataset already registered id=%s fp=%s",
                        sid, existing.id, fp[:12])
            continue

        # A single run only if the whole corpus came from one; otherwise NULL, which
        # is the honest description of a corpus spanning several executions.
        run_ids = {
            r for (r,) in db.execute(
                select(Trade.run_id)
                .where(Trade.stage == _STAGE, Trade.strategy_id == sid)
                .group_by(Trade.run_id)
            ).all()
        }
        run_id = run_ids.pop() if len(run_ids) == 1 else None

        strategy = db.get(Strategy, sid)
        span = db.execute(
            select(func.min(Trade.opened_at), func.max(Trade.opened_at))
            .where(Trade.stage == _STAGE, Trade.strategy_id == sid)
        ).one()

        logger.info("  strategy %s: registering dataset fp=%s rows=%d run_id=%s",
                    sid, fp[:12], len(ds), run_id)
        if not dry:
            db.add(TrainingDataset(
                fingerprint=fp,
                stage=_STAGE,
                strategy_id=sid,
                run_id=run_id,
                n_rows=len(ds),
                n_positive=int(ds.y.sum()),
                span_start=span[0],
                span_end=span[1],
                label_threshold_r=float(settings.ML_LABEL_THRESHOLD_R),
                feature_schema_version=int(ds.feature_schema_version),
                feature_keys_hash=fk_hash,
                description=(
                    f"stage={_STAGE}, strategy="
                    f"{strategy.name if strategy else sid}"
                    + (f", run={run_id}" if run_id else ", spanning multiple/unknown runs")
                ),
            ))
        changed += 1

    if not dry:
        db.commit()
    return changed


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    dry = "--dry-run" in sys.argv
    if dry:
        logger.info("=== DRY RUN — nothing will be written ===")

    db = SessionLocal()
    try:
        n_runs, mapping = step_runs(db, dry)
        total = n_runs + step_trades(db, dry, mapping) + step_datasets(db, dry)
        if dry:
            db.rollback()
            logger.info("\n[dry-run] %d change(s) would be applied", total)
        else:
            logger.info("\n[backfill_lineage] committed %d change(s)", total)
    finally:
        db.close()


if __name__ == "__main__":
    main()
