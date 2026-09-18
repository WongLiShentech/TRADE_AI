"""Give a model an opinion on shadow signals it wasn't running for yet.

Why this exists
----------------
A model only produces a ``model_decisions`` row for signals scored while it was
actually registered to score them. A model added later (a new challenger, a
promoted successor) has zero opinion on every signal recorded before it joined, even
though the feature vector for each of those signals is sitting right there in
``trades.signal_reasoning`` — the same JSON the live path reads, and the same one
``ml.dataset`` reads to build a training corpus.

Scoring is a pure function: model + feature vector -> probability, every time,
with nothing about "when" folded in. So "what would this model have said on
5 August?" has an exact, deterministic answer, and this script computes it and
writes it down — using the SAME ``ml.inference.score`` chokepoint the live shadow
path calls, so a backfilled row is produced by literally the same code as a live
one, not a reimplementation of it that could quietly disagree.

What this is NOT
-----------------
Not training. The model file is read-only; nothing about it changes. Not a
verdict that ever governed anything: every row this script writes carries
``is_authoritative=False`` unconditionally, regardless of whether the target
model is today's champion — backfilling must never fabricate an alternate history
where a model governed decisions before it was ever deployed. The row this
script inserts says "here is what this model would have scored this signal",
nothing more.

Idempotent by construction: ``model_decisions`` carries a UNIQUE(trade_id,
model_id) index, and this script both pre-filters against it and inserts with
``ON CONFLICT DO NOTHING``, so re-running (on a schedule, after adding a new
challenger) only ever fills the gap that has newly appeared.

Scoped per strategy, for the same reason the live path is: a model's labels derive
from ``rr_actual``, which depends on the exit rule, so a model is valid ONLY for the
strategy whose outcomes taught it. Backfilling one across every strategy's rows
would manufacture cross-strategy verdicts indistinguishable from real evidence.

Extensible by construction: targets are resolved from the registry exactly the way
the shadow recorder resolves them (``ml.registry.models_for_strategy`` over every
shadow/live strategy), so a model registered tomorrow is backfillable with this same
script and no code change. ``--model-id`` narrows to one target.

Run from ``backend/``:
    python scripts/backfill_model_decisions.py --dry-run
    python scripts/backfill_model_decisions.py --dry-run --model-id s1_model_v3_schema2_06fb46aca05f
    python scripts/backfill_model_decisions.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config import get_settings
from app.database import SessionLocal
from app.models.model_decision import ModelDecision
from app.models.trade import Trade
from app.services.ml import inference as inf
from app.services.ml.registry import RegisteredModel, models_for_strategy
from app.services.shadow.recorder import _nan_model_features  # same helper the live path uses
from app.services.strategy_registry import active_strategies

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("backfill_model_decisions")

# Every row of every stage the recorder writes carries a stored feature vector.
# 'shadow' and 'sandbox' both qualify; 'backtest' rows are the training corpus
# itself and out of scope here.
_ELIGIBLE_STAGES = ("shadow", "sandbox")

# Batched commits, same reasoning as runner.py's trade batching: a single
# multi-hundred-row transaction is needless lock time on a box this small, and a
# mid-run failure should not cost work already durably committed.
_BATCH_SIZE = 200


def _targets(db, settings, only_model_id: str | None) -> list[tuple[int, RegisteredModel]]:
    """(strategy_id, model) pairs to backfill, resolved from the registry.

    Scoped per strategy on purpose. A model is valid only for the strategy whose
    outcomes taught it, so backfilling one across every strategy's rows would
    manufacture exactly the cross-strategy verdicts the live path was fixed to stop
    producing — and they would be indistinguishable from real evidence afterwards.
    """
    out: list[tuple[int, RegisteredModel]] = []
    for strategy in active_strategies(db):
        models = models_for_strategy(db, settings, strategy.id)
        for m in models.all:
            if only_model_id is None or m.model_id == only_model_id:
                out.append((strategy.id, m))

    if only_model_id is not None and not out:
        raise SystemExit(
            f"--model-id {only_model_id!r} is not registered as champion or challenger "
            f"for any shadow/live strategy. Set its `models.status` and `strategy_id` "
            f"first — this script only backfills models the registry actually runs."
        )
    return out


def _missing_trade_ids(db, model_id: str, strategy_id: int) -> list[int]:
    """This strategy's rows that have a stored feature vector and no verdict from this model."""
    already = select(ModelDecision.trade_id).where(ModelDecision.model_id == model_id)
    rows = db.execute(
        select(Trade.id)
        .where(Trade.stage.in_(_ELIGIBLE_STAGES))
        .where(Trade.strategy_id == strategy_id)
        .where(Trade.signal_reasoning.isnot(None))
        .where(Trade.id.notin_(already))
        .order_by(Trade.id.asc())
    ).scalars().all()
    return list(rows)


def backfill_one_model(db, strategy_id: int, model: RegisteredModel, *, dry_run: bool) -> dict:
    """Score every eligible gap for one model on one strategy. Returns a summary dict."""
    loaded = model.loaded
    trade_ids = _missing_trade_ids(db, model.model_id, strategy_id)
    summary = {
        "model_id": model.model_id, "strategy_id": strategy_id,
        "candidates": len(trade_ids), "written": 0, "failed": 0,
    }
    if not trade_ids:
        return summary

    threshold = model.threshold
    logger.info(
        "%s on strategy %s: %d signal(s) with no opinion on file (its own threshold=%.4f)",
        model.model_id, strategy_id, len(trade_ids), threshold,
    )
    if dry_run:
        return summary

    pending: list[dict] = []
    for start in range(0, len(trade_ids), _BATCH_SIZE):
        chunk_ids = trade_ids[start:start + _BATCH_SIZE]
        trades = db.execute(select(Trade).where(Trade.id.in_(chunk_ids))).scalars().all()
        for trade in trades:
            features = trade.signal_reasoning or {}
            try:
                prob = inf.score(loaded, features)
            except Exception as exc:  # noqa: BLE001 — one bad row must not abort the run
                logger.warning("trade %s: scoring failed (%s) — skipped", trade.id, exc)
                summary["failed"] += 1
                continue
            if prob != prob:  # NaN — same contract as inference.decide(): never invent a decision
                logger.warning("trade %s: model returned NaN probability — skipped", trade.id)
                summary["failed"] += 1
                continue
            decision = inf.DECISION_TAKE if prob >= threshold else inf.DECISION_SKIP
            nan_count = len(_nan_model_features(features))
            pending.append({
                "trade_id": trade.id,
                "model_id": model.model_id,
                "probability": prob,
                "decision": decision,
                "threshold": threshold,
                "nan_features": nan_count,
                # Never True: a backfilled row states what the model WOULD have
                # said, never what actually governed the signal. See module docstring.
                "is_authoritative": False,
            })

        if pending:
            stmt = pg_insert(ModelDecision).values(pending)
            stmt = stmt.on_conflict_do_nothing(index_elements=["trade_id", "model_id"])
            result = db.execute(stmt)
            db.commit()
            summary["written"] += result.rowcount or 0
            pending = []
        logger.info("%s: %d/%d scored", model.model_id, min(start + _BATCH_SIZE, len(trade_ids)), len(trade_ids))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="report counts only, write nothing")
    parser.add_argument(
        "--model-id", default=None,
        help="backfill only this model_id (must be registered as champion or challenger for a shadow/live strategy); default: every registered model",
    )
    args = parser.parse_args()

    settings = get_settings()
    db = SessionLocal()
    try:
        targets = _targets(db, settings, args.model_id)
        if not targets:
            logger.error("no models registered for any shadow/live strategy — nothing to do")
            return

        results = [
            backfill_one_model(db, sid, m, dry_run=args.dry_run) for sid, m in targets
        ]

        print()
        print(f"{'model_id':<40} {'strat':>5} {'candidates':>10} {'written':>8} {'failed':>7}")
        for r in results:
            print(
                f"{r['model_id']:<40} {r['strategy_id']:>5} {r['candidates']:>10} "
                f"{r['written']:>8} {r['failed']:>7}"
            )
        if args.dry_run:
            print("\n(--dry-run: nothing written)")
    finally:
        db.close()


if __name__ == "__main__":
    main()
