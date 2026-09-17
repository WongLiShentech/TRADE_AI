"""Give a model an opinion on shadow signals it wasn't running for yet.

Why this exists
----------------
A model only produces a ``model_decisions`` row for signals scored while it was
actually loaded — champion via ``ML_MODEL_PATH``, challenger via
``ML_CHALLENGER_MODEL_PATHS``. A model added later (a new challenger, a promoted
successor) has zero opinion on every signal recorded before it joined, even
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

Extensible by construction: targets are resolved from live config exactly the
way the shadow recorder resolves them (``inference.load_model`` +
``inference.load_challengers``), so a model added to ``ML_CHALLENGER_MODEL_PATHS``
tomorrow is backfillable with this same script and no code change. ``--model-id``
narrows to one target when you don't want all of them touched.

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
from app.services.shadow.recorder import _nan_model_features  # same helper the live path uses

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


def _candidate_models(settings, only_model_id: str | None) -> list[inf.LoadedModel]:
    """Resolve backfill targets exactly as the live recorder resolves scorers.

    Champion via ``load_model``, challengers via ``load_challengers`` — the same
    two calls ``pipeline.py`` makes per signal. This is deliberate: a model this
    script cannot reach here could never have been live-scoring in the first
    place, so the target set is bounded by what production actually runs.
    """
    models: list[inf.LoadedModel] = []
    try:
        models.append(inf.load_model(settings))
    except Exception as exc:  # noqa: BLE001 — report and continue with challengers
        logger.error("champion failed to load: %s", exc)
    models.extend(inf.load_challengers(settings))

    if only_model_id is None:
        return models
    matched = [m for m in models if m.model_id == only_model_id]
    if not matched:
        available = ", ".join(m.model_id for m in models) or "(none loaded)"
        raise SystemExit(
            f"--model-id {only_model_id!r} is not currently loaded as champion or "
            f"challenger. Loaded: {available}. Point ML_MODEL_PATH / "
            f"ML_CHALLENGER_MODEL_PATHS at it first — this script only backfills "
            f"models the live config actually runs, never an arbitrary file."
        )
    return matched


def _missing_trade_ids(db, model_id: str) -> list[int]:
    """Shadow/sandbox trades with a stored feature vector and no row yet for this model."""
    already = select(ModelDecision.trade_id).where(ModelDecision.model_id == model_id)
    rows = db.execute(
        select(Trade.id)
        .where(Trade.stage.in_(_ELIGIBLE_STAGES))
        .where(Trade.signal_reasoning.isnot(None))
        .where(Trade.id.notin_(already))
        .order_by(Trade.id.asc())
    ).scalars().all()
    return list(rows)


def backfill_one_model(db, loaded: inf.LoadedModel, *, dry_run: bool) -> dict:
    """Score every eligible gap for one model. Returns a summary dict."""
    trade_ids = _missing_trade_ids(db, loaded.model_id)
    summary = {"model_id": loaded.model_id, "candidates": len(trade_ids), "written": 0, "failed": 0}
    if not trade_ids:
        return summary

    threshold = loaded.deployment_threshold
    logger.info(
        "%s: %d signal(s) with no opinion on file (scoring at its own threshold=%.4f)",
        loaded.model_id, len(trade_ids), threshold,
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
                "model_id": loaded.model_id,
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
        logger.info("%s: %d/%d scored", loaded.model_id, min(start + _BATCH_SIZE, len(trade_ids)), len(trade_ids))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="report counts only, write nothing")
    parser.add_argument(
        "--model-id", default=None,
        help="backfill only this model_id (must be currently loaded as champion or challenger); default: all loaded models",
    )
    args = parser.parse_args()

    settings = get_settings()
    db = SessionLocal()
    try:
        models = _candidate_models(settings, args.model_id)
        if not models:
            logger.error("no models loaded (champion failed and no challengers configured) — nothing to do")
            return

        results = [backfill_one_model(db, m, dry_run=args.dry_run) for m in models]

        print()
        print(f"{'model_id':<40} {'candidates':>10} {'written':>8} {'failed':>7}")
        for r in results:
            print(f"{r['model_id']:<40} {r['candidates']:>10} {r['written']:>8} {r['failed']:>7}")
        if args.dry_run:
            print("\n(--dry-run: nothing written)")
    finally:
        db.close()


if __name__ == "__main__":
    main()
