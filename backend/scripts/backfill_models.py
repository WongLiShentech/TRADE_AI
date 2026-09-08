"""Populate the `models` registry from the artifact metadata already on disk.

Idempotent: re-running updates existing rows rather than duplicating them, so it
is safe to run on local and again on the server, or twice by accident.

Reads ``backend/models/*.metadata.json`` — the values are already recorded at
training time; this moves them somewhere the database can see. Nothing is
invented. Fields the metadata cannot answer (which strategy, lifecycle status,
the human description) are supplied here explicitly and stated as such.

Run from ``backend/``:
    python scripts/backfill_models.py            # apply
    python scripts/backfill_models.py --dry-run  # show what would change
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import select

from app.database import SessionLocal
from app.models import MLModel, Strategy, TrainingDataset

MODELS_DIR = BACKEND_ROOT / "models"

# The artifact metadata records how a model was BUILT, not where it sits in its
# lifecycle or which strategy's outcomes taught it. Those are curatorial facts,
# so they are stated here rather than guessed from the filename.
#
# Keys are the FULL artifact stem — `.NOT_PROMOTED` included where present — because
# that is the identifier `trades.ml_model_id` and `model_decisions.model_id` already
# carry. See the comment at the `model_id` assignment below.
CURATION = {
    "s1_xgb_v1_26882eac6c29": {
        "strategy_name": "rule_based_v1",
        "version": 1,
        "status": "retired",
        # n_trials was not recorded when v1 was trained and cannot be reconstructed:
        # the counter is derived from a live sequence, so reading it today would give
        # today's value, not the one v1's results were discounted by.
        "n_trials": None,
        "n_trials_source": None,
        "description": (
            "First XGBoost signal filter. 20 features (schema v1), 2,809 backtest "
            "trades over 2 years. Predates the promotion gate, so it was never "
            "formally evaluated against it. Superseded by v2."
        ),
    },
    "s1_xgb_v2_c430d22dd1ab.NOT_PROMOTED": {
        "strategy_name": "rule_based_v1",
        "version": 2,
        "status": "shadow",
        # Recovered from output/s1_experiment_log.json, which recorded the value used
        # at training time. The source string is not decoration: the counter is
        # derived from backtest_runs_id_seq GLOBALLY, so it differs by host (local
        # reads 7, the server 1) and counts looks at other strategies' data. A bare
        # integer would be uninterpretable in six months.
        "n_trials": 7,
        "n_trials_source": "backtest_runs_id_seq (global, local host) @ 2026-07-11",
        "description": (
            "Current shadow model. 19 features (schema v2), 7,074 backtest trades "
            "over 5 years. Failed the promotion gate on fold-1 expectancy "
            "(0.1473 vs the 0.1500 floor). Scores live signals in observe mode "
            "only; the .NOT_PROMOTED filename is enforced at load time."
        ),
    },
}

# git provenance is NOT curated for v1/v2. Both were trained before it was captured,
# and `git log --before <created_at>` would find the commit that EXISTED, not the one
# that ran — a fabrication dressed as evidence. NULL is the honest value.

STRATEGY_DESCRIPTIONS = {
    "rule_based_v1": (
        "H4 pullback-in-trend across 10 FX majors. 5-condition confluence, 3-of-5 "
        "required. ATR(14) stop, 2R target, partial exit + trailing stop at 0.5R, "
        "10-bar time exit."
    ),
}


def _dt(value):
    return datetime.fromisoformat(value) if value else None


def _resolve_dataset(db, meta: dict, strategy, model_id: str):
    """The dataset row this model trained on — but only when it can be PROVEN.

    Two paths, and both must prove rather than assume:

    * A model trained after fingerprinting existed carries its own
      ``dataset_fingerprint``. Match on it and the link is exact.
    * v1 and v2 predate it, so the only available evidence is that the strategy's
      corpus TODAY still has the row count and span the artifact recorded at training
      time. If the corpus has changed since, linking would attach the model to data it
      never saw — precisely the confusion the datasets table exists to end — so the
      link is refused and the reason printed.
    """
    fp = meta.get("dataset_fingerprint")
    if fp:
        row = db.execute(
            select(TrainingDataset).where(TrainingDataset.fingerprint == fp)
        ).scalar_one_or_none()
        if row is None:
            print(f"  NOTE  {model_id}: fingerprint {fp[:12]} has no dataset row")
        return row.id if row else None

    if strategy is None:
        return None
    span = meta.get("training_span", {})
    rows, start, end = span.get("n_rows"), _dt(span.get("start")), _dt(span.get("end"))
    candidate = db.execute(
        select(TrainingDataset).where(TrainingDataset.strategy_id == strategy.id)
    ).scalars().first()
    if candidate is None:
        return None
    if candidate.n_rows != rows:
        print(f"  NOTE  {model_id}: corpus now has {candidate.n_rows} rows, model "
              f"recorded {rows} — NOT linking (it trained on different data)")
        return None
    if start and candidate.span_start and abs((candidate.span_start - start).days) > 1:
        print(f"  NOTE  {model_id}: corpus span moved ({candidate.span_start} vs "
              f"{start}) — NOT linking")
        return None
    return candidate.id


def main() -> None:
    dry = "--dry-run" in sys.argv
    db = SessionLocal()
    try:
        strategies = {s.name: s for s in db.execute(select(Strategy)).scalars()}
        changed = 0

        for path in sorted(MODELS_DIR.glob("*.metadata.json")):
            # The id MUST be the FULL artifact stem, `.NOT_PROMOTED` included.
            #
            # It is tempting to strip the marker as "a gate verdict, not part of the
            # id" — that was the original reasoning here and it was wrong. The rest of
            # the system does not agree with it: `inference._model_id_for` removes only
            # the `.joblib` suffix, and THAT string is what `shadow.recorder` writes to
            # `trades.ml_model_id` and `model_decisions.model_id`. Stripping the marker
            # produced a registry key that joined to zero rows — a table describing
            # decisions it could not be linked to.
            model_id = path.name.removesuffix(".metadata.json")
            meta = json.loads(path.read_text(encoding="utf-8"))
            cur = CURATION.get(model_id)
            if cur is None:
                print(f"  SKIP {model_id} — no curation entry; add one to backfill it")
                continue

            span = meta.get("training_span", {})
            strategy = strategies.get(cur["strategy_name"])
            fields = dict(
                strategy_id=strategy.id if strategy else None,
                version=cur.get("version"),
                trained_on_stages="backtest",
                training_start=_dt(span.get("start")),
                training_end=_dt(span.get("end")),
                training_rows=span.get("n_rows"),
                decision_threshold=meta.get("deployment_threshold"),
                # None (not False) when the artifact predates the gate — "unknown"
                # is the honest value; False would invent a verdict.
                passed_gate=meta.get("promoted"),
                status=cur["status"],
                n_trials=cur.get("n_trials"),
                n_trials_source=cur.get("n_trials_source"),
                # Present only for models trained after provenance capture existed.
                git_commit=meta.get("git_commit"),
                git_dirty=meta.get("git_dirty"),
                dataset_id=_resolve_dataset(db, meta, strategy, model_id),
                description=cur["description"],
            )

            row = db.execute(
                select(MLModel).where(MLModel.model_id == model_id)
            ).scalar_one_or_none()
            if row is None:
                print(f"  INSERT {model_id}  rows={fields['training_rows']} "
                      f"span={span.get('start','?')[:10]}->{span.get('end','?')[:10]} "
                      f"gate={fields['passed_gate']} status={fields['status']}")
                if not dry:
                    db.add(MLModel(model_id=model_id, **fields))
                changed += 1
            else:
                diffs = [k for k, v in fields.items() if getattr(row, k) != v]
                if diffs:
                    print(f"  UPDATE {model_id} — {', '.join(diffs)}")
                    if not dry:
                        for k, v in fields.items():
                            setattr(row, k, v)
                    changed += 1
                else:
                    print(f"  OK     {model_id} — already current")

        for name, desc in STRATEGY_DESCRIPTIONS.items():
            s = strategies.get(name)
            if s is None:
                print(f"  SKIP strategy {name} — not registered")
            elif s.description != desc:
                print(f"  {'UPDATE' if s.description else 'SET'} strategy {name}.description")
                if not dry:
                    s.description = desc
                changed += 1
            else:
                print(f"  OK     strategy {name}.description — already current")

        if dry:
            db.rollback()
            print(f"\n[dry-run] {changed} change(s) would be applied — nothing written")
        else:
            db.commit()
            print(f"\n[backfill_models] committed {changed} change(s)")
    finally:
        db.close()


if __name__ == "__main__":
    main()
