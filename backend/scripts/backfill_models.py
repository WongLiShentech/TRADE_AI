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
from app.models import MLModel, Strategy

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
        "status": "retired",
        "description": (
            "First XGBoost signal filter. 20 features (schema v1), 2,809 backtest "
            "trades over 2 years. Predates the promotion gate, so it was never "
            "formally evaluated against it. Superseded by v2."
        ),
    },
    "s1_xgb_v2_c430d22dd1ab.NOT_PROMOTED": {
        "strategy_name": "rule_based_v1",
        "status": "shadow",
        "description": (
            "Current shadow model. 19 features (schema v2), 7,074 backtest trades "
            "over 5 years. Failed the promotion gate on fold-1 expectancy "
            "(0.1473 vs the 0.1500 floor). Scores live signals in observe mode "
            "only; the .NOT_PROMOTED filename is enforced at load time."
        ),
    },
}

STRATEGY_DESCRIPTIONS = {
    "rule_based_v1": (
        "H4 pullback-in-trend across 10 FX majors. 5-condition confluence, 3-of-5 "
        "required. ATR(14) stop, 2R target, partial exit + trailing stop at 0.5R, "
        "10-bar time exit."
    ),
}


def _dt(value):
    return datetime.fromisoformat(value) if value else None


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
                trained_on_stages="backtest",
                training_start=_dt(span.get("start")),
                training_end=_dt(span.get("end")),
                training_rows=span.get("n_rows"),
                decision_threshold=meta.get("deployment_threshold"),
                # None (not False) when the artifact predates the gate — "unknown"
                # is the honest value; False would invent a verdict.
                passed_gate=meta.get("promoted"),
                status=cur["status"],
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
