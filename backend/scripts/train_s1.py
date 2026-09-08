"""S1 — train + evaluate the XGBoost signal-filter on the M7 rule-engine corpus.

Run from ``backend/`` — both variables are REQUIRED::

    S1_STRATEGY_ID=2 S1_MODEL_VERSION=3 python scripts/train_s1.py

``S1_STRATEGY_ID`` because labels derive from ``rr_actual`` and ``rr_actual`` depends
on the exit rule, so "the backtest corpus" is no longer one thing. ``S1_MODEL_VERSION``
because it enters the artifact's identity and filename — guessing it risks colliding
with an artifact the live server loads.

Pipeline (single specified recipe — no hyperparameter iteration):
  1. Load that strategy's ``stage='backtest'`` corpus (locked feature contract + label).
  2. Walk-forward evaluate the ML-FILTERED strategy on M7's three expanding folds,
     compared side-by-side against the unfiltered rule baseline, through the SAME
     promotion gate (deflated Sharpe uses the durable honest trial count).
  3. Compute SHAP global importances (training data) — top-10 + dead features.
  4. Train the deployable candidate on ALL rows (fixed seed) and serialize it +
     metadata JSON + model card to ``backend/models/``.
  5. Write a structured experiment log JSON to the output folder.

Read-only against the DB (no rows written). Deterministic given ``ML_SEED``.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings
from app.database import SessionLocal
from app.services.backtester.runner import _durable_n_trials  # honest trial counter
from app.services.ml.artifact import save_artifact, train_production_model
from app.services.ml.dataset import load_dataset
from app.services.ml.evaluate import (
    WalkForwardResult,
    evaluate_walk_forward,
    keep_fraction_report,
)
from app.services.provenance import git_provenance

_OUTPUT_DIR = BACKEND_ROOT.parent / "output"
# Which strategy's corpus to train on. REQUIRED: labels derive from rr_actual, which
# depends on the exit rule, so "the backtest corpus" stopped being a single thing the
# moment a second strategy was registered.
_STRATEGY_ID = int(os.environ["S1_STRATEGY_ID"]) if os.environ.get("S1_STRATEGY_ID") else None
# The sequential MODEL version (1, 2, 3...), NOT the feature schema version. It enters
# the artifact's params_hash and its filename, so it is required rather than defaulted:
# a wrong guess can collide with an existing artifact, and the one on disk is what the
# live server loads.
_MODEL_VERSION = int(os.environ["S1_MODEL_VERSION"]) if os.environ.get("S1_MODEL_VERSION") else None
_MODELS_DIR = BACKEND_ROOT / "models"


def _fmt(v) -> str:
    if v is None:
        return "  n/a"
    if isinstance(v, float):
        if v != v:  # NaN
            return "  nan"
        return f"{v:6.3f}"
    return str(v)


def _print_report(result: WalkForwardResult, artifact_paths: dict) -> None:
    print("\n" + "=" * 78)
    print("S1 WALK-FORWARD EVALUATION — ML-filtered strategy vs rule baseline")
    print("=" * 78)
    print(f"label rule           : y = 1 if rr_actual >= {result.label_threshold_r}R")
    print(f"honest n_trials (DSR): {result.n_trials}")

    print("\nPER-FOLD (OOS) — ML-filtered:")
    hdr = f"{'fold':>4} {'kept/tot':>10} {'thr':>6} {'valAUC':>7} {'PF':>7} {'exp':>7} {'maxDD':>7} {'WR':>7} {'DSR':>7} {'gate':>5}"
    print(hdr)
    for fe in result.folds:
        m = fe.ml_filtered_metrics
        g = "PASS" if fe.ml_gate.get("passed") else "FAIL"
        deg = f"  [{fe.degenerate_reason}]" if fe.degenerate_reason else ""
        print(
            f"{fe.fold:>4} {str(fe.kept)+'/'+str(fe.oos_trade_count):>10} "
            f"{_fmt(fe.threshold)} {_fmt(fe.val_auc)} {_fmt(m['profit_factor'])} "
            f"{_fmt(m['expectancy'])} {_fmt(m['max_drawdown'])} {_fmt(m['win_rate'])} "
            f"{_fmt(m['deflated_sharpe'])} {g:>5}{deg}"
        )

    print("\nOOS KEEP-FRACTION (filter-collapse watch — QA H1):")
    print(f"{'fold':>4} {'kept':>6} {'total':>6} {'keep%':>7} {'thr':>7}   note")
    for row in keep_fraction_report(result):
        note = f"[{row['degenerate_reason']}]" if row["degenerate_reason"] else ""
        collapse = "  <-- COLLAPSE" if (row["keep_fraction"] < 0.2 and not row["degenerate_reason"]) else ""
        print(
            f"{row['fold']:>4} {row['kept']:>6} {row['oos_total']:>6} "
            f"{row['keep_fraction']*100:>6.1f}% {_fmt(row['threshold'])}   {note}{collapse}"
        )

    print("\nPER-FOLD (OOS) — rule baseline (unfiltered):")
    print(hdr.replace("kept/tot", "   count").replace("   thr", "   ").replace("valAUC", "      "))
    for fe in result.folds:
        m = fe.baseline_metrics
        g = "PASS" if fe.baseline_gate.get("passed") else "FAIL"
        print(
            f"{fe.fold:>4} {m['trade_count']:>10} {'':>6} {'':>7} {_fmt(m['profit_factor'])} "
            f"{_fmt(m['expectancy'])} {_fmt(m['max_drawdown'])} {_fmt(m['win_rate'])} "
            f"{_fmt(m['deflated_sharpe'])} {g:>5}"
        )

    mc = result.ml_combined_metrics
    bc = result.baseline_combined_metrics
    print("\nCOMBINED OOS:")
    print(f"  ML-filtered : trades={mc['trade_count']} PF={_fmt(mc['profit_factor'])} "
          f"exp={_fmt(mc['expectancy'])} maxDD={_fmt(mc['max_drawdown'])} "
          f"WR={_fmt(mc['win_rate'])} DSR={_fmt(mc['deflated_sharpe'])}")
    print(f"  baseline    : trades={bc['trade_count']} PF={_fmt(bc['profit_factor'])} "
          f"exp={_fmt(bc['expectancy'])} maxDD={_fmt(bc['max_drawdown'])} "
          f"WR={_fmt(bc['win_rate'])} DSR={_fmt(bc['deflated_sharpe'])}")

    print(f"\nGATE VERDICT: ML-filtered={'PASS' if result.ml_passed else 'FAIL'}  "
          f"baseline={'PASS' if result.baseline_passed else 'FAIL'}")
    print(f"artifact: {artifact_paths['model']}")
    print("=" * 78 + "\n")


def _serialise_result(result: WalkForwardResult, artifact_paths: dict) -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_trials": result.n_trials,
        "label_threshold_r": result.label_threshold_r,
        "ml_passed": result.ml_passed,
        "baseline_passed": result.baseline_passed,
        "ml_combined_metrics": result.ml_combined_metrics,
        "baseline_combined_metrics": result.baseline_combined_metrics,
        "ml_gate_detail": result.ml_gate_detail,
        "baseline_gate_detail": result.baseline_gate_detail,
        "folds": [asdict(fe) for fe in result.folds],
        "keep_fraction_report": keep_fraction_report(result),
        "artifact_paths": artifact_paths,
    }


def main() -> None:
    # Windows consoles default to cp1252; force UTF-8 so report glyphs never crash.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    settings = get_settings()
    db = SessionLocal()
    try:
        if _STRATEGY_ID is None:
            raise SystemExit(
                "S1_STRATEGY_ID is required. Training must name the strategy whose "
                "outcomes it learns from: labels derive from rr_actual and rr_actual "
                "depends on the exit rule, so a corpus spanning two strategies teaches "
                "two contradictory definitions of a win."
            )
        if _MODEL_VERSION is None:
            raise SystemExit(
                "S1_MODEL_VERSION is required. It is the sequential MODEL version "
                "(1, 2, 3...), NOT the feature schema version — and it is part of the "
                "artifact's identity, so guessing it risks overwriting a live model."
            )

        dataset = load_dataset(db, settings.ML_LABEL_THRESHOLD_R, strategy_id=_STRATEGY_ID)
        n_pos = int(dataset.y.sum())
        print(f"[train_s1] strategy={_STRATEGY_ID} version={_MODEL_VERSION} "
              f"corpus={len(dataset)} rows  positives(>= "
              f"{settings.ML_LABEL_THRESHOLD_R}R)={n_pos} ({n_pos/len(dataset):.1%})", flush=True)

        provenance = git_provenance()
        print(f"[train_s1] code: {provenance.commit or 'unknown'}"
              f"{' (DIRTY — uncommitted changes)' if provenance.dirty else ''}", flush=True)
        print(f"[train_s1] dataset fingerprint: {dataset.fingerprint()}", flush=True)

        n_trials = _durable_n_trials(db)
        print(f"[train_s1] honest n_trials (durable counter) = {n_trials}", flush=True)

        result = evaluate_walk_forward(db, settings, n_trials, dataset=dataset)

        print("[train_s1] training production candidate on ALL rows...", flush=True)
        candidate = train_production_model(
            dataset, settings, version=_MODEL_VERSION, provenance=provenance
        )
        # QA fix (a): tag the artifact with the walk-forward gate verdict so a
        # DO-NOT-PROMOTE candidate can never be silently deployed.
        artifact_paths = save_artifact(candidate, _MODELS_DIR, promoted=result.ml_passed)
        if not result.ml_passed:
            print("[train_s1] GATE=DO-NOT-PROMOTE — artifact tagged NOT_PROMOTED "
                  "(serialized for post-mortem only).", flush=True)

        _print_report(result, artifact_paths)
        print("SHAP top-10:", flush=True)
        for k, v in candidate.shap.top10:
            print(f"  {k:<30} {v:.6f}", flush=True)
        print(f"SHAP dead features: {candidate.shap.dead_features or '(none)'}", flush=True)

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # Strategy-scoped so training a second strategy cannot silently destroy the
        # first one's run record — the two are not versions of one result, they
        # describe different corpora and both stay relevant.
        log_path = _OUTPUT_DIR / (
            f"s1_experiment_log_strategy{_STRATEGY_ID}.json" if _STRATEGY_ID
            else "s1_experiment_log.json"
        )
        log_path.write_text(
            json.dumps(_serialise_result(result, artifact_paths), indent=2, default=str),
            encoding="utf-8",
        )
        print(f"[train_s1] experiment log → {log_path.resolve()}", flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    main()
