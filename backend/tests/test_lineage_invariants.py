"""Invariants that keep provenance honest once it exists.

Each of these encodes a defect that actually shipped. They are cheap, and they are
the only thing standing between "we recorded lineage" and "the lineage is true".
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

from app.services.backtester import runner as runner_module

# Rows opened from this instant are written by a runner that stamps run_id. Anything
# earlier predates lineage and is legitimately NULL — see models/trade.py.
LINEAGE_CUTOFF = "2026-09-09"


def test_runner_has_no_hardcoded_strategy_constant():
    """`_STRATEGY = "rule_based_v1"` was written onto every BacktestRun row and into
    fold_breakdown. It was wrong the first time a second strategy existed: run 7
    executed rule_based_v2_fixed and recorded rule_based_v1. The name must be resolved
    from the strategy the run was asked to execute, never from a module constant."""
    assert not hasattr(runner_module, "_STRATEGY"), (
        "runner._STRATEGY is back — a module constant cannot know which strategy a "
        "run executed, and will mislabel every row the moment there are two"
    )


def test_run_backtest_requires_a_strategy_id():
    """An unattributed corpus is not untidy, it is broken: load_dataset treats a NULL
    strategy_id as its own strategy, so one unattributed run trips the mixed-corpus
    guard and blocks training until somebody attributes the rows by hand."""
    import inspect

    sig = inspect.signature(runner_module.run_backtest)
    param = sig.parameters["strategy_id"]
    assert param.default is inspect.Parameter.empty, "strategy_id must have no default"
    assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
        "strategy_id must be keyword-only so it cannot be passed positionally into "
        "the `instruments` slot"
    )


def test_every_model_id_resolves_to_a_registered_model(db):
    """The registry must join to the decisions it describes.

    It did not, on first release: `models.model_id` stripped `.NOT_PROMOTED` while
    `inference._model_id_for` keeps it, so the join returned zero rows and the table
    described nothing. A registry that cannot be joined is decoration.
    """
    orphans = db.execute(
        text(
            "SELECT DISTINCT t.ml_model_id FROM trades t "
            "WHERE t.ml_model_id IS NOT NULL "
            "AND NOT EXISTS (SELECT 1 FROM models m WHERE m.model_id = t.ml_model_id)"
        )
    ).scalars().all()
    assert not orphans, (
        f"trades reference model_ids absent from the registry: {orphans} — "
        f"check that backfill_models keys on the FULL artifact stem"
    )


def test_backtest_rows_written_after_the_cutoff_carry_a_run_id(db):
    """Forward enforcement, not retroactive. The pre-lineage corpus is permanently
    NULL — `trades` has no creation timestamp and the runs that produced those rows
    were deleted — so no constraint can conjure the answer. What CAN be enforced is
    that nothing new joins them."""
    n = db.execute(
        text(
            "SELECT count(*) FROM trades "
            "WHERE stage = 'backtest' AND run_id IS NULL AND opened_at >= :cutoff"
        ),
        {"cutoff": LINEAGE_CUTOFF},
    ).scalar_one()
    assert n == 0, f"{n} backtest rows opened after {LINEAGE_CUTOFF} have no run_id"


def test_backtest_corpus_is_fully_attributed_to_a_strategy(db):
    """A NULL strategy_id is not a wildcard — it is an unknown, and load_dataset
    counts it as its own strategy. One unattributed row therefore breaks every
    unscoped training load."""
    n = db.execute(
        text("SELECT count(*) FROM trades WHERE stage = 'backtest' AND strategy_id IS NULL")
    ).scalar_one()
    assert n == 0, f"{n} backtest rows are unattributed — run backfill_attribution"


def test_backtest_run_strategy_name_agrees_with_its_strategy_id(db):
    """`strategy` is denormalised for readability; `strategy_id` is the authority.
    When they disagree, the readable one is the lie people will act on."""
    mismatches = db.execute(
        text(
            "SELECT r.id, r.strategy, s.name FROM backtest_runs r "
            "JOIN strategies s ON s.id = r.strategy_id "
            "WHERE r.strategy IS DISTINCT FROM s.name"
        )
    ).all()
    assert not mismatches, f"backtest_runs.strategy disagrees with strategy_id: {mismatches}"


def test_no_two_models_claim_the_same_version_of_one_strategy(db):
    """Enforced by a partial unique index; asserted here so a dropped index is caught
    by the suite rather than by a confusing registry six months later."""
    dupes = db.execute(
        text(
            "SELECT strategy_id, version, count(*) FROM models "
            "WHERE version IS NOT NULL GROUP BY 1, 2 HAVING count(*) > 1"
        )
    ).all()
    assert not dupes, f"duplicate (strategy_id, version): {dupes}"


# ── migration hygiene ────────────────────────────────────────────────────────
def test_every_revision_id_fits_alembic_version_column():
    """`alembic_version.version_num` is VARCHAR(32) — a longer revision id fails at
    the END of a successful migration, when Alembic stamps the new version.

    It cost a deploy rehearsal to find: `20260911_shadow_key_multistrategy` is 33
    characters, so the DDL applied, the stamp raised DataError, and the whole
    transaction rolled back. On the server that is a failed `alembic upgrade head`
    inside an entrypoint running `set -e` — the container does not start.
    """
    import re
    from pathlib import Path

    versions = Path(__file__).resolve().parents[1] / "migrations" / "versions"
    offenders = []
    for f in versions.glob("*.py"):
        m = re.search(r"^revision:\s*str\s*=\s*['\"]([^'\"]+)['\"]", f.read_text(encoding="utf-8"), re.M)
        if m and len(m.group(1)) > 32:
            offenders.append((f.name, m.group(1), len(m.group(1))))
    assert not offenders, f"revision ids exceeding 32 chars: {offenders}"


def test_observation_key_covers_strategy_and_sandbox(db):
    """The natural key must include the strategy, or a second strategy firing on the
    same instrument and bar is mistaken for a duplicate and silently discarded — and
    it must cover sandbox, whose rows correspond to real broker orders."""
    row = db.execute(
        text(
            "SELECT indexdef FROM pg_indexes "
            "WHERE tablename = 'trades' AND indexname = 'uq_trades_observation_natural_key'"
        )
    ).first()
    assert row is not None, "uq_trades_observation_natural_key is missing"
    definition = row[0]
    assert "strategy_id" in definition, "strategy is not part of the natural key"
    assert "sandbox" in definition, "sandbox rows are not covered by the guard"
    assert "UNIQUE" in definition.upper()


def test_pre_insert_duplicate_check_matches_the_index(db):
    """The recorder's query-before-insert must key on the same columns as the index.
    Narrower and writes reach the database and raise; wider and real duplicates slip
    through to be caught only by the index."""
    import inspect

    from app.services.shadow import recorder

    src = inspect.getsource(recorder._existing_shadow_row)
    assert "strategy_id" in src, (
        "_existing_shadow_row ignores strategy_id while the unique index includes it — "
        "the two guards disagree"
    )
