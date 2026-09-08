"""Run and dataset lineage — which execution produced a row, which data trained a model

PURELY ADDITIVE: one new table, six new columns across three existing tables, three
indexes. Nothing is dropped, nothing altered, no existing value changed. A running
deployment applies it with no behaviour change — the structures are unpopulated until
``scripts/backfill_lineage.py`` runs.

Touches nothing on ``candles`` (36.8M rows). The largest table this reaches is
``trades`` at ~13,500 rows locally and ~7,200 on the server, where an ACCESS EXCLUSIVE
lock for ``ADD COLUMN`` + ``CREATE INDEX`` is milliseconds. Indexes are built plainly,
not ``CONCURRENTLY``, which cannot run inside Alembic's per-revision transaction.

What each piece is for
----------------------
``trades.run_id``       WHICH EXECUTION produced this row. ``strategy_id`` names the
                        configuration, but two backtests of one configuration — before
                        and after a simulator fix — are otherwise indistinguishable and
                        merge into a single pile. Today a re-run must DELETE the prior
                        corpus to stay unambiguous, destroying the comparison it was
                        for.

``backtest_runs``       ``strategy_id`` (the existing ``strategy`` string was written
  + strategy_id,        from a module constant and was therefore wrong the first time
    git_commit,         two strategies existed — run 7 executed rule_based_v2_fixed and
    git_dirty           recorded rule_based_v1). ``git_commit``/``git_dirty`` record
                        WHICH CODE ran, which cannot be reconstructed afterwards.

``datasets``            The exact corpus a model saw: a declarative definition (stage,
                        strategy, run) plus a content fingerprint proving what it
                        resolved to. ``models.strategy_id`` names a LIVE QUERY whose
                        rows grow and get re-graded, so two models can claim one corpus
                        and have seen different data. UNIQUE on ``fingerprint`` so a
                        retrain over identical data reuses the row — champion and
                        challenger sharing a dataset then becomes a fact the schema
                        states rather than one you verify by hand.

``models``              ``version`` (1, 2, 3 — NOT the feature schema version the old
  + version,            ``s1_xgb_v2_*`` filenames actually encoded; they coincided by
    dataset_id,         accident and diverge at model 3). ``dataset_id`` links to the
    git_commit,         corpus. ``n_trials`` is the multiple-testing count the results
    git_dirty,          were discounted by, and ``n_trials_source`` records where that
    n_trials,           number came from — the counter is currently global and differs
    n_trials_source     by host, so a bare integer would be uninterpretable later.

Nullability
-----------
Every new column is nullable, and that is a statement rather than convenience. NULL
means "not recorded": v1 and v2 predate provenance capture, and the pre-lineage trade
corpus has no recoverable run. ``git_dirty`` deliberately carries NO server_default —
``NULL`` (unknown) and ``False`` (checked, and the tree was clean) are different
claims, and defaulting would convert every unknown into a false assertion of
cleanliness.

Revision ID: 20260909_run_dataset_lineage
Revises: 20260908_models_registry
Create Date: 2026-09-09 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '20260909_run_dataset_lineage'
down_revision: Union[str, None] = '20260908_models_registry'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── datasets ──────────────────────────────────────────────────────────────
    # Created first: models.dataset_id FKs into it.
    op.create_table(
        'datasets',
        sa.Column('id', sa.Integer(), nullable=False),
        # Identity is the CONTENT, not the query that produced it.
        sa.Column('fingerprint', sa.String(), nullable=False),
        # The declarative definition — what was asked for.
        sa.Column('stage', sa.String(), nullable=False),
        sa.Column('strategy_id', sa.Integer(), nullable=True),
        sa.Column('run_id', sa.Integer(), nullable=True),
        # What it resolved to — readable without recomputing the digest.
        sa.Column('n_rows', sa.Integer(), nullable=False),
        sa.Column('n_positive', sa.Integer(), nullable=True),
        sa.Column('span_start', sa.DateTime(), nullable=True),
        sa.Column('span_end', sa.DateTime(), nullable=True),
        # Context for diagnosing a mismatch. Adding one feature key changes EVERY
        # historical fingerprint though no stored row moved; these let that report as
        # "the contract changed" rather than "all your data changed at once".
        sa.Column('label_threshold_r', sa.Float(), nullable=True),
        sa.Column('feature_schema_version', sa.Integer(), nullable=True),
        sa.Column('feature_keys_hash', sa.String(), nullable=True),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('resolved_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['strategy_id'], ['strategies.id']),
        sa.ForeignKeyConstraint(['run_id'], ['backtest_runs.id']),
    )
    op.create_index('uq_datasets_fingerprint', 'datasets', ['fingerprint'], unique=True)
    op.create_index('ix_datasets_strategy_id', 'datasets', ['strategy_id'])

    # ── backtest_runs ─────────────────────────────────────────────────────────
    op.add_column('backtest_runs', sa.Column('strategy_id', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'fk_backtest_runs_strategy_id', 'backtest_runs', 'strategies', ['strategy_id'], ['id']
    )
    op.create_index('ix_backtest_runs_strategy_id', 'backtest_runs', ['strategy_id'])
    op.add_column('backtest_runs', sa.Column('git_commit', sa.String(), nullable=True))
    op.add_column('backtest_runs', sa.Column('git_dirty', sa.Boolean(), nullable=True))

    # ── trades ────────────────────────────────────────────────────────────────
    op.add_column('trades', sa.Column('run_id', sa.Integer(), nullable=True))
    op.create_foreign_key('fk_trades_run_id', 'trades', 'backtest_runs', ['run_id'], ['id'])
    op.create_index('ix_trades_run_id', 'trades', ['run_id'])

    # ── models ────────────────────────────────────────────────────────────────
    op.add_column('models', sa.Column('version', sa.Integer(), nullable=True))
    op.add_column('models', sa.Column('dataset_id', sa.Integer(), nullable=True))
    op.create_foreign_key('fk_models_dataset_id', 'models', 'datasets', ['dataset_id'], ['id'])
    op.create_index('ix_models_dataset_id', 'models', ['dataset_id'])
    op.add_column('models', sa.Column('git_commit', sa.String(), nullable=True))
    op.add_column('models', sa.Column('git_dirty', sa.Boolean(), nullable=True))
    op.add_column('models', sa.Column('n_trials', sa.Integer(), nullable=True))
    op.add_column('models', sa.Column('n_trials_source', sa.String(), nullable=True))
    # PARTIAL unique index: without it two rows can both claim "version 2 of strategy
    # 1" and the version number means nothing. Partial so an unassigned version stays
    # recordable as NULL rather than being forced into the sequence.
    op.create_index(
        'uq_models_strategy_version',
        'models',
        ['strategy_id', 'version'],
        unique=True,
        postgresql_where=sa.text('version IS NOT NULL'),
    )


def downgrade() -> None:
    op.drop_index('uq_models_strategy_version', table_name='models')
    op.drop_column('models', 'n_trials_source')
    op.drop_column('models', 'n_trials')
    op.drop_column('models', 'git_dirty')
    op.drop_column('models', 'git_commit')
    op.drop_index('ix_models_dataset_id', table_name='models')
    op.drop_constraint('fk_models_dataset_id', 'models', type_='foreignkey')
    op.drop_column('models', 'dataset_id')
    op.drop_column('models', 'version')

    op.drop_index('ix_trades_run_id', table_name='trades')
    op.drop_constraint('fk_trades_run_id', 'trades', type_='foreignkey')
    op.drop_column('trades', 'run_id')

    op.drop_column('backtest_runs', 'git_dirty')
    op.drop_column('backtest_runs', 'git_commit')
    op.drop_index('ix_backtest_runs_strategy_id', table_name='backtest_runs')
    op.drop_constraint('fk_backtest_runs_strategy_id', 'backtest_runs', type_='foreignkey')
    op.drop_column('backtest_runs', 'strategy_id')

    op.drop_index('ix_datasets_strategy_id', table_name='datasets')
    op.drop_index('uq_datasets_fingerprint', table_name='datasets')
    op.drop_table('datasets')
