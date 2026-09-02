"""Attribution layer — strategies, model_decisions, trade_paths, MFE/MAE

Phase A of the multi-strategy / autonomy work. This migration is PURELY ADDITIVE:
it creates three tables and adds four nullable columns. Nothing is dropped,
nothing is altered, no existing value changes. A running deployment applies it
with no behaviour change whatsoever — the new structures are simply unpopulated
until the backfill and the writers land.

What each piece is for
----------------------
``strategies``        Registry of parameterised signal configurations. Today
                      ``trades.signal_source`` says WHICH ENGINE fired ('rule_based')
                      but not WHICH CONFIGURATION, so two variants of the rule engine
                      running side by side are indistinguishable. Identity is
                      ``params_hash`` — a digest over engine + signal-affecting
                      parameters, mirroring how ML artifacts already derive theirs.

``model_decisions``   One model's verdict on one signal, MANY rows per trade. The
                      ``trades.ml_*`` columns hold exactly one opinion, so scoring
                      history with a challenger overwrites the champion's verdict and
                      makes "where they disagreed, who was right?" unanswerable —
                      which is precisely the question promotion depends on.

``trade_paths``       Where a trade travelled, bar by bar. ``rr_actual`` alone cannot
                      distinguish a trade that peaked at +0.84R from one that never
                      moved; those are different problems with identical rows.

``trades.mfe_r/mae_r``  Intrabar-exact extremes, denormalised from the path so the
                      common aggregate ("average MFE of losers") is one GROUP BY
                      rather than a join over hundreds of thousands of path rows.
                      Not redundant with ``trade_paths``: the scalars are more
                      precise, the series is more expressive.

``trades.strategy_id``  Which registry entry produced the signal.
``trades.path_truncated``  The intrabar stream ran out before the horizon, so the
                      extremes may understate the true range. Recorded rather than
                      silently short — an incomplete path otherwise looks exactly
                      like a quiet market.

Leakage
-------
Every value in ``trade_paths`` and in ``mfe_r``/``mae_r`` is derived from prices
AFTER the signal timestamp. They are valid as LABELS (predict how far a trade will
run) and invalid as FEATURES (telling a model how far THIS trade ran hands it the
answer). ``tests/test_feature_contract.py`` asserts no path field ever appears in
``FEATURE_KEYS_MODEL``.

Revision ID: 20260902_attribution
Revises: 20260801_shadow_trade_unique
Create Date: 2026-09-02 07:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '20260902_attribution'
down_revision: Union[str, None] = '20260801_shadow_trade_unique'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── strategies ────────────────────────────────────────────────────────────
    op.create_table(
        'strategies',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('engine', sa.String(), nullable=False),
        sa.Column('params_hash', sa.String(), nullable=False),
        sa.Column('params', sa.JSON(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('notes', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name'),
    )
    # Identity is the parameter set, not the label: registering one configuration
    # twice under two names would split a strategy's evidence in half without
    # anyone noticing. Fail loudly instead.
    op.create_index('uq_strategies_params_hash', 'strategies', ['params_hash'], unique=True)
    op.create_index('ix_strategies_status', 'strategies', ['status'])

    # ── model_decisions ───────────────────────────────────────────────────────
    op.create_table(
        'model_decisions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('trade_id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.String(), nullable=False),
        sa.Column('probability', sa.Float(), nullable=False),
        sa.Column('decision', sa.String(), nullable=False),
        sa.Column('threshold', sa.Float(), nullable=False),
        sa.Column('nan_features', sa.Integer(), nullable=True),
        sa.Column('is_authoritative', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('scored_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        # CASCADE: a decision about a deleted trade is orphaned data, not history.
        sa.ForeignKeyConstraint(['trade_id'], ['trades.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_model_decisions_trade_id', 'model_decisions', ['trade_id'])
    op.create_index('ix_model_decisions_model_id', 'model_decisions', ['model_id'])
    # One opinion per model per trade — rescoring updates in place rather than
    # accumulating duplicates that would skew every aggregate.
    op.create_index(
        'uq_model_decisions_trade_model', 'model_decisions', ['trade_id', 'model_id'], unique=True
    )
    # Partial: only one row per trade is authoritative, so indexing the whole
    # table would be mostly dead weight.
    op.create_index(
        'ix_model_decisions_authoritative',
        'model_decisions',
        ['trade_id'],
        postgresql_where=sa.text('is_authoritative'),
    )

    # ── trade_paths ───────────────────────────────────────────────────────────
    op.create_table(
        'trade_paths',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('trade_id', sa.Integer(), nullable=False),
        sa.Column('bar', sa.Integer(), nullable=False),
        sa.Column('r_close', sa.Float(), nullable=False),
        sa.Column('r_best', sa.Float(), nullable=False),
        sa.Column('r_worst', sa.Float(), nullable=False),
        sa.Column('mfe_r', sa.Float(), nullable=False),
        sa.Column('mae_r', sa.Float(), nullable=False),
        sa.Column('beyond_exit', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('degraded', sa.Boolean(), server_default='false', nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['trade_id'], ['trades.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_trade_paths_trade_id', 'trade_paths', ['trade_id'])
    # One row per (trade, bar): re-running the backfill updates in place rather
    # than silently doubling every path.
    op.create_index('uq_trade_paths_trade_bar', 'trade_paths', ['trade_id', 'bar'], unique=True)

    # ── trades: attribution + excursion columns ───────────────────────────────
    # All nullable, no backfill here. Existing rows legitimately have no strategy
    # attribution and no recorded path until scripts/backfill_attribution.py runs;
    # NULL means "not yet computed", which is the truth.
    op.add_column('trades', sa.Column('strategy_id', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'fk_trades_strategy_id', 'trades', 'strategies', ['strategy_id'], ['id']
    )
    op.create_index('ix_trades_strategy_id', 'trades', ['strategy_id'])
    op.add_column('trades', sa.Column('mfe_r', sa.Float(), nullable=True))
    op.add_column('trades', sa.Column('mae_r', sa.Float(), nullable=True))
    op.add_column(
        'trades',
        sa.Column('path_truncated', sa.Boolean(), server_default='false', nullable=False),
    )


def downgrade() -> None:
    op.drop_column('trades', 'path_truncated')
    op.drop_column('trades', 'mae_r')
    op.drop_column('trades', 'mfe_r')
    op.drop_index('ix_trades_strategy_id', table_name='trades')
    op.drop_constraint('fk_trades_strategy_id', 'trades', type_='foreignkey')
    op.drop_column('trades', 'strategy_id')

    op.drop_index('uq_trade_paths_trade_bar', table_name='trade_paths')
    op.drop_index('ix_trade_paths_trade_id', table_name='trade_paths')
    op.drop_table('trade_paths')

    op.drop_index('ix_model_decisions_authoritative', table_name='model_decisions')
    op.drop_index('uq_model_decisions_trade_model', table_name='model_decisions')
    op.drop_index('ix_model_decisions_model_id', table_name='model_decisions')
    op.drop_index('ix_model_decisions_trade_id', table_name='model_decisions')
    op.drop_table('model_decisions')

    op.drop_index('ix_strategies_status', table_name='strategies')
    op.drop_index('uq_strategies_params_hash', table_name='strategies')
    op.drop_table('strategies')
