"""backtest_runs.fold_breakdown — per-fold + params snapshot (M7 Part B runner)

The M7 walk-forward runner persists ONE BacktestRun row per run, but its result
is inherently multi-part: three expanding OOS folds (IS/OOS metrics each), a
params snapshot, and the promotion-gate verdict detail. The existing
backtest_runs columns are all scalar (Float/Integer/String) with no room for that
structure, so this migration adds a single nullable JSON column ``fold_breakdown``
to hold it (per-fold metrics, gate flags, params snapshot, universe, window).

Nullable + no backfill: backtest_runs is empty today; legacy rows (none) stay NULL.

Revision ID: 20260704_fold_breakdown
Revises: 20260704_m7_backtest_schema
Create Date: 2026-07-04 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '20260704_fold_breakdown'
down_revision: Union[str, None] = '20260704_m7_backtest_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('backtest_runs', sa.Column('fold_breakdown', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('backtest_runs', 'fold_breakdown')
