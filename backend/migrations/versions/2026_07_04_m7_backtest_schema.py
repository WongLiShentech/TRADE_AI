"""m7 backtest schema — trades.ambiguous_resolution + backtest_runs metrics

Adds the two M7 schema changes (Component 1b + 1c; 1a price_type shipped earlier
in 20260603_candles_price_type):

* trades.ambiguous_resolution — BOOLEAN NOT NULL DEFAULT FALSE. Set TRUE by the
  triple-barrier simulator when an exit was resolved via the SL-first tie-break
  (an M1 bar touching both barriers) or the degraded H4/mid fallback (no M1 for
  the window). Lets us re-label those rows once M1 coverage improves.
* backtest_runs += 9 nullable metric columns (profit factor, deflated /
  probabilistic Sharpe, the four outcome-bucket counts, avg holding hours, OOS
  sample size). Nullable because they are populated by the M7 runner (Part B);
  both tables are empty today so no backfill is required.

Revision ID: 20260704_m7_backtest_schema
Revises: 20260605_news_calendar_events
Create Date: 2026-07-04 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '20260704_m7_backtest_schema'
down_revision: Union[str, None] = '20260605_news_calendar_events'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'trades',
        sa.Column(
            'ambiguous_resolution',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('false'),
        ),
    )

    op.add_column('backtest_runs', sa.Column('profit_factor', sa.Float(), nullable=True))
    op.add_column('backtest_runs', sa.Column('deflated_sharpe', sa.Float(), nullable=True))
    op.add_column('backtest_runs', sa.Column('probabilistic_sharpe', sa.Float(), nullable=True))
    op.add_column('backtest_runs', sa.Column('trades_full_win', sa.Integer(), nullable=True))
    op.add_column('backtest_runs', sa.Column('trades_partial', sa.Integer(), nullable=True))
    op.add_column('backtest_runs', sa.Column('trades_breakeven', sa.Integer(), nullable=True))
    op.add_column('backtest_runs', sa.Column('trades_loss', sa.Integer(), nullable=True))
    op.add_column('backtest_runs', sa.Column('avg_holding_hours', sa.Float(), nullable=True))
    op.add_column('backtest_runs', sa.Column('oos_sample_size', sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('backtest_runs', 'oos_sample_size')
    op.drop_column('backtest_runs', 'avg_holding_hours')
    op.drop_column('backtest_runs', 'trades_loss')
    op.drop_column('backtest_runs', 'trades_breakeven')
    op.drop_column('backtest_runs', 'trades_partial')
    op.drop_column('backtest_runs', 'trades_full_win')
    op.drop_column('backtest_runs', 'probabilistic_sharpe')
    op.drop_column('backtest_runs', 'deflated_sharpe')
    op.drop_column('backtest_runs', 'profit_factor')

    op.drop_column('trades', 'ambiguous_resolution')
