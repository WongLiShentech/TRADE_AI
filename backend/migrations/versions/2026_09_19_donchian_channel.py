"""Add Donchian channel columns to indicators.

Why
---
``c3_structure`` ("price is at structure") was computed from ``swing_high``/``swing_low``,
which come from a CENTRED window: the value at bar *i* is determined by bars up to
*i+k*. Two consequences, both measured:

* In backtest, indicators are recomputed over full history, so a row at time *i* carries
  a swing that was decided using data after *i*. The engine read those raw
  (``rule_based.py`` took the newest 20 indicator rows), while the leakage firewall
  (``feature_builder.causal_swing_levels``) discards exactly the newest
  ``SWING_LOOKBACK_PERIODS`` = 20 rows. The engine's read window and the firewall's
  discard window were exact complements — 100% look-ahead.
* In live, the swing could not be computed at all (warmup too short for the window to
  close), so ``c3_structure`` was False on 100% of live rows against 20% of backtest
  rows, and it is the condition carrying the edge (+0.654R expectancy when true vs
  +0.044R when false).

A Donchian extreme over a TRAILING window including the current bar is knowable at its
own timestamp, so it answers the same question with no confirmation lag and no
look-ahead, and it is dense (every bar) rather than the ~1.6% of bars carrying a swing.

Nullable and backfilled separately
----------------------------------
Both columns are nullable with no server default. Between this migration and the
backfill (``scripts/recompute_indicators.py``) every row reads NULL, which is the honest
state — "not computed yet", never a fabricated extreme. The signal engine must treat a
NULL Donchian as "cannot evaluate this condition" rather than as a failed condition.

Touches ``indicators`` only. ``swing_high``/``swing_low`` are deliberately kept: they
still back the ``swing_dist_atr`` model feature, which reads them through the causal
accessor.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# Revision id must be <= 32 chars: alembic_version.version_num is VARCHAR(32) and a
# longer id fails at the version STAMP, after the DDL has applied, rolling the whole
# transaction back. Guarded by test_every_revision_id_fits_alembic_version_column.
revision: str = '20260919_donchian_channel'
down_revision: Union[str, None] = '20260911_obs_natural_key'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('indicators', sa.Column('donchian_high', sa.Float(), nullable=True))
    op.add_column('indicators', sa.Column('donchian_low', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('indicators', 'donchian_low')
    op.drop_column('indicators', 'donchian_high')
