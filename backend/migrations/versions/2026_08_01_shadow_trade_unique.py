"""uq_trades_shadow_natural_key — one shadow row per (instrument, signal_time)

M8-Shadow Phase 2 wires shadow observation into the LIVE candle-close pipeline. A
shadow row's natural key is ``(instrument_id, opened_at)`` where ``opened_at`` is
the signal timestamp T, deterministically derived from the decision bar
(``bar_close + 1s``). Re-running the pipeline for the same bar — a retried job, a
manual invocation, two overlapping scheduler fires — must therefore NEVER produce a
second row for the same signal, or the shadow corpus double-counts.

``app/services/shadow/recorder.py`` already query-before-inserts. This partial
unique index is the DURABLE backstop for the race that a query-before-insert cannot
close; the recorder catches the resulting IntegrityError and treats it as
"already recorded".

The index is PARTIAL (``WHERE stage = 'shadow'``) on purpose:

* it constrains only shadow rows, and
* the existing ``stage='backtest'`` corpus is untouched and remains free to repeat
  an (instrument_id, opened_at) pair across backtest re-runs.

Revision ID: 20260801_shadow_trade_unique
Revises: 20260801_ml_shadow_decision
Create Date: 2026-08-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '20260801_shadow_trade_unique'
down_revision: Union[str, None] = '20260801_ml_shadow_decision'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_INDEX_NAME = "uq_trades_shadow_natural_key"


def upgrade() -> None:
    op.create_index(
        _INDEX_NAME,
        "trades",
        ["instrument_id", "opened_at"],
        unique=True,
        postgresql_where=sa.text("stage = 'shadow'"),
    )


def downgrade() -> None:
    op.drop_index(_INDEX_NAME, table_name="trades")
