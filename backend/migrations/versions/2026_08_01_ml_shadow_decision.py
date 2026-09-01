"""trades.ml_probability / ml_decision / ml_model_id — M8-Shadow inference log

Shadow mode scores every live rule-engine signal with the S1 ML artifact and
LOGS the decision without placing any order. Those three nullable columns are
where the decision lands:

* ``ml_probability`` — the model's P(win) for the signal (0.0–1.0).
* ``ml_decision``    — 'take' | 'skip' (P(win) vs ML_DECISION_THRESHOLD).
* ``ml_model_id``    — the artifact stem that produced the score (provenance), so a
  row can always be traced back to the exact serialized model + params_hash.

Shadow rows are written with ``stage='shadow'``. ``trades.stage`` is a plain
VARCHAR with no CHECK constraint (verified against the live schema), so no
constraint change is needed to admit the new stage value.

All three columns are nullable with no backfill: the 2,809 existing
``stage='backtest'`` rows were produced by the rule engine alone and legitimately
have no ML score.

Revision ID: 20260801_ml_shadow_decision
Revises: 20260704_fold_breakdown
Create Date: 2026-08-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '20260801_ml_shadow_decision'
down_revision: Union[str, None] = '20260704_fold_breakdown'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('trades', sa.Column('ml_probability', sa.Float(), nullable=True))
    op.add_column('trades', sa.Column('ml_decision', sa.String(), nullable=True))
    op.add_column('trades', sa.Column('ml_model_id', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('trades', 'ml_model_id')
    op.drop_column('trades', 'ml_decision')
    op.drop_column('trades', 'ml_probability')
