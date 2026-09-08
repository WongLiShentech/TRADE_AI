"""Models registry + strategies.description

PURELY ADDITIVE: creates one table and adds one nullable column. Nothing is
dropped, nothing is altered, no existing value changes. A running deployment
applies it with no behaviour change — the new structures are simply unpopulated
until the backfill runs.

``models``
----------
A model is a FILE (``backend/models/*.joblib``); the database refers to it by a
bare string in ``trades.ml_model_id`` and ``model_decisions.model_id``, and
nothing records what that string MEANS. This table makes the database
self-describing: what exists, over what span, on whose trades.

The load-bearing column is ``strategy_id``. A model's labels come from
``rr_actual``, which depends on the EXIT RULE — relabelling one corpus under a
pure-barrier exit instead of a trailing one flips 12.5% of the training set.
A model is only valid for the strategy whose outcomes taught it, and until now
nothing in the schema could express that, let alone check it.

``strategies.description``
--------------------------
``notes`` was carrying two jobs: what the strategy IS (stable — it changes only
when the config changes, which mints a new ``params_hash``) and what HAPPENED to
it (operational, ever-growing). Splitting them keeps the definition legible once
a second strategy exists to compare against.

Revision ID: 20260908_models_registry
Revises: 20260902_attribution
Create Date: 2026-09-08 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '20260908_models_registry'
down_revision: Union[str, None] = '20260902_attribution'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'models',
        sa.Column('id', sa.Integer(), nullable=False),
        # Matches the artifact filename stem, so a row always resolves to a file.
        sa.Column('model_id', sa.String(), nullable=False),
        # Nullable: a model trained before its strategy was registered still deserves
        # a row. Recording "unknown" beats inventing a link that was never verified.
        sa.Column('strategy_id', sa.Integer(), nullable=True),
        # Which trade populations it learned from — 'backtest' today, and
        # 'backtest,shadow,sandbox' once live rows enter the training set.
        sa.Column('trained_on_stages', sa.String(), nullable=False),
        sa.Column('training_start', sa.DateTime(), nullable=True),
        sa.Column('training_end', sa.DateTime(), nullable=True),
        sa.Column('training_rows', sa.Integer(), nullable=True),
        # Without the threshold a recorded decision cannot be reproduced: the
        # probability alone does not say what it was compared against.
        sa.Column('decision_threshold', sa.Float(), nullable=True),
        # NULLABLE ON PURPOSE. v1 predates the promotion gate; "unknown" is the
        # honest value, and `false` would invent a verdict nobody reached.
        sa.Column('passed_gate', sa.Boolean(), nullable=True),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_id'),
        sa.ForeignKeyConstraint(['strategy_id'], ['strategies.id']),
    )
    op.create_index('ix_models_strategy_id', 'models', ['strategy_id'])
    op.create_index('ix_models_status', 'models', ['status'])

    op.add_column('strategies', sa.Column('description', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('strategies', 'description')
    op.drop_index('ix_models_status', table_name='models')
    op.drop_index('ix_models_strategy_id', table_name='models')
    op.drop_table('models')
