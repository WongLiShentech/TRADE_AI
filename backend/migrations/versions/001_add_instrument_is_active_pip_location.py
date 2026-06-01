"""add is_active and pip_location to instruments

Revision ID: 001
Revises:
Create Date: 2026-05-02
"""
from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "instruments",
        sa.Column("pip_location", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "instruments",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="1"),
    )


def downgrade() -> None:
    op.drop_column("instruments", "is_active")
    op.drop_column("instruments", "pip_location")
