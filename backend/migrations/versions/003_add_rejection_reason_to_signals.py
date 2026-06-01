"""add rejection_reason column to signals

Revision ID: 003
Revises: 002
Create Date: 2026-05-22
"""
from alembic import op
import sqlalchemy as sa

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("signals", sa.Column("rejection_reason", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("signals", "rejection_reason")
