"""recreate signals table for M5 rule-based signal engine

Replaces the M0 stub schema (timestamp, stop_method, stop_pips, tp_pips, rr_ratio,
signal_source, validated) with the M5 schema (granularity, entry, stop, target,
confidence_score, score_breakdown, status, created_at, expires_at).

Safe to run: prior table contained 0 rows and was unused by any business logic.

Revision ID: 002
Revises: 001
Create Date: 2026-05-13
"""
from alembic import op
import sqlalchemy as sa


revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("signals")
    op.create_table(
        "signals",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("instrument_id", sa.Integer(), sa.ForeignKey("instruments.id"), nullable=False),
        sa.Column("granularity", sa.String(), nullable=False),
        sa.Column("direction", sa.String(), nullable=False),
        sa.Column("entry", sa.Float(), nullable=False),
        sa.Column("stop", sa.Float(), nullable=False),
        sa.Column("target", sa.Float(), nullable=False),
        sa.Column("confidence_score", sa.Integer(), nullable=False),
        sa.Column("score_breakdown", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="PENDING"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_signals_instrument_id", "signals", ["instrument_id"])
    op.create_index("ix_signals_instrument_status", "signals", ["instrument_id", "status"])
    op.create_index("ix_signals_created_at", "signals", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_signals_created_at", table_name="signals")
    op.drop_index("ix_signals_instrument_status", table_name="signals")
    op.drop_index("ix_signals_instrument_id", table_name="signals")
    op.drop_table("signals")
    op.create_table(
        "signals",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("instrument_id", sa.Integer(), sa.ForeignKey("instruments.id"), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("direction", sa.String(), nullable=False),
        sa.Column("stop_method", sa.String(), nullable=False),
        sa.Column("stop_pips", sa.Float(), nullable=False),
        sa.Column("tp_pips", sa.Float(), nullable=False),
        sa.Column("rr_ratio", sa.Float(), nullable=False),
        sa.Column("signal_source", sa.String(), nullable=False),
        sa.Column("validated", sa.Boolean(), nullable=False, server_default="0"),
    )
    op.create_index("ix_signals_instrument_id", "signals", ["instrument_id"])
    op.create_index("ix_signals_timestamp", "signals", ["timestamp"])
