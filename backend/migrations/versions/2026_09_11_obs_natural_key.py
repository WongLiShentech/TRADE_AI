"""Widen the shadow idempotency key for multiple strategies and sandbox rows

The old index was::

    uq_trades_shadow_natural_key ON (instrument_id, opened_at) WHERE stage = 'shadow'

Two separate problems with that, both of which only appear once the platform does
what it was built to do.

1. IT BLOCKS MULTIPLE STRATEGIES AT THE DATABASE LEVEL
   Two strategies watching the same 10 pairs will routinely fire on the same
   instrument at the same H4 close — that is not an edge case, it is the normal case
   for variants sharing an entry rule. Under the old index the second write raises
   IntegrityError, the recorder catches it as "already recorded", and that strategy's
   decision is silently discarded. The failure mode is not a crash; it is one
   strategy quietly recording nothing whenever it agrees with the other.

2. IT DOES NOT COVER SANDBOX
   `_derive_stage` routes a take to stage='sandbox' when an order is placed. Those
   rows had no idempotency guard at all, so a re-run of the same candle close could
   double-write a row that corresponds to a REAL broker order.

Why COALESCE(strategy_id, 0) rather than the bare column
--------------------------------------------------------
In Postgres NULL is never equal to NULL, so a unique index containing a nullable
column does not constrain rows where it is NULL — they are all mutually distinct.
Adding `strategy_id` naively would therefore REMOVE the guard from precisely the rows
that need it most: the unattributed ones written when strategy resolution fails.
COALESCE folds them into one bucket that behaves like the old index did.

Reversibility note
------------------
`downgrade()` restores the original index. That can fail if rows exist which are
unique under the new definition but collide under the old one — i.e. two strategies
having recorded the same instrument/bar, which is exactly what this migration
enables. That is correct behaviour: the downgrade is refusing to destroy data, and
the operator must decide which strategy's rows to remove. It is documented here
rather than worked around silently.

Revision ID: 20260911_obs_natural_key
Revises: 20260909_run_dataset_lineage
Create Date: 2026-09-11 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '20260911_obs_natural_key'
down_revision: Union[str, None] = '20260909_run_dataset_lineage'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_OLD = "uq_trades_shadow_natural_key"
_NEW = "uq_trades_observation_natural_key"
_STAGES = "('shadow', 'sandbox')"


def upgrade() -> None:
    # Fail loudly BEFORE dropping the old guard if the new one cannot be built.
    # Creating a unique index over existing duplicates raises anyway, but by then the
    # old index is already gone and the table is briefly unprotected — on a live
    # deployment recording every four hours, "briefly" is long enough to matter.
    conn = op.get_bind()
    dupes = conn.execute(sa.text(f"""
        SELECT count(*) FROM (
            SELECT instrument_id, opened_at, COALESCE(strategy_id, 0)
            FROM trades WHERE stage IN {_STAGES}
            GROUP BY 1, 2, 3 HAVING count(*) > 1
        ) d
    """)).scalar_one()
    if dupes:
        raise RuntimeError(
            f"{dupes} (instrument, opened_at, strategy) group(s) already duplicated in "
            f"stage IN {_STAGES} — resolve them before applying this migration. The "
            f"new unique index cannot be created over existing duplicates."
        )

    op.create_index(
        _NEW,
        "trades",
        ["instrument_id", "opened_at", sa.text("COALESCE(strategy_id, 0)")],
        unique=True,
        postgresql_where=sa.text(f"stage IN {_STAGES}"),
    )
    op.drop_index(_OLD, table_name="trades")


def downgrade() -> None:
    conn = op.get_bind()
    dupes = conn.execute(sa.text("""
        SELECT count(*) FROM (
            SELECT instrument_id, opened_at FROM trades WHERE stage = 'shadow'
            GROUP BY 1, 2 HAVING count(*) > 1
        ) d
    """)).scalar_one()
    if dupes:
        raise RuntimeError(
            f"cannot restore {_OLD}: {dupes} (instrument, opened_at) pair(s) in "
            f"stage='shadow' are held by more than one strategy. Downgrading would "
            f"require deleting one strategy's observations — an operator decision, "
            f"not something this migration will do silently."
        )
    op.create_index(
        _OLD,
        "trades",
        ["instrument_id", "opened_at"],
        unique=True,
        postgresql_where=sa.text("stage = 'shadow'"),
    )
    op.drop_index(_NEW, table_name="trades")
