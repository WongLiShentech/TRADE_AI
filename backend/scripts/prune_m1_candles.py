"""OPTIONAL retention prune for trailing-window (M1) candles.

NOT SCHEDULED. Nothing calls this. It exists because an always-on deployment
accumulates intrabar candles forever and a Raspberry Pi's microSD card does not
have "forever" of space — measured growth is ~122 MB/month with the current
universe and ``M1_LIVE_LOOKBACK_HOURS``. On a 12 GB Oracle A1 with block storage
you will probably never need it; on a 32 GB card it is the difference between a
deployment that runs for years and one that fills up.

It is deliberately a manual tool, not a cron job. Deleting the intrabar series is
the one operation that can silently destroy the ability to HONESTLY resolve a
shadow row — the outcome resolver walks M1 Bid/Ask to decide whether a trade hit
its stop or its target and in what order (see services/shadow/resolver.py). Prune
too aggressively and rows resolve `forced` + ``ambiguous_resolution=True``, which
is not a crash, not a log line anyone will read, and permanently contaminates the
forward-evidence corpus. A human should decide when that risk is worth taking.

WHAT IS SAFE TO DELETE
----------------------
A pending shadow row is resolvable only until ``T + fetch_cap_hours`` (the
simulator's own closure-proof cap — ``(SIGNAL_MAX_HOLD_BARS + 1) x period_hours x
closure_cap_multiplier``, ~7 days on H4). Past that the resolver force-resolves it
regardless of coverage. So M1 older than:

    fetch_cap_hours(signal timeframe)  +  M1_LIVE_LOOKBACK_HOURS  +  safety margin

can no longer affect any resolution that has not already happened. The retention
floor is DERIVED from those settings, never written down as a number here.

The script additionally refuses to delete anything newer than the OLDEST pending
(``closed_at IS NULL``) shadow row's horizon — a real, data-driven guard that
survives a mis-set margin.

DO NOT RUN THIS ON THE DEVELOPMENT MACHINE
------------------------------------------
Read this before typing ``--apply`` anywhere other than a server.

The dev database holds the ~36.7M-row HISTORICAL M1 Bid/Ask corpus that the M7
backtest was resolved against (back to the start of the execution window). That
corpus is not "live data the job will refetch" — it is the evidence base for the
2,809-trade training set, it took a very long operational backfill to build, and
essentially ALL of it is older than any retention horizon this script computes. A
dry run on the dev machine reports ~36.69M of 36.71M rows as prunable, and it is
right: nothing PENDING needs them. They are still the thing you least want to lose.

This script is written for a SERVER, where the slim seed deliberately ships zero M1
rows and every M1 row present was accumulated live by the hourly job. There it is
safe. Everywhere else, run the dry run, read the row count, and stop.

WHAT IT WILL NOT TOUCH
----------------------
* Any timeframe without a ``trailing_window_setting`` in the Timeframe registry.
  The decision series (H4) and the trend series (D1) are small, irreplaceable
  without a broker backfill, and warm up every indicator. Identified from the
  registry, so a future trailing-window timeframe is covered with no edit here.
* Anything at all, unless ``--apply`` is passed. Default is a DRY RUN that only
  counts.

USAGE
-----
    # from backend/ — dry run, prints what WOULD go
    python scripts/prune_m1_candles.py

    # keep an extra 14 days beyond the derived floor, then actually delete
    python scripts/prune_m1_candles.py --margin-days 14 --apply

    # inside compose
    docker compose -f docker-compose.prod.yml exec backend \\
        python scripts/prune_m1_candles.py --margin-days 14 --apply

After a large delete, reclaim the space (plain DELETE only marks rows dead):

    docker compose -f docker-compose.prod.yml exec db \\
        psql -U <user> -d <db> -c "VACUUM (ANALYZE) candles;"

VACUUM FULL would reclaim more but takes an ACCESS EXCLUSIVE lock and rewrites the
whole table — on a Pi that is a long outage and a large burst of card writes. Plain
VACUUM returns the space to Postgres for reuse, which is what you actually want.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import func  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.domain.timeframes import TIMEFRAMES  # noqa: E402
from app.models.candle import Candle  # noqa: E402
from app.models.trade import Trade  # noqa: E402
from app.services.backtester.simulator import horizon_bounds  # noqa: E402
from app.services.shadow.recorder import STAGE_SHADOW  # noqa: E402


def _prunable_codes() -> list[str]:
    """Timeframe codes the scheduled job rebuilds from a trailing window.

    Registry-driven: a timeframe is prunable exactly when the platform already
    treats it as disposable-and-refetchable. No timeframe code appears literally.
    """
    return [tf.code for tf in TIMEFRAMES.values() if tf.trailing_window_setting is not None]


def _signal_codes() -> list[str]:
    """Timeframes that author signals — whose horizons define what M1 must cover."""
    return [tf.code for tf in TIMEFRAMES.values() if tf.fires_signals]


def retention_cutoff(settings, margin_days: float, now: datetime) -> tuple[datetime, str]:
    """Oldest M1 timestamp worth keeping, and the arithmetic that produced it."""
    caps = {code: horizon_bounds(settings, code)[1] for code in _signal_codes()}
    if not caps:
        raise RuntimeError(
            "no signal-firing timeframe in the registry — cannot derive a retention "
            "horizon; refusing to guess"
        )
    worst_code, worst_cap = max(caps.items(), key=lambda kv: kv[1])
    lookback = float(settings.M1_LIVE_LOOKBACK_HOURS)
    total_hours = worst_cap + lookback + margin_days * 24.0
    cutoff = now - timedelta(hours=total_hours)
    explain = (
        f"fetch_cap({worst_code})={worst_cap:.0f}h + M1_LIVE_LOOKBACK_HOURS={lookback:.0f}h "
        f"+ margin={margin_days}d({margin_days * 24:.0f}h) = {total_hours:.0f}h "
        f"({total_hours / 24:.1f}d)"
    )
    return cutoff, explain


def pending_floor(db, settings) -> tuple[datetime | None, int]:
    """Earliest M1 timestamp any UNRESOLVED shadow row could still need.

    Data-driven backstop: whatever ``--margin-days`` says, never delete inside the
    window an actually-pending row will be resolved against.
    """
    pending = (
        db.query(Trade)
        .filter(Trade.stage == STAGE_SHADOW, Trade.closed_at.is_(None))
        .order_by(Trade.opened_at.asc())
        .first()
    )
    count = (
        db.query(func.count(Trade.id))
        .filter(Trade.stage == STAGE_SHADOW, Trade.closed_at.is_(None))
        .scalar()
        or 0
    )
    if pending is None or pending.opened_at is None:
        return None, count
    # Start of the window the resolver reads for that row: its entry instant, minus
    # the trailing lookback that keeps the stream contiguous around it.
    return pending.opened_at - timedelta(hours=float(settings.M1_LIVE_LOOKBACK_HOURS)), count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prune trailing-window (M1) candles older than the resolver horizon.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--margin-days",
        type=float,
        default=14.0,
        help="extra days kept BEYOND the derived resolver horizon",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually DELETE. Without this the script only counts (dry run).",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    now = datetime.now(timezone.utc).replace(tzinfo=None)  # candles.timestamp is naive UTC
    codes = _prunable_codes()
    if not codes:
        print("no trailing-window timeframe in the registry - nothing is prunable")
        return 0

    cutoff, explain = retention_cutoff(settings, args.margin_days, now)
    db = SessionLocal()
    try:
        floor, n_pending = pending_floor(db, settings)
        if floor is not None and floor < cutoff:
            print(
                f"NOTE: {n_pending} pending shadow row(s); the oldest needs M1 from "
                f"{floor:%Y-%m-%d %H:%M}, which is older than the derived cutoff. "
                f"Clamping to protect it."
            )
            cutoff = floor

        print(f"retention horizon : {explain}")
        print(f"cutoff            : {cutoff:%Y-%m-%d %H:%M} UTC")
        print(f"prunable granularities: {', '.join(codes)}")
        print(f"pending shadow rows   : {n_pending}")

        q = db.query(Candle).filter(
            Candle.granularity.in_(codes), Candle.timestamp < cutoff
        )
        doomed = q.with_entities(func.count(Candle.id)).scalar() or 0
        oldest = (
            db.query(func.min(Candle.timestamp))
            .filter(Candle.granularity.in_(codes))
            .scalar()
        )
        total = (
            db.query(func.count(Candle.id))
            .filter(Candle.granularity.in_(codes))
            .scalar()
            or 0
        )
        print(f"\n{doomed:,} of {total:,} rows are older than the cutoff "
              f"(oldest stored: {oldest})")

        if not args.apply:
            print("\nDRY RUN - nothing deleted. Re-run with --apply to delete.")
            return 0
        if doomed == 0:
            print("\nnothing to delete")
            return 0

        deleted = q.delete(synchronize_session=False)
        db.commit()
        print(f"\ndeleted {deleted:,} rows")
        print("run VACUUM (ANALYZE) candles; to return the space to Postgres")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
