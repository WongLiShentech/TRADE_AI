"""Backtest a registered strategy, driven by its own stored parameters.

Why this exists
---------------
``run_strategy2_backtest.py`` set ``os.environ["BACKTEST_TRAILING_LOCK_PCT"]`` before
importing Settings, because Settings is built once and cached. That works, but it means
a per-strategy script per strategy, each poking a different env var, each a chance to
drive the runner with parameters that do not match the row the results get attributed to
— which is the exact failure ``backtest_runs.strategy`` already had once.

``strategy_registry.settings_for`` exists precisely to avoid that: it projects a
strategy's stored ``params`` onto a Settings copy, applying only keys in
``IDENTITY_PARAMS``. The runner then reads every threshold and window off that object as
usual, with no knowledge that strategies exist. One mechanism, and the parameters that
run are by construction the ones the strategy row states.

Re-running
----------
A second backtest of one strategy is a different RUN of the same configuration, and
``trades.run_id`` distinguishes them. But two corpora for one strategy would make an
unscoped ``load_dataset(strategy_id=N)`` return both, mixing a pre-fix and post-fix
grading of the same entries — so ``--force`` deletes the prior rows rather than
accumulating. Without it, an existing corpus is refused.

Run from ``backend/``:
    python scripts/backtest_strategy.py --strategy-id 3 --dry-run
    python scripts/backtest_strategy.py --strategy-id 3
    python scripts/backtest_strategy.py --strategy-id 3 --force
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings
from app.database import SessionLocal
from app.models.strategy import Strategy
from app.models.trade import Trade
from app.services.backtester.runner import run_backtest
from app.services.strategy_registry import IDENTITY_PARAMS, settings_for

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("backtest_strategy")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--strategy-id", type=int, required=True)
    ap.add_argument(
        "--instruments", default=None,
        help="comma-separated symbols; default: every is_active instrument",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="delete this strategy's existing backtest rows first (see module docstring)",
    )
    ap.add_argument("--dry-run", action="store_true", help="show what would run, execute nothing")
    args = ap.parse_args()

    settings = get_settings()
    db = SessionLocal()
    try:
        strategy = db.get(Strategy, args.strategy_id)
        if strategy is None:
            raise SystemExit(f"no strategy with id {args.strategy_id}")

        s_settings = settings_for(settings, strategy)
        existing = (
            db.query(Trade)
            .filter(Trade.strategy_id == strategy.id, Trade.stage == "backtest")
            .count()
        )

        print(f"\nstrategy    {strategy.id} {strategy.name} (status={strategy.status})")
        print(f"params_hash {strategy.params_hash}")
        print("parameters the runner will use")
        for k in IDENTITY_PARAMS:
            print(f"  {k:<42} {getattr(s_settings, k, '(absent)')!r}")
        print(f"existing backtest rows for this strategy: {existing}")

        if existing and not args.force:
            raise SystemExit(
                f"\nstrategy {strategy.id} already has {existing} backtest rows. Two corpora "
                f"for one strategy would be read as a single mixed one by load_dataset. "
                f"Pass --force to delete them and rebuild."
            )

        if args.dry_run:
            print("\n(--dry-run: nothing executed)")
            return

        if existing and args.force:
            logger.warning(
                "--force: deleting %d existing backtest rows for strategy %s", existing, strategy.id
            )
            db.query(Trade).filter(
                Trade.strategy_id == strategy.id, Trade.stage == "backtest"
            ).delete(synchronize_session=False)
            db.commit()

        instruments = (
            [s.strip() for s in args.instruments.split(",") if s.strip()]
            if args.instruments else None
        )

        started = time.time()
        logger.info("running backtest for strategy %s (%s)...", strategy.id, strategy.name)
        result = run_backtest(db, s_settings, instruments, strategy_id=strategy.id)
        elapsed = time.time() - started

        rows = (
            db.query(Trade)
            .filter(Trade.strategy_id == strategy.id, Trade.stage == "backtest")
            .count()
        )
        print(f"\ncompleted in {elapsed / 60:.1f} min — {rows} backtest rows")
        print(json.dumps(
            {k: v for k, v in (result or {}).items() if not isinstance(v, (dict, list))},
            indent=2, default=str,
        ))
    finally:
        db.close()


if __name__ == "__main__":
    main()
