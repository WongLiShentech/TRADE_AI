"""Register a parameterised signal configuration as a strategy.

Why this exists
---------------
``run_strategy2_backtest.py`` registered strategy 2 by setting an environment variable
before importing Settings, hardcoding the name and description in the module, and
importing the identity helpers out of ``backfill_attribution``. That worked once. Doing
it again per strategy means a new near-duplicate script each time, and every copy is a
chance for the identity rule to drift — at which point the same configuration hashes two
ways and its evidence silently splits in half.

Identity is the parameter set, not the label
--------------------------------------------
``params_hash`` digests the engine plus the parameters that affect SIGNAL GENERATION.
Same configuration ⇒ same id forever; any difference ⇒ a different strategy. So this
script does not let you name a strategy into existence: it takes a base configuration
plus explicit overrides, hashes the result, and refuses if that hash already exists.

Types matter to the hash
------------------------
``params_hash`` serialises with ``default=str``, so the integer 2 and the string "2"
digest differently. Overrides are therefore coerced to the type ``Settings`` declares for
that field before hashing — otherwise a strategy registered here would never match the
one ``resolve_active_strategy`` computes from the running config, and the live rows would
attribute to a different strategy than the backtest ones.

Status
------
Registered as ``research``: a strategy must not start consuming live signals merely
because it exists in the table. Promoting it to ``shadow`` is a deliberate, separate act
(see ``active_strategies``).

Run from ``backend/``:
    python scripts/register_strategy.py --name rule_based_v3_loose \\
        --from-strategy 2 --set SIGNAL_MIN_CONFLUENCE_SCORE=2 \\
        --description "..." --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.config import Settings, get_settings
from app.database import SessionLocal
from app.models.strategy import Strategy
from app.services.strategy_registry import (
    ENGINE,
    IDENTITY_PARAMS,
    identity_params,
    params_hash,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("register_strategy")

STATUS_RESEARCH = "research"


def _coerce(key: str, raw: str):
    """Cast an override to the type ``Settings`` declares for that field.

    The hash is type-sensitive (see module docstring), so this is correctness, not
    convenience: an uncoerced "2" would produce a strategy the live path can never match.
    """
    field = Settings.model_fields.get(key)
    if field is None:
        raise SystemExit(f"--set {key}: no such Settings field")
    annotation = field.annotation
    try:
        return annotation(raw)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"--set {key}={raw!r}: cannot coerce to {annotation} ({exc})")


def _base_params(db, settings: Settings, from_strategy: int | None) -> dict:
    """Identity params to start from — another strategy's, or the running config's."""
    if from_strategy is None:
        return identity_params(settings)
    parent = db.get(Strategy, from_strategy)
    if parent is None:
        raise SystemExit(f"--from-strategy {from_strategy}: no such strategy")
    base = {k: v for k, v in (parent.params or {}).items() if k in IDENTITY_PARAMS}
    missing = [k for k in IDENTITY_PARAMS if k not in base]
    if missing:
        # Fall back for keys the parent row does not carry, so the derived strategy is
        # still fully described rather than inheriting a hole.
        live = identity_params(settings)
        for k in missing:
            base[k] = live[k]
        logger.warning(
            "strategy %s lacks %s — filled from the running config", from_strategy, missing
        )
    return base


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--name", required=True, help="human label; does NOT affect identity")
    ap.add_argument("--description", required=True, help="WHAT this strategy is (stable)")
    ap.add_argument("--notes", default="", help="WHAT HAPPENED to it (ever-growing)")
    ap.add_argument(
        "--from-strategy", type=int, default=None,
        help="strategy id whose params are the base; default: the running config",
    )
    ap.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="override an identity parameter; repeatable",
    )
    ap.add_argument("--dry-run", action="store_true", help="show the hash and params, write nothing")
    args = ap.parse_args()

    settings = get_settings()
    db = SessionLocal()
    try:
        params = _base_params(db, settings, args.from_strategy)

        changed = {}
        for item in args.set:
            if "=" not in item:
                raise SystemExit(f"--set {item!r}: expected KEY=VALUE")
            key, raw = item.split("=", 1)
            key = key.strip()
            if key not in IDENTITY_PARAMS:
                raise SystemExit(
                    f"--set {key}: not an identity parameter. Only {list(IDENTITY_PARAMS)} "
                    f"affect which signals fire, so only those define a strategy."
                )
            value = _coerce(key, raw.strip())
            changed[key] = (params.get(key), value)
            params[key] = value

        phash = params_hash(ENGINE, params)

        print(f"\nname        {args.name}")
        print(f"engine      {ENGINE}")
        print(f"params_hash {phash}")
        if changed:
            print("changed")
            for k, (old, new) in changed.items():
                print(f"  {k}: {old!r} -> {new!r}")
        else:
            print("changed     (nothing — this is the base configuration)")

        existing = db.query(Strategy).filter_by(params_hash=phash).first()
        if existing is not None:
            raise SystemExit(
                f"\nthis configuration is ALREADY registered as strategy {existing.id} "
                f"({existing.name}, status={existing.status}). Identity is the parameter "
                f"set, not the name — registering it again under a new name would split "
                f"one strategy's evidence across two rows."
            )

        if args.dry_run:
            print("\n(--dry-run: nothing written)")
            return

        strategy = Strategy(
            name=args.name,
            engine=ENGINE,
            params_hash=phash,
            params=params,
            status=STATUS_RESEARCH,
            description=args.description,
            notes=args.notes,
        )
        db.add(strategy)
        db.commit()
        db.refresh(strategy)
        print(f"\nregistered as strategy {strategy.id} (status={STATUS_RESEARCH})")
        print(
            "Next: backtest it (scripts/backtest_strategy.py --strategy-id "
            f"{strategy.id}), then set status='shadow' when you want it evaluating live bars."
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
