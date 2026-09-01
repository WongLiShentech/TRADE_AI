"""Export the MINIMUM database corpus a fresh deployment needs, as a portable seed.

Why this exists
---------------
The dev database is ~7.2 GB, of which ~7.18 GB is the ``candles`` table and almost
all of THAT is M1 Bid/Ask intrabar data. Moving it to a small always-on VM over a
home uplink is neither necessary nor sensible: the hourly trailing-window job
(``pipeline.run_trailing_window_refresh``, ``M1_LIVE_LOOKBACK_HOURS``) rebuilds the
live M1 series from empty within one cycle, and the *historical* M1 corpus exists
only to resolve *historical* backtest trades — which the server never re-resolves.

Dropping M1 is therefore the ~99.8% size win, and it costs the deployment nothing.

What IS exported, and why each one is load-bearing
--------------------------------------------------
``instruments``            ALL rows. The universe + ``is_active`` flags + ``pip_size``.
                           Everything else FKs to it, and the app reads the active
                           list ONCE at startup.
``macro_data``             ALL rows. Bitemporal FRED vintages. ``feature_builder``
                           reads them point-in-time; a gap here NaNs model features
                           on every live signal, and the fundamentals cron only
                           back-fills ``FUNDAMENTAL_REFRESH_LOOKBACK_DAYS``.
``news_calendar_events``   ALL rows. The forward calendar spine, including future
                           events the refresh job would not re-derive.
``candles`` H4 Mid         Last N bars per instrument. The decision series. Needs
                           enough history to warm up SMA50/ATR14/RSI14 before the
                           first live signal.
``candles`` D  Mid         Last N bars per instrument. The trend filter + the
                           D1-derived features (``dist_to_sma50_atr``, ``d1_close``,
                           ``d1_sma50``). SMA50 on D1 needs 50+ bars.
``candles`` M1             **NOTHING.** See above.
``indicators`` H4 / D      Last N rows per instrument. Not strictly required (they
                           are recomputable from candles) but shipping them means
                           the first H4 job after deploy can fire a signal instead
                           of spending a cycle recomputing.
``trades``                 Optional, ON by default (``--no-trades`` to skip). The
                           2,809-trade M7 backtest corpus + the S1 training rows,
                           ~12 MB. Include it if you ever want to retrain on the
                           server; skip it for the leanest possible observation-only
                           deployment. Shadow rows recorded on the dev machine come
                           along with it — that is usually what you want, since they
                           are the forward evidence.

Deliberately NOT exported
-------------------------
``alembic_version``  — written by ``alembic upgrade head`` on the target. Seeding it
                       would pin the target to the source's revision even if the
                       target schema differs.
``signals`` / ``orders`` / ``equity_points`` / ``bot_state`` / ``backtest_runs``
                     — operational state of the SOURCE host. A fresh deployment
                       starts its own; ``bot_state`` is created by the app lifespan.

Output format
-------------
One directory (optionally tarred into a single ``.tar.gz``) containing:

    <table>.csv     one CSV per table, header row included
    load.sql        psql script: \\copy in FK-safe order + sequence setval
    LOAD_ORDER.md   the runbook, including the restart requirement

CSV + psql rather than ``pg_dump -Fc`` on purpose: the files are inspectable and
diffable, the load order is explicit and documented rather than implied, and
loading needs nothing but ``psql`` (no version-matched ``pg_restore``).

RUNBOOK ORDER — all three steps, in this order
----------------------------------------------
    1. alembic upgrade head        # schema first; the seed carries no DDL
    2. psql -f load.sql            # data
    3. RESTART THE BACKEND         # ← do not skip

Step 3 is not optional and not cosmetic. ``app.main.lifespan`` reads the active
instrument list EXACTLY ONCE, at startup, to decide which symbols to stream. A
backend that was running while the seed loaded has an empty symbol list, no price
stream, and will sit there looking healthy and doing nothing until it is restarted.

Usage
-----
    # from backend/
    python scripts/export_slim_seed.py
    python scripts/export_slim_seed.py --out ../output/slim_seed --archive
    python scripts/export_slim_seed.py --no-trades --candle-bars H4=2000,D=400
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Ensure backend/ is on sys.path so `app.*` imports resolve when run as a script.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.database import engine  # noqa: E402
from app.domain.timeframes import TIMEFRAMES  # noqa: E402
from app.models.candle import Candle  # noqa: E402
from app.models.indicator import Indicator  # noqa: E402
from app.models.instrument import Instrument  # noqa: E402
from app.models.macro_data import MacroData  # noqa: E402
from app.models.news_calendar_event import NewsCalendarEvent  # noqa: E402
from app.models.trade import Trade  # noqa: E402

# ── defaults (overridable on the command line; nothing here is a magic constant
# buried in business logic — this is an ops script and every number is a CLI arg) ──
#
# Per-instrument bar budgets, keyed by Timeframe.code. Sized against what the app
# actually needs to be immediately productive after a deploy:
#   H4 1000 bars ≈ 166 trading days — far more than SMA50/ATR14 warm-up, and enough
#                  context for the swing-structure lookback.
#   D   200 bars ≈ 9 months — 4× the SMA50(D1) window.
DEFAULT_CANDLE_BARS = {"H4": 1000, "D": 200}
DEFAULT_INDICATOR_ROWS = {"H4": 1000, "D": 60}

# The decision price series. Bid/Ask are execution-only and, on the seeded
# timeframes, are not what indicators or features read.
DECISION_PRICE_TYPE = "M"

_CSV_COPY_OPTIONS = "(FORMAT csv, HEADER true)"


@dataclass(frozen=True)
class TableExport:
    """One CSV in the seed: its table, its columns, and the query that fills it."""

    table: str
    columns: tuple[str, ...]
    query: str
    note: str

    @property
    def filename(self) -> str:
        return f"{self.table}.csv"


def _columns(model) -> tuple[str, ...]:
    """Column names straight off the SQLAlchemy model — never a hand-typed list.

    Naming them explicitly in both the COPY OUT and the \\copy IN makes the seed
    robust to column ORDER differences and to columns added by a later migration on
    the target (they simply take their default / NULL).
    """
    return tuple(c.name for c in model.__table__.columns)


def _full_table_export(model, note: str) -> TableExport:
    cols = _columns(model)
    table = model.__tablename__
    return TableExport(
        table=table,
        columns=cols,
        query=f"SELECT {', '.join(cols)} FROM {table} ORDER BY id",
        note=note,
    )


def _tail_per_instrument_export(
    model, *, table_alias: str, budgets: dict[str, int], price_type: str | None, note: str
) -> TableExport:
    """Newest ``budgets[granularity]`` rows PER (instrument, granularity).

    A window function rather than a date cut-off: instruments do not all have the
    same newest timestamp (a pair can stop updating), and a fixed date would ship a
    different amount of warm-up history for each one.
    """
    cols = _columns(model)
    table = model.__tablename__
    qualified = ", ".join(f"t.{c}" for c in cols)
    price_filter = f"AND price_type = '{price_type}'" if price_type else ""

    branches = []
    for gran, limit in budgets.items():
        branches.append(
            f"""
        SELECT {', '.join(cols)},
               row_number() OVER (PARTITION BY instrument_id ORDER BY timestamp DESC) AS rn,
               {limit} AS budget
        FROM {table}
        WHERE granularity = '{gran}' {price_filter}"""
        )
    union = "\n        UNION ALL".join(branches)
    return TableExport(
        table=table_alias,
        columns=cols,
        query=f"SELECT {qualified} FROM ({union}\n        ) t WHERE t.rn <= t.budget ORDER BY t.id",
        note=note,
    )


def build_exports(
    candle_bars: dict[str, int], indicator_rows: dict[str, int], include_trades: bool
) -> list[TableExport]:
    """Assemble the export set. Order IS the FK-safe load order."""
    exports = [
        _full_table_export(
            Instrument,
            "ALL rows — the universe, is_active flags and pip_size everything else FKs to.",
        ),
        _full_table_export(
            MacroData,
            "ALL rows — bitemporal FRED vintages read point-in-time by feature_builder.",
        ),
        _full_table_export(
            NewsCalendarEvent,
            "ALL rows — the forward calendar spine, including future-dated events.",
        ),
        _tail_per_instrument_export(
            Candle,
            table_alias=Candle.__tablename__,
            budgets=candle_bars,
            price_type=DECISION_PRICE_TYPE,
            note=(
                f"price_type='{DECISION_PRICE_TYPE}' only, newest "
                + ", ".join(f"{n} {g}" for g, n in candle_bars.items())
                + " bars per instrument. NO M1 ROWS — the hourly trailing-window job "
                "rebuilds the live M1 series from empty (this is the ~99.8% size win)."
            ),
        ),
        _tail_per_instrument_export(
            Indicator,
            table_alias=Indicator.__tablename__,
            budgets=indicator_rows,
            price_type=None,
            note=(
                "newest "
                + ", ".join(f"{n} {g}" for g, n in indicator_rows.items())
                + " rows per instrument — recomputable, shipped so the first live "
                "candle-close job can fire immediately."
            ),
        ),
    ]
    if include_trades:
        exports.append(
            _full_table_export(
                Trade,
                "ALL rows — M7 backtest corpus (S1 training set) + any recorded shadow "
                "rows. ~12 MB. Omit with --no-trades for an observation-only deploy.",
            )
        )
    return exports


def _assert_seedable_timeframes(budgets: dict[str, int], what: str) -> None:
    """Fail loud on a granularity that is not in the Timeframe registry, or is M1.

    M1 is identified by its registry entry (``trailing_window_setting`` is set), not
    by its name — so if a future timeframe becomes trailing-window managed it is
    excluded automatically, with no edit here.
    """
    for gran in budgets:
        tf = TIMEFRAMES.get(gran)
        if tf is None:
            raise ValueError(
                f"{what}: '{gran}' is not in the Timeframe registry "
                f"(known: {', '.join(TIMEFRAMES)})"
            )
        if tf.trailing_window_setting is not None:
            raise ValueError(
                f"{what}: refusing to seed '{gran}' — it is a trailing-window timeframe "
                f"({tf.trailing_window_setting}) that the scheduled job rebuilds from "
                f"empty. Seeding it is the entire 7 GB this script exists to avoid."
            )


def export(out_dir: Path, exports: list[TableExport]) -> dict[str, int]:
    """Run every COPY, writing one CSV per export. Returns {filename: bytes}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}

    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            for spec in exports:
                path = out_dir / spec.filename
                sql = f"COPY ({spec.query}) TO STDOUT WITH {_CSV_COPY_OPTIONS}"
                with path.open("wb") as fh:
                    cur.copy_expert(sql, fh)
                sizes[spec.filename] = path.stat().st_size
                print(f"  {spec.filename:<28} {_human(sizes[spec.filename]):>10}")
    finally:
        raw.close()
    return sizes


def _row_count(path: Path) -> int:
    """Data rows in a CSV (total lines minus the header).

    Counts newlines rather than parsing: JSON columns in ``trades`` contain quoted
    embedded newlines, so this is an UPPER bound, reported as such.
    """
    with path.open("rb") as fh:
        lines = sum(1 for _ in fh)
    return max(lines - 1, 0)


def write_load_sql(out_dir: Path, exports: list[TableExport]) -> Path:
    """Emit the psql loader: \\copy in FK order, then fix every id sequence."""
    lines = [
        "-- ══════════════════════════════════════════════════════════════════════",
        "-- Slim seed loader — generated by backend/scripts/export_slim_seed.py",
        f"-- Generated: {datetime.now(timezone.utc).isoformat()}",
        "--",
        "-- PREREQUISITE: run `alembic upgrade head` on the target FIRST. This file",
        "-- carries DATA ONLY — no DDL. Loading into a database without the schema",
        "-- fails on the first \\copy.",
        "--",
        "-- Run from INSIDE the directory holding the .csv files (\\copy resolves",
        "-- paths client-side, relative to psql's working directory):",
        "--",
        "--     psql -v ON_ERROR_STOP=1 -h <host> -U <user> -d <db> -f load.sql",
        "--",
        "-- ON_ERROR_STOP=1 matters: without it psql reports failures and keeps",
        "-- going, leaving a half-loaded database that looks like it worked.",
        "--",
        "-- Target tables must be EMPTY. There is no TRUNCATE here on purpose —",
        "-- a seed loader that silently wipes tables is one fat-finger away from",
        "-- destroying a live corpus. A primary-key collision aborting the run is",
        "-- the safe failure.",
        "-- ══════════════════════════════════════════════════════════════════════",
        "",
        "BEGIN;",
        "",
    ]
    for spec in exports:
        lines += [
            f"-- {spec.table}: {spec.note}",
            f"\\copy {spec.table} ({', '.join(spec.columns)}) "
            f"FROM '{spec.filename}' WITH {_CSV_COPY_OPTIONS}",
            "",
        ]

    lines += [
        "-- ── sequence repair ──────────────────────────────────────────────────",
        "-- \\copy writes explicit id values, which does NOT advance the identity",
        "-- sequence. Without this block the next INSERT reuses id=1 and dies on a",
        "-- duplicate-key error — and it dies inside a scheduled job on the server,",
        "-- hours after the deploy looked successful.",
        "-- coalesce(max(id), 0) + 1 with is_called=false handles an empty table.",
        "",
    ]
    for spec in exports:
        if "id" not in spec.columns:
            continue
        lines.append(
            f"SELECT setval(pg_get_serial_sequence('{spec.table}', 'id'), "
            f"coalesce((SELECT max(id) FROM {spec.table}), 0) + 1, false);"
        )
    lines += ["", "COMMIT;", ""]

    path = out_dir / "load.sql"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_runbook(
    out_dir: Path, exports: list[TableExport], sizes: dict[str, int], include_trades: bool
) -> Path:
    """Emit LOAD_ORDER.md — the human runbook that ships inside the artifact."""
    total = sum(sizes.values())
    rows = "\n".join(
        f"| `{s.table}` | {_row_count(out_dir / s.filename):,} | "
        f"{_human(sizes[s.filename])} | {s.note} |"
        for s in exports
    )
    trades_line = (
        "`trades` IS included (`--no-trades` to omit)."
        if include_trades
        else "`trades` is NOT included (exported with `--no-trades`)."
    )
    content = f"""# Slim seed — load runbook

Generated {datetime.now(timezone.utc).isoformat()} by `backend/scripts/export_slim_seed.py`.

Total artifact size: **{_human(total)}** across {len(exports)} CSVs. {trades_line}

## Contents

| table | rows (approx) | csv size | what it is |
|---|---:|---:|---|
{rows}

**No `candles` M1 rows.** That omission is the point of this artifact: M1 Bid/Ask is
~99.8% of the source database and the hourly trailing-window job rebuilds the live
window from empty within one cycle.

**No `alembic_version`, `signals`, `orders`, `equity_points`, `bot_state` or
`backtest_runs`.** Schema revision comes from `alembic upgrade head`; the rest is
operational state belonging to the source host.

## Load order — all three steps, in this order

```bash
# 1. SCHEMA FIRST — the seed carries data only, no DDL.
cd <repo>/backend
alembic upgrade head

# 2. DATA — run from inside this directory (\\copy resolves paths client-side).
cd <this directory>
psql -v ON_ERROR_STOP=1 -h <host> -U <user> -d <db> -f load.sql

# 3. RESTART THE BACKEND.
docker compose -f docker-compose.prod.yml restart backend
```

### Step 3 is not optional

`app.main.lifespan` reads the active-instrument list **exactly once, at startup**, to
decide which symbols to subscribe the price stream to. A backend that was already
running when the seed landed holds an empty symbol list: no stream, no ticks, and
`/health` reporting `price_stream: ok — intentionally not started`. It will sit there
looking fine and doing nothing until it is restarted.

### Loading into compose

`db` publishes no port in `docker-compose.prod.yml`, so reach it through the
container:

```bash
docker compose -f docker-compose.prod.yml cp <this directory> db:/tmp/seed
docker compose -f docker-compose.prod.yml exec db \\
    sh -c "cd /tmp/seed && psql -v ON_ERROR_STOP=1 -U <user> -d <db> -f load.sql"
```

## Verifying the load

```sql
SELECT count(*) FROM instruments WHERE is_active;          -- expect the live universe
SELECT granularity, price_type, count(*) FROM candles GROUP BY 1, 2;   -- expect NO M1
SELECT max(release_time) FROM macro_data;                  -- expect ~the export date
```

Then confirm the app agrees:

```bash
curl -s localhost:8000/health | python -m json.tool
```

`database: ok`, `ml_model: ok`, and — after the restart, during market hours —
`price_stream: ok`.
"""
    path = out_dir / "LOAD_ORDER.md"
    path.write_text(content, encoding="utf-8")
    return path


def archive(out_dir: Path) -> Path:
    """Tar+gzip the seed directory into a single portable file beside it."""
    target = out_dir.with_suffix(".tar.gz")
    with tarfile.open(target, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)
    return target


def _human(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _parse_budgets(raw: str, what: str) -> dict[str, int]:
    """Parse ``H4=1000,D=200`` into ``{'H4': 1000, 'D': 200}``."""
    budgets: dict[str, int] = {}
    for part in (p.strip() for p in raw.split(",") if p.strip()):
        if "=" not in part:
            raise ValueError(f"{what}: expected GRANULARITY=N, got {part!r}")
        gran, _, count = part.partition("=")
        budgets[gran.strip()] = int(count)
    if not budgets:
        raise ValueError(f"{what}: no budgets parsed from {raw!r}")
    return budgets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export the minimum DB corpus for a fresh deployment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=BACKEND_ROOT.parent / "output" / "slim_seed",
        help="destination directory for the CSVs + load.sql + LOAD_ORDER.md",
    )
    parser.add_argument(
        "--candle-bars",
        default=",".join(f"{g}={n}" for g, n in DEFAULT_CANDLE_BARS.items()),
        help="per-instrument candle budget, GRANULARITY=N[,GRANULARITY=N]",
    )
    parser.add_argument(
        "--indicator-rows",
        default=",".join(f"{g}={n}" for g, n in DEFAULT_INDICATOR_ROWS.items()),
        help="per-instrument indicator budget, GRANULARITY=N[,GRANULARITY=N]",
    )
    parser.add_argument(
        "--no-trades",
        action="store_true",
        help="omit the trades table (~12 MB); keep it if you may retrain on the server",
    )
    parser.add_argument(
        "--archive", action="store_true", help="also produce a single .tar.gz beside --out"
    )
    args = parser.parse_args(argv)

    candle_bars = _parse_budgets(args.candle_bars, "--candle-bars")
    indicator_rows = _parse_budgets(args.indicator_rows, "--indicator-rows")
    _assert_seedable_timeframes(candle_bars, "--candle-bars")
    _assert_seedable_timeframes(indicator_rows, "--indicator-rows")

    exports = build_exports(candle_bars, indicator_rows, include_trades=not args.no_trades)

    print(f"exporting slim seed -> {args.out}")
    sizes = export(args.out, exports)
    load_sql = write_load_sql(args.out, exports)
    runbook = write_runbook(args.out, exports, sizes, include_trades=not args.no_trades)

    print(f"  {load_sql.name:<28} {_human(load_sql.stat().st_size):>10}")
    print(f"  {runbook.name:<28} {_human(runbook.stat().st_size):>10}")
    print(f"TOTAL {_human(sum(sizes.values()))} (csv only)")

    if args.archive:
        tar_path = archive(args.out)
        print(f"archive: {tar_path} ({_human(tar_path.stat().st_size)})")

    print(
        "\nRUNBOOK: 1) alembic upgrade head  2) psql -f load.sql  "
        "3) RESTART THE BACKEND (active instruments are read once, at startup)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
