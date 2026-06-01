"""
One-shot migration: copies all rows from SQLite → Postgres.

Run AFTER:
  1. docker compose up -d db
  2. alembic upgrade head   (creates the schema in Postgres first)

Run FROM the backend/ directory:
  python scripts/migrate_sqlite_to_postgres.py
"""

import os
import sys

import pandas as pd
from sqlalchemy import create_engine, inspect, text

SQLITE_URL = "sqlite:///./data/trading.db"
PG_URL = os.environ.get(
    "DATABASE_URL", "postgresql://trader:trader@localhost:5432/trading"
)

TABLES_IN_ORDER = [
    "instruments",
    "candles",
    "equity_points",
    "signals",
    "trades",
    "backtest_runs",
]

BATCH_SIZE = 1_000


def migrate() -> None:
    print(f"Source : {SQLITE_URL}")
    print(f"Target : {PG_URL}\n")

    sqlite = create_engine(SQLITE_URL)
    pg = create_engine(PG_URL)

    sqlite_tables = set(inspect(sqlite).get_table_names())
    pg_tables = set(inspect(pg).get_table_names())

    for table in TABLES_IN_ORDER:
        if table not in sqlite_tables:
            print(f"  {table}: not in SQLite, skipping")
            continue
        if table not in pg_tables:
            print(f"  {table}: not in Postgres (run alembic upgrade head first), skipping")
            continue

        df = pd.read_sql(f"SELECT * FROM {table}", sqlite)  # noqa: S608
        sqlite_count = len(df)

        if sqlite_count == 0:
            print(f"  {table}: 0 rows in SQLite, nothing to copy")
            continue

        with pg.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))

        for start in range(0, sqlite_count, BATCH_SIZE):
            batch = df.iloc[start : start + BATCH_SIZE]
            batch.to_sql(table, pg, if_exists="append", index=False)

        with pg.connect() as conn:
            pg_count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()

        status = "OK" if pg_count == sqlite_count else "MISMATCH"
        print(f"  {table}: {sqlite_count} rows -> Postgres {pg_count} rows  [{status}]")
        if status == "MISMATCH":
            print(f"    WARNING: counts differ — investigate before proceeding")
            sys.exit(1)

    print("\nMigration complete.")


if __name__ == "__main__":
    migrate()
