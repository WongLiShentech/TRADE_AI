"""Shared pytest fixtures — plus the live-database guard.

These are integration tests: they read a real Postgres via the app's own
``SessionLocal``, exactly as the M7 runner and the live pipeline do. No mocks —
the point is to verify PIT/leakage behaviour against real staged data.

────────────────────────────────────────────────────────────────────────────────
LIVE-DATABASE GUARD
────────────────────────────────────────────────────────────────────────────────
Because the fixtures use the app's own ``SessionLocal``, a bare ``pytest`` run
points straight at whatever ``DATABASE_URL`` says — which on a developer machine
is the LIVE dev database currently recording M8-Shadow observations. Several
tests in this suite create and DELETE rows (a QA pass previously caught fixtures
deleting live data). One careless run is enough to destroy forward evidence that
takes weeks of real market time to regenerate and cannot be recreated.

So the session refuses to start when the effective ``DATABASE_URL`` resolves to
the same database name as ``backend/.env`` — that file IS, by definition, the
live one on this host. Nothing is hardcoded: the protected name is read from
config, not written down here.

Opt out only when you genuinely mean it::

    ALLOW_TESTS_ON_LIVE_DB=1 python -m pytest

The recommended path is a throwaway database instead — see the refusal message,
which prints the exact commands.
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import pytest
from dotenv import dotenv_values

from app.config import get_settings
from app.database import SessionLocal
from app.models.instrument import Instrument

# Env var a caller sets to accept the risk deliberately. Any non-empty value that
# is not a recognised "false" spelling counts as consent.
ALLOW_LIVE_DB_ENV = "ALLOW_TESTS_ON_LIVE_DB"
_FALSEY = {"", "0", "false", "no", "off"}

# backend/ — the .env beside it defines what "live" means on this host.
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_ENV_FILE = _BACKEND_ROOT / ".env"


def _database_name(url: str | None) -> str | None:
    """Extract the database name from a SQLAlchemy URL, or None if unparseable."""
    if not url:
        return None
    parsed = urlparse(url)
    name = (parsed.path or "").lstrip("/")
    return name or None


def _live_database_name() -> str | None:
    """The database name declared in backend/.env — the live one on this machine.

    Read straight from the FILE rather than from ``get_settings()``: an environment
    variable overrides the file, so once a caller has pointed DATABASE_URL at a
    scratch DB, settings no longer knows what the live one was.
    """
    if not _ENV_FILE.exists():
        return None
    return _database_name(dotenv_values(_ENV_FILE).get("DATABASE_URL"))


def _opted_in() -> bool:
    return os.environ.get(ALLOW_LIVE_DB_ENV, "").strip().lower() not in _FALSEY


def pytest_configure(config: pytest.Config) -> None:
    """Abort the whole session before a single fixture can touch the live database."""
    if _opted_in():
        return

    effective_url = get_settings().DATABASE_URL
    effective_db = _database_name(effective_url)
    live_db = _live_database_name()

    if live_db is None or effective_db is None or effective_db != live_db:
        return

    parsed = urlparse(effective_url)
    host = parsed.hostname or "(unknown host)"
    port = parsed.port or 5432

    pytest.exit(
        "\n"
        "=======================================================================\n"
        " REFUSING TO RUN: tests are pointed at the LIVE database\n"
        "=======================================================================\n"
        f"  effective DATABASE_URL -> database '{effective_db}' on {host}:{port}\n"
        f"  backend/.env declares  -> database '{live_db}'  (i.e. the live one)\n"
        "\n"
        "  This suite is integration-level: fixtures INSERT and DELETE rows via the\n"
        "  app's own SessionLocal. Running it here can destroy live M8-Shadow\n"
        "  observations, which are forward evidence that only real market time can\n"
        "  regenerate.\n"
        "\n"
        "  Run against a throwaway database instead (from backend/):\n"
        "\n"
        f"    createdb -h {host} -p {port} -U <user> {live_db}_test\n"
        f"    DATABASE_URL=postgresql://<user>:<pw>@{host}:{port}/{live_db}_test \\\n"
        "        alembic upgrade head\n"
        f"    DATABASE_URL=postgresql://<user>:<pw>@{host}:{port}/{live_db}_test \\\n"
        "        python -m pytest\n"
        "\n"
        "  (Data-dependent tests need a seeded DB — see\n"
        "   scripts/export_slim_seed.py to copy the minimum corpus across.)\n"
        "\n"
        f"  To override deliberately:  {ALLOW_LIVE_DB_ENV}=1 python -m pytest\n"
        "=======================================================================\n",
        returncode=pytest.ExitCode.USAGE_ERROR,
    )


@pytest.fixture()
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def settings():
    return get_settings()


@pytest.fixture()
def instrument(db):
    def _load(symbol: str) -> Instrument:
        inst = db.query(Instrument).filter_by(symbol=symbol).first()
        assert inst is not None, f"instrument {symbol} not found — run /instruments/sync"
        return inst

    return _load
