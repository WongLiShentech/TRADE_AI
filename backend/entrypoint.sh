#!/bin/sh
#
# Container entrypoint — migrate, then serve.
#
# Order matters and is not negotiable: the app's lifespan reads the
# active-instrument list and (when SHADOW_MODE_ENABLED) loads the ML artifact at
# startup, both of which assume the schema is already at head. Running
# `alembic upgrade head` here — in the same image, with the same DATABASE_URL the
# app will use — makes "deployed code newer than deployed schema" structurally
# impossible.
#
# `set -e` + no `|| true`: a failed migration MUST abort the boot. A container
# that serves traffic on a half-migrated schema is worse than a container that
# is visibly down.
#
# `exec` on the last line replaces this shell with uvicorn so uvicorn becomes
# PID 1 and receives SIGTERM directly — without it, `docker stop` would kill the
# shell and the FastAPI lifespan shutdown (price-stream close, scheduler
# shutdown) would never run.
#
# --reload is deliberately absent: the reloader forks a second process, which
# would run a SECOND APScheduler and a SECOND price stream against the same DB.
set -e

# `python -m alembic` rather than the bare `alembic` console script: the -m form
# puts the working directory on sys.path, which migrations/env.py needs for its
# `from app.database import Base`. PYTHONPATH=/app in the Dockerfile covers this
# too — belt and braces, because a broken migration step is a broken deploy.
echo "[entrypoint] alembic upgrade head"
python -m alembic upgrade head

echo "[entrypoint] starting uvicorn on 0.0.0.0:8000"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
