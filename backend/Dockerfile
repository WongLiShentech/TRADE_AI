# Production image — targets always-on Linux (built and verified on Oracle Cloud
# ARM / aarch64; the tag is multi-arch so the same Dockerfile builds on x86_64).
FROM python:3.12-slim

# PYTHONUNBUFFERED: logs reach `docker logs` immediately instead of sitting in a
#   4KB pipe buffer — the difference between "I can see the scheduler firing" and
#   "the container looks dead".
# PYTHONDONTWRITEBYTECODE: no .pyc churn in a read-only-ish app dir.
# TZ=UTC: every cron trigger, candle timestamp and feature timestamp in this
#   platform is UTC. A container inheriting a local timezone would silently shift
#   the H4 candle-close jobs.
# PYTHONPATH: alembic's `migrations/env.py` does `from app.database import Base`.
#   Console-script entry points do NOT put the working directory on sys.path (and
#   have not since 3.11), so a bare `alembic upgrade head` inside the container dies
#   with ModuleNotFoundError: No module named 'app' — while the identical command
#   works in dev, where `python -m alembic` implicitly adds CWD. Pinning it here
#   makes the container behave the same however alembic is invoked.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    TZ=UTC

WORKDIR /app

# libgomp1: XGBoost's OpenMP runtime. The manylinux wheel links against it but
# does not vendor it, so `import xgboost` fails on a bare slim image — which on
# this platform means the S1 artifact cannot load and shadow mode would refuse to
# boot (see the startup assertion in app/main.py).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Layer split: dependencies before source ──────────────────────────────────
# requirements.txt changes rarely; app/ changes every commit. Copying them in
# separate layers keeps the ~1GB scientific-stack install (xgboost, sklearn,
# shap, pandas, numpy) cached across ordinary code deploys.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Entrypoint copied before the source so a source edit does not invalidate it.
# `sed` strips CR: the repo is developed on Windows and a CRLF shebang line makes
# the kernel look for an interpreter literally named "/bin/sh\r".
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh

# .dockerignore governs what actually lands here. Note that backend/models/ IS
# copied on purpose (it is gitignored but ML_MODEL_PATH resolves into it at
# runtime) and backend/.env is NOT (secrets arrive via env_file at run time).
COPY . .

# ── Non-root ─────────────────────────────────────────────────────────────────
# Fixed uid/gid so a bind-mounted host path keeps stable ownership. Created after
# COPY so the chown covers everything in one layer.
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Readiness, not liveness-by-TCP: /health verifies the DB session, the price-cache
# freshness and (when SHADOW_MODE_ENABLED) that the ML artifact is loaded. A
# 503 here means the process is up but cannot do its job — exactly the state that
# previously looked healthy while recording zero rows.
# Uses python+urllib rather than curl so no extra package is installed.
# start-period covers migrations + artifact load + first stream connect.
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=8)"]

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
