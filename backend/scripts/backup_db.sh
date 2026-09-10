#!/usr/bin/env bash
#
# Nightly database backup for the production deployment.
#
# Why this exists
# ---------------
# The server holds forward evidence — live model decisions and their resolved
# outcomes — that took weeks of real market time to accumulate and CANNOT be
# regenerated. The backtest corpus can be rebuilt from candles; a shadow row
# recorded on 5 August cannot be rebuilt by anything.
#
# What it does, and what "backup" means here
# -------------------------------------------
# A dump nobody has ever restored is not a backup, it is a file. Every run
# therefore does two things:
#
#   1. pg_dump in custom format (-Fc): compressed, and restorable table-by-table
#      rather than all-or-nothing.
#   2. `pg_restore --list` on the result. This parses the archive's table of
#      contents and fails on a truncated or corrupt file — which a plain
#      "did pg_dump exit 0?" check does NOT catch, because a dump interrupted by
#      a full disk can still exit cleanly with a partial file.
#
# `--verify-restore` goes further and restores into a throwaway database, then
# counts rows in the tables that matter. Slower (~30s at this size), and the
# only check that actually proves the file is usable. Worth running weekly.
#
# Usage (from the deployment directory, e.g. ~/Trade_AI):
#     scripts/backup_db.sh                    # dump + integrity check
#     scripts/backup_db.sh --verify-restore   # ...plus a real restore test
#
# Install as a cron entry — see DEPLOYMENT.md.
#
set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-$HOME/Trade_AI}"
BACKUP_DIR="${BACKUP_DIR:-$HOME/backups}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
# Keep 14 days of daily dumps. At ~15 MB compressed against 19 GB free this is
# nothing; the limit exists so an unattended box cannot fill its own disk.
RETAIN="${RETAIN:-14}"
VERIFY_RESTORE=0
[[ "${1:-}" == "--verify-restore" ]] && VERIFY_RESTORE=1

cd "$DEPLOY_DIR"
mkdir -p "$BACKUP_DIR"

# Credentials come from the deployment's own .env — never duplicated here, so
# rotating them does not silently break backups.
DB_USER="$(grep -E '^POSTGRES_DB_USER=' .env | cut -d= -f2-)"
DB_NAME="$(grep -E '^POSTGRES_DB_NAME=' .env | cut -d= -f2-)"
: "${DB_USER:?POSTGRES_DB_USER missing from .env}"
: "${DB_NAME:?POSTGRES_DB_NAME missing from .env}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$BACKUP_DIR/${DB_NAME}_${STAMP}.dump"

log() { printf '%s  %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

log "backup starting -> $OUT"

CID="$(docker compose -f "$COMPOSE_FILE" ps -q db)"
: "${CID:?db container is not running}"
IN_CONTAINER="/tmp/backup_$$.dump"
cleanup() { docker exec "$CID" rm -f "$IN_CONTAINER" 2>/dev/null || true; }
trap cleanup EXIT

# The dump is written INSIDE the container first, not streamed to stdout.
#
# A custom-format (-Fc) archive is a SEEKABLE format: pg_restore jumps around its
# table of contents rather than reading start to finish. Streaming it through a
# pipe therefore produces a file that pg_dump writes happily and pg_restore cannot
# inspect — `pg_restore --list /dev/stdin` fails on "not a valid archive" even
# though the bytes are fine. Writing to a real file keeps it seekable, so the
# integrity check below can actually run.
docker exec "$CID" pg_dump -U "$DB_USER" -d "$DB_NAME" -Fc -f "$IN_CONTAINER"

# ── integrity: can the archive actually be read? ─────────────────────────────
# This runs BEFORE the file reaches the host, so a corrupt archive never lands in
# the backup directory looking legitimate. pg_dump exiting 0 is not sufficient
# evidence — a dump interrupted by a full disk can still exit cleanly.
if ! docker exec "$CID" pg_restore --list "$IN_CONTAINER" > /dev/null 2>&1; then
    log "FATAL: pg_restore --list failed — the archive is unreadable. Nothing retained."
    exit 1
fi

TABLES=$(docker exec "$CID" pg_restore --list "$IN_CONTAINER" 2>/dev/null | grep -c 'TABLE DATA' || true)
log "integrity OK — $TABLES tables present in the archive"

# A dump that is readable but holds no trades is a successful backup of nothing.
if [[ "$TABLES" -lt 10 ]]; then
    log "FATAL: only $TABLES tables in the archive; expected 15+. Nothing retained."
    exit 1
fi

# Only now does it become a backup on the host.
docker cp "$CID:$IN_CONTAINER" "$OUT.partial"
mv "$OUT.partial" "$OUT"
SIZE=$(stat -c %s "$OUT")
log "dump written: $(numfmt --to=iec "$SIZE")"

# ── optional: prove it restores ──────────────────────────────────────────────
# The only check that proves the file is USABLE rather than merely well-formed.
# Restores the in-container copy (still seekable) into a throwaway database.
if [[ "$VERIFY_RESTORE" -eq 1 ]]; then
    SCRATCH="restore_verify_$$"
    log "verify: restoring into $SCRATCH"
    docker exec "$CID" psql -U "$DB_USER" -d postgres \
        -c "DROP DATABASE IF EXISTS $SCRATCH;" -c "CREATE DATABASE $SCRATCH;" > /dev/null
    docker exec "$CID" pg_restore -U "$DB_USER" -d "$SCRATCH" --no-owner "$IN_CONTAINER" > /dev/null 2>&1 || true
    COUNTS=$(docker exec "$CID" psql -U "$DB_USER" -d "$SCRATCH" -t -A -F' ' \
        -c "SELECT (SELECT count(*) FROM trades), (SELECT count(*) FROM trades WHERE stage='shadow');")
    docker exec "$CID" psql -U "$DB_USER" -d postgres -c "DROP DATABASE $SCRATCH;" > /dev/null
    log "verify: restored trades/shadow = $COUNTS"
    # The whole point of the exercise: shadow rows are the irreplaceable ones.
    # A restore that yields zero of them has proved the backup is worthless.
    if [[ "$(echo "$COUNTS" | awk '{print $2}')" -lt 1 ]]; then
        log "FATAL: restored database has no shadow rows — backup is not usable."
        exit 1
    fi
fi

# ── retention ────────────────────────────────────────────────────────────────
# Prune only well-formed dumps; .CORRUPT/.SUSPECT files are left behind on
# purpose so a failure is still visible tomorrow.
mapfile -t OLD < <(ls -1t "$BACKUP_DIR"/${DB_NAME}_*.dump 2>/dev/null | tail -n +$((RETAIN + 1)))
for f in "${OLD[@]:-}"; do
    [[ -n "$f" ]] && { log "pruning $(basename "$f")"; rm -f "$f"; }
done

log "backup complete — $(ls -1 "$BACKUP_DIR"/${DB_NAME}_*.dump 2>/dev/null | wc -l) retained, $(du -sh "$BACKUP_DIR" | cut -f1) total"
