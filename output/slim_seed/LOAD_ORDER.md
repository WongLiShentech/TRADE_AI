# Slim seed — load runbook

Generated 2026-09-01T17:53:26.673710+00:00 by `backend/scripts/export_slim_seed.py`.

Total artifact size: **12.4 MB** across 6 CSVs. `trades` IS included (`--no-trades` to omit).

## Contents

| table | rows (approx) | csv size | what it is |
|---|---:|---:|---|
| `instruments` | 127 | 5.7 KB | ALL rows — the universe, is_active flags and pip_size everything else FKs to. |
| `macro_data` | 8,847 | 450.3 KB | ALL rows — bitemporal FRED vintages read point-in-time by feature_builder. |
| `news_calendar_events` | 212 | 27.4 KB | ALL rows — the forward calendar spine, including future-dated events. |
| `candles` | 12,000 | 843.9 KB | price_type='M' only, newest 1000 H4, 200 D bars per instrument. NO M1 ROWS — the hourly trailing-window job rebuilds the live M1 series from empty (this is the ~99.8% size win). |
| `indicators` | 10,600 | 766.7 KB | newest 1000 H4, 60 D rows per instrument — recomputable, shipped so the first live candle-close job can fire immediately. |
| `trades` | 7,119 | 10.4 MB | ALL rows — M7 backtest corpus (S1 training set) + any recorded shadow rows. ~12 MB. Omit with --no-trades for an observation-only deploy. |

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

# 2. DATA — run from inside this directory (\copy resolves paths client-side).
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
docker compose -f docker-compose.prod.yml exec db \
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
