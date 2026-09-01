# DEPLOYMENT.md — one runbook, three targets

Deploying the **M8-Shadow** observation stack: FastAPI + APScheduler + PostgreSQL in
Docker Compose. It records what the ML model *would* have done on live signals and
**places no orders** (`ORDER_PLACEMENT_ENABLED=false`, enforced in code, not config).

The **same artifacts** deploy everywhere. Only `.env` differs.

| | Target A | Target B | Target C |
|---|---|---|---|
| Host | **Raspberry Pi 400** | **Oracle Cloud A1** | x86_64 VPS (Hetzner) |
| Arch | arm64 | arm64 | amd64 |
| RAM | 4 GB | 12 GB | any ≥ 2 GB |
| Network | home LAN, **behind NAT** | **PUBLIC IP** | **PUBLIC IP** |
| Storage | microSD / USB-SSD | block storage | SSD |
| Status | supported | supported | documented fallback |

Divergences are marked **[A]** / **[B]** / **[C]** throughout. Anything unmarked is
identical on all three.

> **Read this once before you start.** Two things silently produce a deployment that
> looks perfectly healthy and collects nothing:
> 1. **`backend/models/` is gitignored.** A `git clone` does **not** contain the ML
>    artifact. See [Step 5](#step-5--transfer-the-three-things-git-does-not-carry).
> 2. **The backend must be restarted *after* the seed loads.** The active-instrument
>    list is read once, at startup. See [Step 8](#step-8--restart-the-backend-not-optional).

---

## Table of contents

1. [Prerequisites](#step-1--prerequisites)
2. [Install Docker](#step-2--install-docker)
3. [Pi-only: enable cgroup memory](#step-3--a-only-enable-cgroup-memory)
4. [Get the code onto the host](#step-4--get-the-code-onto-the-host)
5. [Transfer the three things git does not carry](#step-5--transfer-the-three-things-git-does-not-carry)
6. [Configure `.env` — per-target matrix](#step-6--configure-env)
7. [Build and start](#step-7--build-and-start)
8. [Load the seed, then restart](#step-8--restart-the-backend-not-optional)
9. [The 11-step verification checklist](#the-11-step-verification-checklist)
10. [Checking results day to day](#checking-results-day-to-day)
11. [Updating later](#updating-later)
12. [Stopping and starting](#stopping-and-starting)
13. [Storage and card endurance](#storage-and-card-endurance-a-mainly)
14. [Troubleshooting](#troubleshooting)
15. [Per-target config matrix (reference)](#per-target-config-matrix-reference)

---

## Step 1 — Prerequisites

On the host:

- 64-bit OS. **[A]** Raspberry Pi OS Lite **64-bit** or Ubuntu arm64 — *not* the
  32-bit image (`uname -m` must say `aarch64`, not `armv7l`; there are no 32-bit ARM
  wheels for numpy/xgboost and a source build will not finish on a Pi).
- ≥ 8 GB free storage. **[A]** 32 GB card minimum; USB-SSD strongly preferred
  (see [endurance](#storage-and-card-endurance-a-mainly)).
- Outbound HTTPS. Needs to reach `api-fxpractice.oanda.com`,
  `stream-fxpractice.oanda.com`, `api.stlouisfed.org` (FRED), `pypi.org` and
  `docker.io`. No inbound access is required on any target.
- SSH access.

On your workstation: the repo, a populated `backend/.env`, `backend/models/`, and
`scp`.

```bash
uname -m          # expect: aarch64  [A][B]   or  x86_64  [C]
free -h           # expect: ~4Gi [A] / ~12Gi [B]
df -h /           # expect: ≥ 8G available
```

---

## Step 2 — Install Docker

The convenience script works on Raspberry Pi OS **and** Ubuntu, arm64 and amd64:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker            # or log out and back in
docker version           # expect Client + Server, no permission error
docker compose version   # expect v2.x  ← must be the PLUGIN, not docker-compose v1
```

**[B] Oracle Cloud only** — OCI images ship with a restrictive iptables ruleset that
blocks Docker's bridge networking:

```bash
sudo iptables -I INPUT 1 -i docker0 -j ACCEPT
sudo netfilter-persistent save     # apt install iptables-persistent if missing
```

Do **not** open port 8000 in the OCI security list or in `ufw`. The API stays on
loopback and you reach it by SSH tunnel (Step 6).

---

## Step 3 — [A] only: enable cgroup memory

**Skip on [B]/[C].**

Raspberry Pi OS ships with the kernel memory cgroup **disabled**. Docker then
*silently ignores* `mem_limit` — you get a warning at `up` time and no limit. That
matters here: without limits, a runaway process invites the host OOM killer, which
chooses its victim by heuristic and may well pick `sshd`.

```bash
# Pi OS Bookworm+ uses /boot/firmware/cmdline.txt; older uses /boot/cmdline.txt
sudo sed -i '1 s/$/ cgroup_enable=memory cgroup_memory=1/' /boot/firmware/cmdline.txt
sudo reboot
```

`cmdline.txt` is **one single line** — append, never add a newline.

Verify after reboot:

```bash
cat /sys/fs/cgroup/cgroup.controllers      # must contain: memory
docker info 2>&1 | grep -i "limit support" # must print NOTHING
```

If `docker info` prints `WARNING: No memory limit support`, the edit did not take.

---

## Step 4 — Get the code onto the host

```bash
git clone <your-repo-url> ~/Trade_AI
cd ~/Trade_AI
git log --oneline -1     # confirm you are on the commit you meant to deploy
```

No repo on the host? `rsync` works too — but then read Step 5 *very* carefully,
because rsync will happily copy `backend/.env` and `backend/models/` and you will not
learn the lesson until the day you deploy from a clean clone.

---

## Step 5 — Transfer the three things git does not carry

**This is the step that silently breaks deployments.** Three artifacts are excluded
from version control on purpose and must be copied by hand.

### 5a. The ML model artifact — `backend/models/`

`backend/models/` is in `.gitignore` (serialized binaries are not source). It is
**required at runtime**: `ML_MODEL_PATH` resolves into it and, with
`SHADOW_MODE_ENABLED=true`, `app/main.py` **refuses to start** without a loadable
artifact. That startup assertion exists precisely because the previous behaviour was
to boot, pass every health probe, and record **zero rows** indefinitely.

```bash
# from your workstation, repo root
scp -r backend/models <user>@<host>:~/Trade_AI/backend/
```

Verify on the host **before building** — `.dockerignore` deliberately does *not*
exclude `models/`, so whatever is there gets baked into the image:

```bash
ls -la ~/Trade_AI/backend/models/
# expect the .joblib named by ML_MODEL_PATH, plus its .metadata.json / .model_card.md
```

### 5b. Config — `backend/.env`

```bash
scp backend/.env <user>@<host>:~/Trade_AI/backend/.env
scp .env         <user>@<host>:~/Trade_AI/.env          # root: compose vars
```

Then apply the per-target edits in [Step 6](#step-6--configure-env). `.env` files are
never baked into the image (`.dockerignore` excludes them); they are injected at run
time via `env_file`.

### 5c. The database seed

The dev database is ~7.2 GB, ~99.8% of which is historical M1 Bid/Ask that the server
never re-reads. Export the slim seed on your workstation:

```bash
# workstation, from backend/
python scripts/export_slim_seed.py --archive
# -> output/slim_seed/  (~12 MB of CSV) and output/slim_seed.tar.gz (~2 MB)
```

```bash
scp output/slim_seed.tar.gz <user>@<host>:~/
ssh <user>@<host> 'tar xzf ~/slim_seed.tar.gz -C ~/'
```

Contents and rationale: `output/slim_seed/LOAD_ORDER.md` (generated with the seed).
Add `--no-trades` for the leanest possible observation-only deploy; keep trades if you
may ever retrain on the server, and to carry existing shadow rows across.

---

## Step 6 — Configure `.env`

Two files, two jobs:

| File | Configures | Reference |
|---|---|---|
| `~/Trade_AI/.env` | the **containers** (ports, memory, Postgres flags) | `.env.example` |
| `~/Trade_AI/backend/.env` | the **application** | `backend/.env.example`, `backend/.env.production.example` |

### 6a. Application config — `backend/.env`

Walk `backend/.env.production.example` top to bottom; it lists every key that must
differ from the Windows dev machine and why. The essentials:

```bash
# Windows paths do not exist on Linux. VAULT_PATH=C:/SecondBrain is parsed on Linux
# as a RELATIVE directory literally named "C:" and would be silently CREATED.
VAULT_PATH=/app/vault
WIKI_INGEST_ENABLED=false
ANTHROPIC_API_KEY=            # leave EMPTY: a second guarantee no credits are spent

# The database lives on the compose network, not localhost.
# (docker-compose.prod.yml already overrides this; set it anyway for non-compose runs.)
DATABASE_URL=postgresql://<user>:<pw>@db:5432/<db>

# Safety flags — RE-READ THESE ON EVERY DEPLOY.
ORDER_PLACEMENT_ENABLED=false
SHADOW_MODE_ENABLED=true
ML_MODEL_PATH=models/<your-artifact>.joblib
ML_ALLOW_UNPROMOTED_MODEL=true        # required while shadowing a .NOT_PROMOTED model

# Confirm these are the PRACTICE endpoints. Practice vs live is URL-only.
OANDA_BASE_URL=https://api-fxpractice.oanda.com
OANDA_STREAM_URL=https://stream-fxpractice.oanda.com
```

> `ORDER_PLACEMENT_ENABLED=true` while `ML_MODEL_PATH` points at a `.NOT_PROMOTED`
> artifact is refused at load time and the container will not start. That is correct
> behaviour, not a bug.

**CORS** — only matters for browser JavaScript, never for `curl` or for opening a URL
directly:

- **[B][C]** leave as-is, or set `http://localhost:8000` (the tunnelled origin).
- **[A]** if you later serve the React app from another machine, list that browser
  origin: `CORS_ORIGINS=http://tradeai.local:5173,http://192.168.1.50:5173`.
  Never `*` — `allow_credentials=True` is set, and browsers reject that combination.

### 6b. Container config — root `.env`

Every knob has a safe default; an empty value means "use the default". The **only**
line most deployments change:

```bash
# [B] Oracle Cloud / [C] VPS — PUBLIC IP: leave EMPTY (defaults to 127.0.0.1).
API_BIND_HOST=

# [A] Raspberry Pi on a home LAN behind NAT — to browse from another device:
API_BIND_HOST=0.0.0.0
```

**Why the default is loopback.** This API has **no authentication**. On a public-IP
host, publishing on `0.0.0.0` puts it on the internet — and Docker inserts its own
`DOCKER-USER` iptables rules *ahead* of `ufw`/`firewalld`, so "the firewall is on"
does **not** protect a published port. `0.0.0.0` is acceptable on **[A]** only because
there is no inbound path from the internet. If that Pi ever gets a port-forward,
change this back **first**.

**[B][C] access pattern** — SSH tunnel from your workstation:

```bash
ssh -L 8000:127.0.0.1:8000 <user>@<host>
# then browse http://localhost:8000/docs
```

**[A] access pattern** — directly from any LAN device:

```
http://<pi-hostname>.local:8000/api/v1/shadow/performance
```

Postgres publishes **no port on any target**. It is reachable only over the compose
network; use `docker compose exec db psql` to get a shell.

Optional per-target sizing (all default to the 4 GB Pi profile):

```bash
# [B] 12 GB Oracle A1 — more headroom if you intend to retrain on the server
BACKEND_MEM_LIMIT=3g
DB_MEM_LIMIT=2g
PG_SHARED_BUFFERS=1GB
PG_EFFECTIVE_CACHE_SIZE=4GB
```

---

## Step 7 — Build and start

The image is built **on the target**, so it is natively `aarch64` on [A]/[B] and
`amd64` on [C]. Do **not** add `--platform` — that is what turns a native build into
an emulated one.

```bash
cd ~/Trade_AI
docker compose -f docker-compose.prod.yml up -d --build
```

**[A]** First build takes **20–40 minutes** on a Pi (the scientific stack is ~1 GB of
wheels; none are compiled from source — every pin has an `aarch64` wheel). Run it
under `tmux`/`screen` so an SSH drop does not kill it. Later builds reuse the cached
dependency layer and take under a minute.

Watch it come up:

```bash
docker compose -f docker-compose.prod.yml logs -f backend
```

Expected, in order:

```
[entrypoint] alembic upgrade head
INFO  [alembic.runtime.migration] Running upgrade ... -> <head>
[entrypoint] starting uvicorn on 0.0.0.0:8000
INFO:     app.main — startup check OK — ML artifact loaded: model_id=... promoted=False
INFO:     app.services.scheduler — scheduler started; jobs: ...
INFO:     app.main — no active instruments — price stream skipped (run POST /instruments/sync then restart)
```

That last line is **expected and correct** on a first boot — the database is empty
until you load the seed. Fix it in Step 8.

---

## Step 8 — Restart the backend (not optional)

Order is **schema → data → restart**. The entrypoint already did the schema.

```bash
# 1. DATA — copy the seed into the db container and load it
docker compose -f docker-compose.prod.yml cp ~/slim_seed db:/tmp/seed
docker compose -f docker-compose.prod.yml exec db \
    sh -c "cd /tmp/seed && psql -v ON_ERROR_STOP=1 -U <user> -d <db> -f load.sql"

# 2. RESTART — do not skip
docker compose -f docker-compose.prod.yml restart backend
```

`ON_ERROR_STOP=1` matters: without it `psql` reports failures and keeps going, leaving
a half-loaded database that looks like it worked.

### Why the restart is mandatory

`app.main.lifespan` reads the active-instrument list **exactly once, at startup**, to
decide which symbols to subscribe the price stream to. A backend that was already
running when the seed landed holds an **empty symbol list**: no stream, no ticks, and
`/health` cheerfully reporting `price_stream: ok — intentionally not started`. It will
sit there looking fine and doing nothing until it is restarted.

After the restart you should see:

```
INFO:     app.main — price stream started for 10 instruments
```

> Deploying with **no** seed? Then run `POST /api/v1/instruments/sync` to discover the
> universe from the broker, mark the ones you want active, and restart. Same rule.

---

## The 11-step verification checklist

Run all eleven. Steps 1–8 are immediate; 9–11 need market time.

```bash
cd ~/Trade_AI
C="docker compose -f docker-compose.prod.yml"
```

**1. Both containers up, backend healthy**

```bash
$C ps
```
Expect `db` and `backend` both `Up`, backend `(healthy)` within ~2 minutes. `(starting)`
is normal during the 120 s `start_period`. `(unhealthy)` → go to [Troubleshooting](#troubleshooting).

**2. Schema is at head**

```bash
$C exec backend python -m alembic current
```
Expect one revision id followed by `(head)`.

**3. Health endpoint is green**

```bash
$C exec backend python -c "import urllib.request,json; print(json.dumps(json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')),indent=2))"
```
Expect `database: ok`, `ml_model: ok`, and `price_stream: ok` during market hours.
`price_stream: degraded` at a weekend is expected and returns HTTP 200 by design.

**4. The ML artifact actually loaded** (not merely "the file exists")

```bash
$C logs backend | grep "startup check OK"
```
Expect `model_id=... promoted=False`. **No output = shadow mode is not observing
anything.** With `SHADOW_MODE_ENABLED=true` the container should not even have started;
if it did, check `SHADOW_MODE_ENABLED`.

**5. Order placement is OFF**

```bash
$C exec backend python -c "from app.config import get_settings as g; s=g(); print('ORDER_PLACEMENT_ENABLED =', s.ORDER_PLACEMENT_ENABLED); print('SHADOW_MODE_ENABLED    =', s.SHADOW_MODE_ENABLED)"
```
Expect `False` then `True`. Anything else — stop and fix before market open.

**6. Practice endpoints, not live**

```bash
$C exec backend python -c "from app.config import get_settings as g; print(g().OANDA_BASE_URL, g().OANDA_STREAM_URL)"
```
Both must contain `fxpractice`.

**7. Seed loaded and instruments active**

```bash
$C exec db psql -U <user> -d <db> -c "SELECT count(*) FILTER (WHERE is_active) AS active, count(*) AS total FROM instruments;"
$C exec db psql -U <user> -d <db> -c "SELECT granularity, price_type, count(*) FROM candles GROUP BY 1,2 ORDER BY 1,2;"
```
Expect your active universe, and **no M1 rows yet** (the hourly job creates them).

**8. Price stream is subscribed**

```bash
$C logs backend | grep "price stream started"
```
Expect `price stream started for N instruments`, N = your active count. If instead you
see `no active instruments — price stream skipped`, you skipped the restart in Step 8.

**9. Scheduler jobs are registered** *(immediate)* **and firing** *(within 4 h)*

```bash
$C logs backend | grep "scheduler started"
# after the next :20 past the hour:
$C logs backend | grep "trailing window refresh complete"
```
The M1 job runs hourly and is the fastest proof the scheduler works.

**10. Macro data is fresh** *(within 4 h — the intraday fundamentals cron)*

```bash
$C logs backend | grep -i "macro"
```
Expect `macro staleness: all N registered series fresh`. A `macro series STALE` warning
means the named series' features will be **NaN on every live signal** — see
[Troubleshooting](#macro-series-stale).

**11. Shadow rows are being recorded** *(within one H4 cycle that produces a signal —
can legitimately take days)*

```bash
$C exec db psql -U <user> -d <db> -c "SELECT count(*), min(opened_at), max(opened_at) FROM trades WHERE stage='shadow';"
```
Signals are rare by design (2–4/week target, `SIGNAL_MIN_CONFLUENCE_SCORE`). Zero rows
after a day is not yet alarming; zero after a week with a healthy stream is — see
[No shadow rows appearing](#no-shadow-rows-appearing).

---

## Checking results day to day

### API — the two endpoints that matter

**[A]** from any LAN device · **[B][C]** through the SSH tunnel:

```
GET http://<host>:8000/api/v1/shadow/performance   # take-vs-skip cohorts, filter effect
GET http://<host>:8000/api/v1/shadow/decisions     # paged per-signal decision log
GET http://<host>:8000/docs                        # interactive OpenAPI
```

```bash
curl -s http://127.0.0.1:8000/api/v1/shadow/performance | python -m json.tool
```

### SQL — the ground truth

```bash
C="docker compose -f docker-compose.prod.yml"

# decision mix + how many are resolved
$C exec db psql -U <user> -d <db> -c "
SELECT ml_decision,
       count(*) AS n,
       count(*) FILTER (WHERE closed_at IS NOT NULL) AS resolved,
       round(avg(rr_actual)::numeric, 3) AS avg_r
FROM trades WHERE stage='shadow' GROUP BY 1 ORDER BY 1;"

# resolution quality — ambiguous rows are NOT comparable to the M7 corpus
$C exec db psql -U <user> -d <db> -c "
SELECT ambiguous_resolution, count(*) FROM trades
WHERE stage='shadow' AND closed_at IS NOT NULL GROUP BY 1;"

# the pending queue (waiting on hold horizon + M1 coverage)
$C exec db psql -U <user> -d <db> -c "
SELECT count(*) AS pending, min(opened_at) AS oldest FROM trades
WHERE stage='shadow' AND closed_at IS NULL;"

# database size — watch this on [A]
$C exec db psql -U <user> -d <db> -c "
SELECT pg_size_pretty(pg_database_size('<db>')) AS total,
       pg_size_pretty(pg_total_relation_size('candles')) AS candles;"
```

---

## Updating later

### New application code

```bash
cd ~/Trade_AI
git pull
docker compose -f docker-compose.prod.yml up -d --build
```

The entrypoint runs `alembic upgrade head` before uvicorn, so a schema change ships
with its code automatically. There is no source bind-mount: **code changes only take
effect on a rebuild**. `restart` alone re-runs the old image.

**[A]** A dependency change re-installs the whole ~1 GB scientific stack (20–40 min).
A code-only change reuses the cached layer.

### New ML model

```bash
scp backend/models/<new-artifact>.* <user>@<host>:~/Trade_AI/backend/models/
ssh <user>@<host>
cd ~/Trade_AI
nano backend/.env      # ML_MODEL_PATH= and ML_DECISION_THRESHOLD= (from the metadata.json)
docker compose -f docker-compose.prod.yml up -d --build      # REBUILD: models/ is baked in
docker compose -f docker-compose.prod.yml logs backend | grep "startup check OK"
```

`restart` is **not** enough — the artifact lives inside the image. Confirm the new
`model_id` in that log line. Changing the model mid-experiment makes rows before and
after non-comparable: record the switch instant.

### Config-only change

```bash
nano backend/.env
docker compose -f docker-compose.prod.yml up -d       # recreates with new env; no rebuild
```

---

## Stopping and starting

```bash
C="docker compose -f docker-compose.prod.yml"

$C stop            # stop, keep containers and data
$C start           # start again
$C restart backend # backend only (use after a seed load or a manual DB change)
$C down            # remove containers; NAMED VOLUMES (your data) SURVIVE
$C down -v         # ← DESTROYS pgdata. Every shadow observation is gone. Never routine.
```

Both services are `restart: unless-stopped`, so they come back automatically after a
reboot or a power cut. Postgres gets `SIGINT` (fast shutdown) with a 60 s grace period
so a `stop` is a clean checkpoint rather than a crash-recovery on next boot.

**Backup before anything risky:**

```bash
$C exec db pg_dump -U <user> -d <db> --no-owner | gzip > ~/tradeai-$(date +%F).sql.gz
```

---

## Storage and card endurance ([A] mainly)

Measured growth is **~122 MB/month**, almost entirely the hourly M1 Bid/Ask job.

**Use a USB SSD, not a microSD card.** Postgres write patterns (WAL, full-page images,
autovacuum) are close to worst-case for SD flash, and a card that dies takes the whole
corpus with it. A cheap USB-SSD is the single highest-value hardware decision here.
Booting the Pi from USB, or at minimum putting the `pgdata` volume on one, both work.

Already configured for you in `docker-compose.prod.yml`:

- `wal_compression=on` + `checkpoint_timeout=15min` — cuts WAL volume (full-page
  images after each checkpoint are the main write amplifier). **Crash-safe**; only
  recovery time grows.
- `random_page_cost=1.1` — flash, not spinning rust; keeps the planner on index scans.
- Docker `json-file` log rotation, `max-size=10m max-file=3` per service — an
  unbounded log is the classic way an always-on host fills its storage and takes the
  database down with it (and on a card it is pure avoidable wear).
- `stop_signal: SIGINT` — clean Postgres shutdown, no crash recovery per restart.

**Deliberately not enabled** (they trade durability for writes):

- `PG_SYNCHRONOUS_COMMIT=off` in the root `.env` cuts fsync frequency further. Risks
  losing the last **~200 ms of committed transactions** on an unclean power loss —
  never corruption. Reasonable for a Pi collecting observations; your call.
- `fsync=off` / `full_page_writes=off` — **never**. These corrupt the database on
  power loss, the single most likely failure for a Pi on a desk.

### Optional M1 retention prune

`backend/scripts/prune_m1_candles.py` deletes intrabar candles older than the outcome
resolver could ever need. It is **not scheduled** and defaults to a **dry run** —
pruning too aggressively makes shadow rows resolve `forced` +
`ambiguous_resolution=true`, which is silent and permanently contaminates the corpus.

> **Server only.** Never `--apply` this on the development machine. That database
> holds the ~36.7M-row historical M1 corpus the M7 backtest was resolved against —
> almost all of it older than any computed retention horizon, and none of it
> refetchable by the hourly job. On a server the slim seed ships **zero** M1 rows, so
> everything the script can see was accumulated live and is genuinely disposable.

```bash
C="docker compose -f docker-compose.prod.yml"
$C exec backend python scripts/prune_m1_candles.py                       # dry run
$C exec backend python scripts/prune_m1_candles.py --margin-days 14 --apply
$C exec db psql -U <user> -d <db> -c "VACUUM (ANALYZE) candles;"          # reclaim space
```

The retention floor is derived from `SIGNAL_MAX_HOLD_BARS` and `M1_LIVE_LOOKBACK_HOURS`
(never a written-down number) and is additionally clamped so it can never delete inside
the window an actually-pending shadow row will be resolved against.

Use plain `VACUUM`, not `VACUUM FULL`: the latter takes an ACCESS EXCLUSIVE lock and
rewrites the whole table — a long outage and a large burst of card writes.

---

## Troubleshooting

### Backend restart-loops / `unhealthy`

```bash
docker compose -f docker-compose.prod.yml logs --tail 100 backend
```

| Log line | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'app'` | `PYTHONPATH` lost | rebuild; the Dockerfile sets `PYTHONPATH=/app` |
| `could not translate host name "db"` | not run through compose | use `docker compose -f docker-compose.prod.yml`, not `docker run` |
| `ValidationError ... field required` | a key is missing from `backend/.env` | config has **zero defaults**; compare against `backend/.env.example` |
| `libgomp.so.1: cannot open shared object file` | image built without `libgomp1` | rebuild with `--no-cache`; the Dockerfile installs it |
| `alembic ... Can't locate revision` | host DB is ahead of the code | `git pull` to the matching commit |

### Model not found / not loaded

```
RuntimeError: SHADOW_MODE_ENABLED=true but the ML artifact at ML_MODEL_PATH=... could
not be loaded ... Resolved path: /app/models/...
```

The container is **correctly refusing to start**. In order of likelihood:

1. `backend/models/` was never copied to the host → [Step 5a](#5a-the-ml-model-artifact--backendmodels).
2. It was copied *after* the build. `models/` is baked into the image — you must
   `up -d --build` again, not `restart`.
3. `ML_MODEL_PATH` has a typo. It resolves **relative to `backend/`** (i.e.
   `/app` inside the container), so `models/x.joblib`, not `backend/models/x.joblib`.
4. The artifact is `.NOT_PROMOTED` and `ML_ALLOW_UNPROMOTED_MODEL=false`. Set it true
   — permitted only while `ORDER_PLACEMENT_ENABLED=false`.

```bash
docker compose -f docker-compose.prod.yml exec backend ls -la /app/models/
docker compose -f docker-compose.prod.yml exec backend python -c "from app.config import get_settings; from app.services.ml import inference as i; print(i.resolve_model_path(get_settings()))"
```

### Price stream dead

`/health` shows `price_stream: unhealthy — price stream task is not running`.

The reconnect loop retries indefinitely, so a *finished* task means the coroutine
itself died and nothing will revive it without a restart.

```bash
docker compose -f docker-compose.prod.yml logs backend | grep -i "stream"
docker compose -f docker-compose.prod.yml restart backend
```

Checklist: is `OANDA_STREAM_URL` reachable from the host
(`curl -sI https://stream-fxpractice.oanda.com`)? Is the API key still valid (OANDA
practice tokens can be revoked)? Are there active instruments at all
(`price stream skipped` means none)?

`price_stream: degraded — newest tick is Ns old` **outside market hours is expected**
and returns HTTP 200 on purpose; a container must not be restarted for a weekend.

### Macro series stale

```
WARNING: macro series STALE (6/N) — every feature derived from these is NaN on live
signals: DGS2(daily, 22d > 10d), VIXCLS(daily, 22d > 10d), ...
```

The feature builder refuses to carry a stale series forward (that would let the model
memorise a frozen-calendar regime), so those features go NaN. XGBoost tolerates NaN,
but rows with holes are **not comparable** to the M7 training corpus.

1. Is `FUNDAMENTAL_DATA_API_KEY` (FRED) present in `backend/.env`? Missing key = every
   FRED series stalls.
2. Can the host reach FRED?
   `docker compose -f docker-compose.prod.yml exec backend python -c "import httpx; print(httpx.get('https://api.stlouisfed.org/fred/', timeout=10).status_code)"`
3. Force a refresh by restarting the backend, then wait for the next
   `intraday_fundamentals` run (`FUNDAMENTAL_INTRADAY_REFRESH_HOURS`, at `:15`).
4. A genuinely long gap needs `python scripts/ingest_macro.py`.

Affected rows are still recorded, with the NaN count in
`signal_reasoning['shadow']['nan_model_features']` so they can be excluded later.

### No shadow rows appearing

Work down this list — it is ordered by how often each one is the real cause.

```bash
C="docker compose -f docker-compose.prod.yml"

# 1. Is the pipeline even running? (H4 jobs at :01 on 1,5,9,13,17,21 UTC)
$C logs backend | grep -i "candle_close\|pipeline"

# 2. Are candles arriving?
$C exec db psql -U <user> -d <db> -c "SELECT granularity, max(timestamp) FROM candles GROUP BY 1;"

# 3. Are signals being generated at all?
$C exec db psql -U <user> -d <db> -c "SELECT count(*), max(created_at) FROM signals;"

# 4. Is the bot paused by the circuit breaker?
$C exec db psql -U <user> -d <db> -c "SELECT * FROM bot_state;"
```

Most common causes, in order:

1. **The restart after seeding was skipped** — no stream, no signals. Grep for
   `price stream started`.
2. **Nothing qualified yet.** This is *usually* the answer. The strategy targets 2–4
   trades/week across the whole universe and `SIGNAL_MIN_CONFLUENCE_SCORE` gates hard.
   A quiet week is a real outcome, not a fault.
3. **`SHADOW_MODE_ENABLED=false`** — signals fire, nothing is recorded. Check step 5 of
   the verification checklist.
4. **Circuit breaker paused the bot** — `POST /api/v1/bot/resume` (manual, by design).
5. **Weekend / market closed** — no H4 closes, no signals.

Rows appear with `closed_at IS NULL` and are filled in later by the resolver (hourly at
`:35`), once the hold horizon has elapsed **and** the M1 coverage to label them
honestly exists. A pending row is normal for ~2 days on H4.

### `docker compose` says `mem_limit` is not supported / limits ignored

**[A]** cgroup memory is not enabled — [Step 3](#step-3--a-only-enable-cgroup-memory).
Everywhere else: you are on `docker-compose` v1. Install the v2 plugin
(`docker compose version` must print `v2.x`).

### Out of disk

```bash
df -h /
docker system df
docker image prune -f        # old build layers are usually the bulk
```

Then see [the retention prune](#optional-m1-retention-prune).

---

## Per-target config matrix (reference)

| Setting | Where | **[A] Pi 400** | **[B] Oracle A1** | **[C] x86 VPS** |
|---|---|---|---|---|
| `API_BIND_HOST` | root `.env` | `0.0.0.0` *(LAN, NAT only)* | *(empty →* `127.0.0.1`*)* | *(empty →* `127.0.0.1`*)* |
| `API_PORT` | root `.env` | `8000` | `8000` | `8000` |
| Access method | — | `http://<host>.local:8000` | SSH tunnel | SSH tunnel |
| `CORS_ORIGINS` | `backend/.env` | LAN browser origins | `http://localhost:8000` | `http://localhost:8000` |
| Postgres port | — | **never published** | **never published** | **never published** |
| `BACKEND_MEM_LIMIT` | root `.env` | *(empty → 1200m)* | `3g` optional | *(empty → 1200m)* |
| `DB_MEM_LIMIT` | root `.env` | *(empty → 768m)* | `2g` optional | *(empty → 768m)* |
| `PG_SHARED_BUFFERS` | root `.env` | *(empty → 256MB)* | `1GB` optional | *(empty → 256MB)* |
| `PG_SYNCHRONOUS_COMMIT` | root `.env` | `off` *(optional, SD wear)* | *(empty → on)* | *(empty → on)* |
| cgroup memory kernel flag | host | **required edit** | already on | already on |
| Docker bridge iptables fix | host | not needed | **required** | usually not needed |
| M1 retention prune | ops | **recommended** | optional | optional |
| USB SSD | hardware | **strongly recommended** | n/a | n/a |
| First build time | — | 20–40 min | 5–10 min | 3–5 min |
| `VAULT_PATH` | `backend/.env` | `/app/vault` | `/app/vault` | `/app/vault` |
| `WIKI_INGEST_ENABLED` | `backend/.env` | `false` | `false` | `false` |
| `ORDER_PLACEMENT_ENABLED` | `backend/.env` | `false` | `false` | `false` |
| `SHADOW_MODE_ENABLED` | `backend/.env` | `true` | `true` | `true` |
| `DATABASE_URL` | compose | `...@db:5432/...` | `...@db:5432/...` | `...@db:5432/...` |

**Identical on every target:** all `RISK_*`, `ATR_*`, `MIN_RR_RATIO`, `SIGNAL_*`,
`ML_*` hyperparameters, `EXECUTION_WINDOW_*`, `MID_WARMUP_START`, `BACKTEST_*`. The
strategy is the strategy — identical values are what make dev results and server
results comparable. Do not "tune" them per host.

---

## Architecture portability notes

- **Nothing pins a CPU architecture.** `python:3.12-slim` is a multi-arch tag; a plain
  `docker build` on the target produces a native image. Never add `platform:`.
- **Every pinned dependency ships both `x86_64` and `aarch64` wheels** — verified for
  all 18 direct pins and for the compiled transitive deps (`uvloop`, `httptools`,
  `websockets`, `watchfiles`, `greenlet`, `scipy`, `numba`, `llvmlite`). Nothing
  compiles from source on the Pi.
- **`libgomp1`** is installed explicitly: XGBoost's manylinux wheel links against
  OpenMP but does not vendor it, on either architecture.
- **32-bit ARM is not supported.** There are no `armv7l` wheels for numpy/xgboost. Use
  a 64-bit OS image.
