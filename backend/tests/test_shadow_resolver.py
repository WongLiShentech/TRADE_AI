"""Integration tests for app.services.shadow.resolver — M8-Shadow Phase 3.

These run against the live dev Postgres (same convention as the Phase 2 recorder
suite) because every property under test is a persistence or a data-sufficiency
property: the outcome columns must actually land on the row, the resolvability rule
must actually consult stored M1, and a re-run must actually be a no-op at the DB
level. A mocked session would prove none of that.

Trades are simulated against REAL stored M1 Bid/Ask at a historical signal time, so
the resolver is exercised through the same triple-barrier path the M7 corpus used
rather than through a stub.

Every test writes only ``stage='shadow'`` rows for the test symbol and an autouse
fixture deletes them before and after each test, so the real corpus
(``stage='backtest'``) is never touched.

Run from backend/:  python -m pytest tests/test_shadow_resolver.py -v
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import and_, or_

from app.domain.timeframes import get_timeframe
from app.models.candle import Candle
from app.models.trade import Trade
from app.services.backtester.runner import label_outcome
from app.services.backtester.simulator import horizon_bounds, simulate
from app.services.ml.inference import DECISION_SKIP, DECISION_TAKE
from app.services.shadow import recorder as rec
from app.services.shadow import resolver as res

_SYMBOL = "EUR_USD"
_GRAN = "H4"
# A historical signal time with real stored M1 Bid/Ask on both sides of it, far from
# the live right edge so the hold horizon is unambiguously in the past.
_T_BASE = datetime(2025, 8, 1, 13, 0, 1)
# SENTINEL TIME BANDS. This suite writes rows in exactly two disjoint windows, and
# the purge fixture / queue scope are bounded to their union.
#
# The bound is a safety requirement. The purge runs against the real dev database
# via the app's own SessionLocal on a live-active symbol, so an unbounded
# ``instrument_id = X AND stage = 'shadow'`` DELETE would irreversibly destroy real
# live shadow observations on every pytest run.
#
# Band 1 — COVERED: historical, real stored M1 Bid/Ask on both sides of it.
_COVERED_WINDOW = timedelta(days=30)
_COVERED_END = _T_BASE + _COVERED_WINDOW
# Band 2 — UNCOVERED: far future, so there is provably no M1 after it. The
# data-sufficiency tests used to derive these timestamps from ``utcnow()`` / the
# newest stored M1 bar, which planted rows in the LIVE present — exactly where real
# shadow rows land, and exactly what a time-bounded purge cannot safely clean up.
# ``resolve_pending`` takes an injectable ``now``, so a deterministic future sentinel
# tests the same property (zero observable bars) without ever touching live time.
_T_UNCOVERED = datetime(2099, 1, 1, 0, 0, 1)
_UNCOVERED_END = _T_UNCOVERED + timedelta(days=30)
_COVERED_BAND = (_T_BASE, _COVERED_END)
_UNCOVERED_BAND = (_T_UNCOVERED, _UNCOVERED_END)
# Stop distance as a multiple of ATR — mirrors SIGNAL_STOP_ATR_MULTIPLIER's role
# without importing a live signal; the exact value is irrelevant to the resolver.
_STOP_ATR_MULT = 1.5
_RR_TARGET = 2.0
# Sentinel meaning "use the real ATR fixture" — distinct from None, which is the
# JSON null a NaN feature becomes once feature_builder.json_safe has run.
_REAL_ATR = object()


# ── fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _clean_shadow_rows(db, instrument):
    """Delete every shadow row for the test symbol before and after each test."""
    inst = instrument(_SYMBOL)

    def _purge():
        db.query(Trade).filter(
            Trade.instrument_id == inst.id,
            Trade.stage == rec.STAGE_SHADOW,
            # TIME-BOUNDED to the union of the two sentinel bands — see the module
            # constants. Never widen this to an unbounded delete: it would wipe real
            # live shadow rows on every pytest run.
            or_(
                and_(Trade.opened_at >= _T_BASE, Trade.opened_at <= _COVERED_END),
                and_(
                    Trade.opened_at >= _T_UNCOVERED,
                    Trade.opened_at <= _UNCOVERED_END,
                ),
            ),
        ).delete(synchronize_session=False)
        db.commit()

    _purge()
    yield
    _purge()


@pytest.fixture()
def inst(instrument):
    return instrument(_SYMBOL)


@pytest.fixture()
def atr(db, settings, inst):
    """A realistic ATR(14) for the test window, read off the stored indicator row."""
    from app.models.indicator import Indicator

    row = (
        db.query(Indicator)
        .filter(
            Indicator.instrument_id == inst.id,
            Indicator.granularity == _GRAN,
            Indicator.timestamp <= _T_BASE,
            Indicator.atr14.isnot(None),
        )
        .order_by(Indicator.timestamp.desc())
        .first()
    )
    assert row is not None, "test needs a stored ATR(14) at the sentinel signal time"
    return float(row.atr14)


@pytest.fixture()
def price(db, inst):
    """The real Mid close at the sentinel signal time — keeps synthetic barriers
    realistic so the simulator produces a sane R rather than an instant gap-through."""
    row = (
        db.query(Candle.close)
        .filter(
            Candle.instrument_id == inst.id,
            Candle.granularity == _GRAN,
            Candle.price_type == "M",
            Candle.timestamp <= _T_BASE,
        )
        .order_by(Candle.timestamp.desc())
        .first()
    )
    assert row is not None, "test needs a stored Mid candle at the sentinel signal time"
    return float(row[0])


# ── helpers ──────────────────────────────────────────────────────────────────
def _barriers(price: float, atr: float, direction: str) -> tuple[float, float, float]:
    side = 1 if direction == "BUY" else -1
    entry = price
    stop = entry - side * _STOP_ATR_MULT * atr
    target = entry + side * _RR_TARGET * abs(entry - stop)
    return entry, stop, target


def _make_row(
    db,
    inst,
    atr: float,
    price: float,
    *,
    t: datetime = _T_BASE,
    direction: str = "BUY",
    ml_decision: str | None = DECISION_TAKE,
    atr_value=_REAL_ATR,
) -> Trade:
    """Insert one pending shadow row shaped exactly as the Phase 2 recorder writes it."""
    entry, stop, target = _barriers(price, atr, direction)
    reasoning = {
        "atr14": atr if atr_value is _REAL_ATR else atr_value,
        "confluence_score": 4,
        "session": "london",
        rec.REASONING_SHADOW_KEY: {
            "granularity": _GRAN,
            "signal_time": t.isoformat(),
            "risk_passed": True,
            "risk_rejection_reason": None,
        },
    }
    trade = Trade(
        instrument_id=inst.id,
        direction=direction,
        entry_price=entry,
        stop_price=stop,
        tp_price=target,
        units=1000,
        risk_amount=2.0,
        expected_pip_loss=abs(entry - stop) / inst.pip_size,
        rr_entry=_RR_TARGET,
        signal_source="rule_based",
        stage=rec.STAGE_SHADOW,
        opened_at=t,
        closed_at=None,
        outcome=None,
        exit_reason=None,
        ambiguous_resolution=False,
        signal_reasoning=reasoning,
        confluence_score=4,
        session="london",
        stop_method="atr",
        ml_probability=0.5 if ml_decision else None,
        ml_decision=ml_decision,
        ml_model_id="test_model",
    )
    db.add(trade)
    db.commit()
    db.refresh(trade)
    return trade


def _resolve(db, settings, inst, *, band=(_T_BASE, _COVERED_END), **kwargs):
    """Resolve, scoped to the test instrument AND to one sentinel band.

    The instrument scope alone is not enough: a REAL pending live shadow row for the
    same symbol would land in the queue and inflate every ``checked`` / ``resolved``
    count the assertions below depend on — and, worse, would be resolved for real by
    a test run. Bounding ``opened_at`` makes a pass provably incapable of touching
    anything but this suite's own rows.
    """
    return res.resolve_pending(
        db, settings, instrument_ids=[inst.id],
        opened_from=band[0], opened_to=band[1], **kwargs
    )


# ── 1. the happy path: every outcome column is filled, and correctly ─────────
def test_resolves_pending_row_and_fills_every_outcome_column(db, settings, inst, atr, price):
    trade = _make_row(db, inst, atr, price)
    assert trade.closed_at is None and trade.outcome is None

    summary = _resolve(db, settings, inst)

    assert summary["checked"] == 1 and summary["resolved"] == 1
    db.refresh(trade)
    assert trade.outcome in ("win", "loss", "breakeven")
    assert trade.rr_actual is not None
    assert trade.exit_price is not None
    assert trade.exit_reason in ("tp_hit", "sl_hit", "trailing_stop", "time_exit")
    assert trade.closed_at is not None and trade.closed_at > trade.opened_at
    assert isinstance(trade.ambiguous_resolution, bool)


def test_resolution_matches_a_direct_simulator_call(db, settings, inst, atr, price):
    """The resolver must be a thin wrapper over the M7 simulator — not a reimplementation."""
    trade = _make_row(db, inst, atr, price)
    expected = simulate(
        db, settings, inst.id, trade.direction,
        trade.entry_price, trade.stop_price, trade.tp_price,
        _T_BASE, atr, _GRAN,
    )

    _resolve(db, settings, inst)

    db.refresh(trade)
    assert trade.rr_actual == pytest.approx(expected.rr_actual)
    assert trade.exit_price == pytest.approx(expected.exit_price)
    assert trade.exit_reason == expected.exit_reason
    assert trade.closed_at == expected.exit_time
    assert trade.ambiguous_resolution == expected.ambiguous_resolution


def test_outcome_label_uses_the_runner_label_outcome_convention(db, settings, inst, atr, price):
    trade = _make_row(db, inst, atr, price)

    _resolve(db, settings, inst)

    db.refresh(trade)
    assert trade.outcome == label_outcome(trade.rr_actual)


def test_short_side_resolves_too(db, settings, inst, atr, price):
    """Direction is a parameter, never an assumption — SELL must resolve like BUY."""
    trade = _make_row(db, inst, atr, price, direction="SELL")

    summary = _resolve(db, settings, inst)

    assert summary["resolved"] == 1
    db.refresh(trade)
    assert trade.outcome == label_outcome(trade.rr_actual)
    assert trade.closed_at is not None


# ── 2. idempotency ───────────────────────────────────────────────────────────
def test_rerun_resolves_nothing_and_changes_nothing(db, settings, inst, atr, price):
    trade = _make_row(db, inst, atr, price)
    _resolve(db, settings, inst)
    db.refresh(trade)
    snapshot = (
        trade.outcome, trade.rr_actual, trade.exit_price,
        trade.exit_reason, trade.closed_at, trade.ambiguous_resolution,
    )

    second = _resolve(db, settings, inst)

    assert second["checked"] == 0, "a resolved row must not re-enter the pending queue"
    assert second["resolved"] == 0
    db.refresh(trade)
    assert (
        trade.outcome, trade.rr_actual, trade.exit_price,
        trade.exit_reason, trade.closed_at, trade.ambiguous_resolution,
    ) == snapshot


def test_pending_selector_is_closed_at_is_null(db, settings, inst, atr, price):
    """A row with closed_at set is invisible to the resolver even if outcome is NULL."""
    trade = _make_row(db, inst, atr, price)
    trade.closed_at = _T_BASE + timedelta(hours=8)
    db.commit()

    summary = _resolve(db, settings, inst)

    assert summary["checked"] == 0
    db.refresh(trade)
    assert trade.outcome is None, "the resolver must not have touched it"


# ── 3. a row that is not yet knowable is LEFT PENDING, never guessed ─────────
def test_row_whose_horizon_has_not_elapsed_stays_pending(db, settings, inst, atr, price):
    trade = _make_row(db, inst, atr, price)
    base_hours, _cap = horizon_bounds(settings, _GRAN)
    just_before = _T_BASE + timedelta(hours=base_hours - 1)

    summary = _resolve(db, settings, inst, now=just_before)

    assert summary["not_yet"] == 1 and summary["resolved"] == 0
    db.refresh(trade)
    assert trade.closed_at is None and trade.outcome is None and trade.rr_actual is None


def test_row_at_the_live_right_edge_stays_pending(db, settings, inst, atr, price):
    """A signal whose hold horizon has not run out yet is left alone — the realistic
    live case the scheduled resolver meets on every pass.

    Uses the UNCOVERED sentinel band with an injected ``now`` rather than real
    wall-clock time: a row planted at ``utcnow() - 1h`` would sit in the live present
    alongside genuine shadow rows, which is precisely what the purge fixture must not
    be allowed to reach.
    """
    now = _T_UNCOVERED + timedelta(hours=1)
    trade = _make_row(db, inst, atr, price, t=_T_UNCOVERED)

    summary = _resolve(db, settings, inst, band=_UNCOVERED_BAND, now=now)

    assert summary["resolved"] == 0 and summary["not_yet"] == 1
    db.refresh(trade)
    assert trade.closed_at is None


def test_horizon_elapsed_but_no_m1_coverage_stays_pending(db, settings, inst, atr, price):
    """Wall-clock time alone must never be enough. This row's hold window sits past
    the stored M1 right edge: the horizon HAS elapsed, yet resolving it would hit the
    simulator's truncated-right-edge branch and mislabel a barrier as a time exit."""
    base_hours, cap_hours = horizon_bounds(settings, _GRAN)
    # T far beyond the newest stored M1 bar → zero observable bars, deterministically.
    t = _T_UNCOVERED
    trade = _make_row(db, inst, atr, price, t=t)

    # Horizon elapsed, but still inside the closure cap → must stay pending.
    summary = _resolve(
        db, settings, inst, band=_UNCOVERED_BAND, now=t + timedelta(hours=base_hours + 1)
    )

    assert summary["not_yet"] == 1 and summary["resolved"] == 0
    db.refresh(trade)
    assert trade.closed_at is None
    assert res.observable_bars(
        db, inst.id, t, _GRAN, t + timedelta(hours=cap_hours), settings
    ) == 0


def test_past_the_closure_cap_a_thin_row_is_resolved_and_flagged(db, settings, inst, atr, price):
    """The escape valve: beyond the simulator's own closure cap, missing M1 is a real
    data gap. The row resolves degraded and is flagged, rather than sitting forever."""
    _base, cap_hours = horizon_bounds(settings, _GRAN)
    t = _T_UNCOVERED
    trade = _make_row(db, inst, atr, price, t=t)

    summary = _resolve(
        db, settings, inst, band=_UNCOVERED_BAND, now=t + timedelta(hours=cap_hours + 1)
    )

    assert summary["resolved"] == 1
    assert summary["forced"] == 1, "a thin resolution must be counted as degraded"
    db.refresh(trade)
    assert trade.closed_at is not None
    assert trade.ambiguous_resolution is True, "a degraded resolution must be flagged"


def test_observable_bars_counts_trading_bars_not_wall_clock(db, settings, inst):
    """The coverage test mirrors the simulator's bar counter: a window with real M1
    yields >= SIGNAL_MAX_HOLD_BARS observable bars; a future window yields none."""
    _base, cap_hours = horizon_bounds(settings, _GRAN)
    covered = res.observable_bars(
        db, inst.id, _T_BASE, _GRAN, _T_BASE + timedelta(hours=cap_hours), settings
    )
    assert covered >= int(settings.SIGNAL_MAX_HOLD_BARS)

    assert res.observable_bars(
        db, inst.id, _T_UNCOVERED, _GRAN, _T_UNCOVERED + timedelta(hours=cap_hours), settings
    ) == 0


# ── 4. BOTH decisions resolve — the skip cohort is the counterfactual ────────
def test_skip_rows_are_resolved_too(db, settings, inst, atr, price):
    skipped = _make_row(db, inst, atr, price, ml_decision=DECISION_SKIP)

    summary = _resolve(db, settings, inst)

    assert summary["resolved"] == 1
    assert summary["by_decision"]["skip"] == 1
    db.refresh(skipped)
    assert skipped.outcome is not None and skipped.rr_actual is not None


def test_unscored_rows_are_resolved_too(db, settings, inst, atr, price):
    """ml_decision IS NULL rows (scoring failures) still deserve an honest outcome —
    they are excluded from the cohort comparison at REPORTING time, not here."""
    unscored = _make_row(db, inst, atr, price, ml_decision=None)

    summary = _resolve(db, settings, inst)

    assert summary["resolved"] == 1
    assert summary["by_decision"]["unscored"] == 1
    db.refresh(unscored)
    assert unscored.outcome is not None


def test_take_skip_and_unscored_resolve_in_one_pass(db, settings, inst, atr, price):
    _make_row(db, inst, atr, price, t=_T_BASE, ml_decision=DECISION_TAKE)
    _make_row(db, inst, atr, price, t=_T_BASE + timedelta(hours=4), ml_decision=DECISION_SKIP)
    _make_row(db, inst, atr, price, t=_T_BASE + timedelta(hours=8), ml_decision=None)

    summary = _resolve(db, settings, inst)

    assert summary["resolved"] == 3
    assert summary["by_decision"] == {"take": 1, "skip": 1, "unscored": 1}
    assert sum(summary["by_outcome"].values()) == 3


# ── 5. failure isolation ────────────────────────────────────────────────────
def test_one_bad_row_does_not_stall_the_queue(db, settings, inst, atr, price, monkeypatch):
    """A per-row exception is caught, counted and logged; every sibling still resolves."""
    bad = _make_row(db, inst, atr, price, t=_T_BASE)
    good_a = _make_row(db, inst, atr, price, t=_T_BASE + timedelta(hours=4))
    good_b = _make_row(db, inst, atr, price, t=_T_BASE + timedelta(hours=8))
    real_simulate = res.simulate

    def _wrapper(db_, settings_, instrument_id, direction, entry, stop, target, signal_time,
                 atr14, granularity, **kwargs):
        if signal_time == _T_BASE:
            raise RuntimeError("simulator exploded")
        return real_simulate(db_, settings_, instrument_id, direction, entry, stop,
                             target, signal_time, atr14, granularity, **kwargs)

    monkeypatch.setattr(res, "simulate", _wrapper)

    summary = _resolve(db, settings, inst)

    assert summary["checked"] == 3
    assert summary["failed"] == 1
    assert summary["resolved"] == 2
    assert summary["still_pending"] == 1
    db.refresh(bad); db.refresh(good_a); db.refresh(good_b)
    assert bad.closed_at is None, "the failing row stays pending, never half-written"
    assert bad.outcome is None
    assert good_a.closed_at is not None and good_b.closed_at is not None


def test_failed_row_is_retried_on_the_next_pass(db, settings, inst, atr, price, monkeypatch):
    trade = _make_row(db, inst, atr, price)
    real_simulate = res.simulate
    monkeypatch.setattr(res, "simulate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert _resolve(db, settings, inst)["failed"] == 1

    monkeypatch.setattr(res, "simulate", real_simulate)
    assert _resolve(db, settings, inst)["resolved"] == 1

    db.refresh(trade)
    assert trade.outcome is not None


def test_null_atr_leaves_the_row_pending(db, settings, inst, atr, price):
    """atr14 is the trailing-stop distance basis. Inventing one would invent an exit,
    and therefore a training label — so the row waits instead.

    JSON null is the REALISTIC shape of a missing ATR on a persisted row: the recorder
    runs ``feature_builder.json_safe``, which turns a NaN feature into null (Postgres
    rejects the raw NaN token outright)."""
    trade = _make_row(db, inst, atr, price, atr_value=None)

    summary = _resolve(db, settings, inst)

    assert summary["no_atr"] == 1 and summary["resolved"] == 0
    db.refresh(trade)
    assert trade.closed_at is None and trade.outcome is None


@pytest.mark.parametrize("bad_atr", [0.0, -1.0])
def test_non_positive_atr_leaves_the_row_pending(db, settings, inst, atr, price, bad_atr):
    _make_row(db, inst, atr, price, atr_value=bad_atr)

    summary = _resolve(db, settings, inst)

    assert summary["no_atr"] == 1 and summary["resolved"] == 0


@pytest.mark.parametrize(
    "value", [None, 0.0, -1.0, float("nan"), float("inf"), "not-a-number", {}]
)
def test_atr_extraction_rejects_every_unusable_value(value):
    """Unit-level guard on the ATR reader for shapes the DB cannot even hold (NaN,
    inf) plus the ones it can. Anything unusable must yield None -> row stays pending."""
    class _Row:
        signal_reasoning = {"atr14": value}

    assert res._atr_of(_Row()) is None


def test_atr_extraction_accepts_a_valid_value():
    class _Row:
        signal_reasoning = {"atr14": 0.0012}

    assert res._atr_of(_Row()) == pytest.approx(0.0012)


# ── 6. blast radius: the backtest corpus is never touched ───────────────────
def test_resolver_ignores_non_shadow_stages(db, settings, inst, atr, price):
    """An unscoped, whole-queue pass (exactly what the scheduled job runs) must leave
    the 7k-row backtest corpus completely alone — the selector is stage='shadow'."""
    backtest_before = db.query(Trade).filter(Trade.stage == "backtest").count()
    fingerprint_before = _corpus_fingerprint(db)
    shadow_row = _make_row(db, inst, atr, price)

    summary = res.resolve_pending(db, settings)

    assert summary["resolved"] >= 1
    db.refresh(shadow_row)
    assert shadow_row.closed_at is not None
    assert db.query(Trade).filter(Trade.stage == "backtest").count() == backtest_before
    assert _corpus_fingerprint(db) == fingerprint_before


def _corpus_fingerprint(db) -> tuple:
    """Cheap invariant over the backtest corpus: row count plus summed rr/outcome
    counts. Any write by the resolver would move at least one of these."""
    from sqlalchemy import func

    return db.query(
        func.count(Trade.id),
        func.sum(Trade.rr_actual),
        func.count(Trade.closed_at),
        func.count(Trade.outcome),
    ).filter(Trade.stage == "backtest").one()


def test_label_outcome_convention_matches_the_backtest_corpus(db):
    """The shadow label and the M7 training label must be the SAME function. Verified
    against the persisted corpus rather than by re-reading the source."""
    corpus = (
        db.query(Trade.rr_actual, Trade.outcome)
        .filter(Trade.stage == "backtest", Trade.rr_actual.isnot(None))
        .limit(500)
        .all()
    )
    assert corpus, "the M7 backtest corpus is required for this comparison"
    for rr, outcome in corpus:
        assert label_outcome(float(rr)) == outcome


@pytest.mark.parametrize(
    "rr, expected",
    [(2.0, "win"), (0.06, "win"), (0.05, "breakeven"), (0.0, "breakeven"),
     (-0.05, "breakeven"), (-0.06, "loss"), (-1.0, "loss")],
)
def test_label_outcome_boundaries(rr, expected):
    """The +/-0.05R convention, pinned explicitly at its boundaries."""
    assert label_outcome(rr) == expected


# ── 8. M1 DENSITY FLOOR (QA HIGH 3) ─────────────────────────────────────────
#
# Before this, a trading-TF bucket counted as observable if it held ONE complete
# Bid+Ask minute. A bucket with 3 of 240 minutes was therefore worth exactly as much
# as a full one, so ``simulate`` could walk a stream full of holes, sail past the
# minute price actually touched SL or TP, and write a CONFIDENT-BUT-WRONG label. That
# is the worst failure mode available here: silent, and it contaminates training data.
#
# The floor is a FRACTION of the timeframe's own minute count, which is what keeps it
# timeframe-agnostic — 0.2 means 48/240 minutes on H4 and 288/1440 on D1 with no
# per-timeframe table anywhere.
def test_min_bars_per_bucket_is_a_fraction_of_the_timeframes_own_minutes(settings):
    minutes = int(get_timeframe(_GRAN).period_hours * 60)

    half = res.min_bars_per_bucket(
        settings.model_copy(update={"SHADOW_MIN_BUCKET_M1_DENSITY": 0.5}), _GRAN
    )

    assert half == minutes // 2
    # Derived per timeframe, never a table: D1 has 6x the minutes of H4.
    d1 = res.min_bars_per_bucket(
        settings.model_copy(update={"SHADOW_MIN_BUCKET_M1_DENSITY": 0.5}), "D"
    )
    assert d1 == int(get_timeframe("D").period_hours * 60) // 2


def test_min_bars_per_bucket_never_drops_below_one(settings):
    """A bucket with no complete Bid+Ask minute can never be observable, whatever the
    configured density — density 0 must not mean 'an empty bucket counts'."""
    assert res.min_bars_per_bucket(
        settings.model_copy(update={"SHADOW_MIN_BUCKET_M1_DENSITY": 0.0}), _GRAN
    ) == 1


def test_a_thin_bucket_does_not_count_toward_observability(db, settings, inst):
    """The branch the QA finding is about: the SAME window, counted with a floor of
    one bar vs a floor of nearly-every-minute. Real stored M1, no synthetic data.

    With an impossible floor every bucket is disqualified, so coverage collapses —
    proving buckets are being judged on density and not merely on presence.
    """
    _base, cap_hours = horizon_bounds(settings, _GRAN)
    window_end = _T_BASE + timedelta(hours=cap_hours)

    permissive = res.observable_bars(
        db, inst.id, _T_BASE, _GRAN, window_end,
        settings.model_copy(update={"SHADOW_MIN_BUCKET_M1_DENSITY": 0.0}),
    )
    strict = res.observable_bars(
        db, inst.id, _T_BASE, _GRAN, window_end,
        settings.model_copy(update={"SHADOW_MIN_BUCKET_M1_DENSITY": 1.01}),
    )

    assert permissive >= int(settings.SIGNAL_MAX_HOLD_BARS)
    assert strict == 0, "an unsatisfiable density floor must disqualify every bucket"
    assert strict < permissive


def test_the_configured_floor_still_accepts_real_market_data(db, settings, inst):
    """The floor must reject holes, not normal trading. At the configured density the
    historical window still yields enough bars to resolve — otherwise the fix would
    have traded a silent mislabel for a permanently stuck queue."""
    _base, cap_hours = horizon_bounds(settings, _GRAN)

    covered = res.observable_bars(
        db, inst.id, _T_BASE, _GRAN, _T_BASE + timedelta(hours=cap_hours), settings
    )

    assert covered >= int(settings.SIGNAL_MAX_HOLD_BARS)
    assert settings.SHADOW_MIN_BUCKET_M1_DENSITY > 0.0, "the floor must actually be armed"


def test_density_floor_keeps_a_thin_row_pending_inside_the_closure_cap(
    db, settings, inst, atr, price
):
    """Branch A — coverage exists by presence but fails on density: the row must NOT
    resolve while there is still time for real M1 to land."""
    base_hours, _cap = horizon_bounds(settings, _GRAN)
    strict = settings.model_copy(update={"SHADOW_MIN_BUCKET_M1_DENSITY": 1.01})
    trade = _make_row(db, inst, atr, price)

    summary = _resolve(
        db, strict, inst, now=_T_BASE + timedelta(hours=base_hours + 1)
    )

    assert summary["not_yet"] == 1 and summary["resolved"] == 0
    db.refresh(trade)
    assert trade.closed_at is None, "a thin-bucket row must never be silently labelled"


def test_density_floor_forced_resolution_is_always_flagged_ambiguous(
    db, settings, inst, atr, price
):
    """Branch B — past the closure cap the same thin row DOES resolve, but the label
    is marked untrustworthy.

    This is the sharp case: the M1 here is real and dense enough that the simulator
    finds a CLEAN barrier and would report ``ambiguous_resolution=False`` on its own.
    The resolver overrides that, because the coverage that would make the barrier
    trustworthy is exactly what the density floor says was missing.
    """
    _base, cap_hours = horizon_bounds(settings, _GRAN)
    strict = settings.model_copy(update={"SHADOW_MIN_BUCKET_M1_DENSITY": 1.01})
    trade = _make_row(db, inst, atr, price)

    # What the simulator alone would have said about this trade.
    unforced = simulate(
        db, settings, inst.id, trade.direction,
        float(trade.entry_price), float(trade.stop_price), float(trade.tp_price),
        _T_BASE, atr, _GRAN,
    )
    assert unforced.ambiguous_resolution is False, (
        "fixture precondition: the simulator finds a clean barrier here"
    )

    summary = _resolve(db, strict, inst, now=_T_BASE + timedelta(hours=cap_hours + 1))

    assert summary["resolved"] == 1
    assert summary["forced"] == 1
    db.refresh(trade)
    assert trade.closed_at is not None, "past the cap it must not stay pending forever"
    assert trade.ambiguous_resolution is True, (
        "a forced resolution must be flagged even when the simulator was confident"
    )


def test_a_normally_covered_row_is_not_flagged_ambiguous(db, settings, inst, atr, price):
    """The control for the test above: at the CONFIGURED density the same row resolves
    unforced and keeps the simulator's own (clean) verdict."""
    trade = _make_row(db, inst, atr, price)

    summary = _resolve(db, settings, inst)

    assert summary["resolved"] == 1
    assert summary["forced"] == 0
    db.refresh(trade)
    assert trade.ambiguous_resolution is False
