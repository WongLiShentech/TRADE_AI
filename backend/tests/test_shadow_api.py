"""Integration tests for app.api.v1.shadow — M8-Shadow Phase 3 API.

The cohort arithmetic in ``/performance`` is the number the whole shadow experiment
is judged on, so it is verified against HAND-COMPUTED expectations from a fixture of
known R-multiples — not against a re-implementation of the same formulas, which
would only prove the code agrees with itself.

Fixture cohort (all resolved unless stated):

    model_take :  +2.0, -1.0, +1.5      -> 2 wins / 3
    model_skip :  -1.0, -1.0            -> 0 wins / 2
    unscored   :  +1.0 resolved, 1 pending   (excluded from every cohort)
    plus one PENDING take row                 (counted, never measured)

Runs against the live dev Postgres via the app's own ``get_db``; an autouse fixture
purges the test symbol's shadow rows before and after each test, so the real corpus
(``stage='backtest'``) is never touched.

Run from backend/:  python -m pytest tests/test_shadow_api.py -v
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import shadow as api
from app.main import app
from app.models.trade import Trade
from app.services.ml.inference import DECISION_SKIP, DECISION_TAKE
from app.services.shadow import recorder as rec

_SYMBOL = "GBP_USD"
_OTHER_SYMBOL = "EUR_USD"
_GRAN = "H4"
_T_BASE = datetime(2025, 9, 1, 12, 0, 1)
# SENTINEL TIME BAND. Every row this suite writes has ``opened_at`` inside
# [_T_BASE, _T_BASE + _SENTINEL_WINDOW], and BOTH the purge fixture and the query
# helpers are bounded to it.
#
# This bound is a safety requirement, not a tidiness one. The purge runs against the
# real dev database through the app's own SessionLocal, on live-active symbols; an
# unbounded ``instrument_id = X AND stage = 'shadow'`` DELETE would irreversibly
# destroy real live shadow observations every time anyone ran pytest. Bounding it to
# the suite's own historical sentinel band makes that impossible by construction.
#
# It also keeps the assertions honest: the cohort maths below is hand-computed over
# exactly these rows, so a real live row leaking into a count would fail the suite
# for no real reason.
_SENTINEL_WINDOW = timedelta(days=30)
_SENTINEL_END = _T_BASE + _SENTINEL_WINDOW
_HOLD_HOURS = 8.0

_DECISIONS = "/api/v1/shadow/decisions"
_PERFORMANCE = "/api/v1/shadow/performance"

# Hand-computed expectations for the fixture above.
_TAKE_RRS = [2.0, -1.0, 1.5]
_SKIP_RRS = [-1.0, -1.0]
_EXPECTED = {
    "take_win_rate": 2 / 3,
    "take_expectancy": (2.0 - 1.0 + 1.5) / 3,          # 0.8333...
    "take_profit_factor": (2.0 + 1.5) / 1.0,            # 3.5
    "skip_win_rate": 0.0,
    "skip_expectancy": -1.0,
    "skip_profit_factor": 0.0,                          # zero gross win, 2.0 gross loss
    "all_expectancy": (2.0 - 1.0 + 1.5 - 1.0 - 1.0) / 5,  # 0.1
    "all_profit_factor": 3.5 / 3.0,                      # 1.1666...
    "all_win_rate": 2 / 5,
    "keep_rate": 3 / 5,
}


# ── fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    """TestClient WITHOUT the context manager: FastAPI only runs the lifespan on
    __enter__, so the scheduler and the price stream never start during tests."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clean_shadow_rows(db, instrument):
    inst = instrument(_SYMBOL)
    other = instrument(_OTHER_SYMBOL)

    def _purge():
        db.query(Trade).filter(
            Trade.instrument_id.in_([inst.id, other.id]),
            Trade.stage == rec.STAGE_SHADOW,
            # TIME-BOUNDED — see _SENTINEL_WINDOW. Never remove this clause: without
            # it this DELETE wipes real live shadow rows on every pytest run.
            Trade.opened_at >= _T_BASE,
            Trade.opened_at <= _SENTINEL_END,
        ).delete(synchronize_session=False)
        db.commit()

    _purge()
    yield
    _purge()


@pytest.fixture()
def inst(instrument):
    return instrument(_SYMBOL)


@pytest.fixture()
def cohort(db, inst):
    """Materialise the documented fixture cohort. Returns the created rows."""
    rows = []
    offset = 0
    for rr in _TAKE_RRS:
        rows.append(_row(db, inst, offset, DECISION_TAKE, rr=rr))
        offset += 1
    for rr in _SKIP_RRS:
        rows.append(_row(db, inst, offset, DECISION_SKIP, rr=rr))
        offset += 1
    rows.append(_row(db, inst, offset, None, rr=1.0))          # unscored, resolved
    offset += 1
    rows.append(_row(db, inst, offset, None, rr=None))          # unscored, pending
    offset += 1
    rows.append(_row(db, inst, offset, DECISION_TAKE, rr=None))  # take, pending
    return rows


# ── helpers ──────────────────────────────────────────────────────────────────
def _row(db, inst, hours_offset: int, ml_decision, *, rr):
    """One shadow row. ``rr=None`` leaves it PENDING (no closed_at, no outcome)."""
    from app.services.backtester.runner import label_outcome

    t = _T_BASE + timedelta(hours=4 * hours_offset)
    resolved = rr is not None
    trade = Trade(
        instrument_id=inst.id,
        direction="BUY",
        entry_price=1.3000,
        stop_price=1.2950,
        tp_price=1.3100,
        exit_price=1.3050 if resolved else None,
        units=1000,
        risk_amount=2.0,
        expected_pip_loss=50.0,
        rr_entry=2.0,
        rr_actual=rr,
        signal_source="rule_based",
        stage=rec.STAGE_SHADOW,
        outcome=label_outcome(rr) if resolved else None,
        exit_reason="tp_hit" if resolved else None,
        ambiguous_resolution=False,
        opened_at=t,
        closed_at=t + timedelta(hours=_HOLD_HOURS) if resolved else None,
        signal_reasoning={
            "atr14": 0.0012,
            rec.REASONING_SHADOW_KEY: {
                "granularity": _GRAN,
                "risk_passed": True,
                "risk_rejection_reason": None,
            },
        },
        confluence_score=4,
        session="london",
        stop_method="atr",
        ml_probability=0.42 if ml_decision else None,
        ml_decision=ml_decision,
        ml_model_id="test_model",
    )
    db.add(trade)
    db.commit()
    db.refresh(trade)
    return trade


def _scoped(params: dict) -> dict:
    """Default every query to the suite's sentinel band.

    Keeps the hand-computed cohort maths valid no matter how many REAL live shadow
    rows exist for the test symbol — the endpoints' own ``start``/``end`` filters do
    the scoping, so nothing has to be deleted to make a count come out right.
    """
    params.setdefault("instrument", _SYMBOL)
    params.setdefault("start", _T_BASE.isoformat())
    params.setdefault("end", _SENTINEL_END.isoformat())
    return params


def _perf(client, **params):
    response = client.get(_PERFORMANCE, params=_scoped(params))
    assert response.status_code == 200, response.text
    return response.json()


def _decisions(client, **params):
    response = client.get(_DECISIONS, params=_scoped(params))
    assert response.status_code == 200, response.text
    return response.json()


# ── 1. /decisions — filters and pagination ──────────────────────────────────
def test_decisions_returns_every_shadow_row_for_the_instrument(client, cohort):
    body = _decisions(client)

    assert body["total"] == len(cohort)
    assert len(body["items"]) == len(cohort)
    assert {item["instrument"] for item in body["items"]} == {_SYMBOL}


def test_decisions_are_newest_signal_first(client, cohort):
    opened = [item["opened_at"] for item in _decisions(client)["items"]]

    assert opened == sorted(opened, reverse=True)


def test_decisions_filters_by_ml_decision(client, cohort):
    takes = _decisions(client, ml_decision="take")
    skips = _decisions(client, ml_decision="skip")
    unscored = _decisions(client, ml_decision="unscored")

    assert takes["total"] == len(_TAKE_RRS) + 1      # + the pending take
    assert skips["total"] == len(_SKIP_RRS)
    assert unscored["total"] == 2
    assert all(item["ml_decision"] == "take" for item in takes["items"])
    assert all(item["ml_decision"] is None for item in unscored["items"])


def test_decisions_filters_by_resolved_and_pending(client, cohort):
    resolved = _decisions(client, status="resolved")
    pending = _decisions(client, status="pending")

    assert resolved["total"] == 6           # 3 take + 2 skip + 1 unscored
    assert pending["total"] == 2            # 1 unscored + 1 take
    assert all(item["resolved"] and item["closed_at"] for item in resolved["items"])
    assert all(not item["resolved"] and item["closed_at"] is None for item in pending["items"])


def test_decisions_filters_by_date_range(client, cohort):
    """Bounds are inclusive on opened_at (the signal time T)."""
    window = _decisions(
        client,
        start=_T_BASE.isoformat(),
        end=(_T_BASE + timedelta(hours=4)).isoformat(),
    )

    assert window["total"] == 2
    assert all(item["opened_at"] <= (_T_BASE + timedelta(hours=4)).isoformat()
               for item in window["items"])


def test_decisions_paginates(client, cohort):
    first = _decisions(client, limit=3, offset=0)
    second = _decisions(client, limit=3, offset=3)

    assert first["total"] == second["total"] == len(cohort)
    assert len(first["items"]) == 3 and len(second["items"]) == 3
    assert {item["id"] for item in first["items"]}.isdisjoint(
        {item["id"] for item in second["items"]}
    )


def test_decisions_scopes_to_the_requested_instrument(client, db, instrument, cohort):
    """Instrument is a parameter — another pair's rows must never leak into a page."""
    other = instrument(_OTHER_SYMBOL)
    _row(db, other, 20, DECISION_TAKE, rr=2.0)

    assert _decisions(client)["total"] == len(cohort)
    assert _decisions(client, instrument=_OTHER_SYMBOL)["total"] == 1


def test_decisions_surfaces_the_ml_and_risk_columns(client, cohort):
    item = _decisions(client, ml_decision="take", status="resolved")["items"][0]

    assert item["ml_model_id"] == "test_model"
    assert item["ml_probability"] == pytest.approx(0.42)
    assert item["granularity"] == _GRAN
    assert item["risk_passed"] is True
    assert item["risk_rejection_reason"] is None
    assert item["holding_hours"] == pytest.approx(_HOLD_HOURS)


def test_unknown_instrument_is_404_not_an_empty_page(client):
    """A typo must fail loudly — an empty page reads as 'no signals yet'."""
    assert client.get(_DECISIONS, params={"instrument": "NOT_A_PAIR"}).status_code == 404
    assert client.get(_PERFORMANCE, params={"instrument": "NOT_A_PAIR"}).status_code == 404


def test_unknown_ml_decision_value_is_rejected(client):
    assert client.get(_DECISIONS, params={"ml_decision": "maybe"}).status_code == 422


# ── 2. /performance — the cohort math, against hand-computed values ─────────
def test_performance_take_cohort_matches_hand_computed_values(client, cohort):
    take = _perf(client)["cohorts"][api.COHORT_TAKE]

    assert take["total"] == 4            # 3 resolved + 1 pending
    assert take["count"] == 3            # metrics use the resolved sample only
    assert take["pending"] == 1
    assert take["win_rate"] == pytest.approx(_EXPECTED["take_win_rate"])
    assert take["expectancy"] == pytest.approx(_EXPECTED["take_expectancy"])
    assert take["profit_factor"] == pytest.approx(_EXPECTED["take_profit_factor"])
    assert take["avg_holding_hours"] == pytest.approx(_HOLD_HOURS)


def test_performance_skip_cohort_matches_hand_computed_values(client, cohort):
    skip = _perf(client)["cohorts"][api.COHORT_SKIP]

    assert skip["total"] == skip["count"] == 2 and skip["pending"] == 0
    assert skip["win_rate"] == pytest.approx(_EXPECTED["skip_win_rate"])
    assert skip["expectancy"] == pytest.approx(_EXPECTED["skip_expectancy"])
    assert skip["profit_factor"] == pytest.approx(_EXPECTED["skip_profit_factor"])


def test_performance_all_signals_cohort_is_take_union_skip(client, cohort):
    everything = _perf(client)["cohorts"][api.COHORT_ALL]

    assert everything["total"] == 6      # 4 take (1 pending) + 2 skip
    assert everything["count"] == 5
    assert everything["expectancy"] == pytest.approx(_EXPECTED["all_expectancy"])
    assert everything["profit_factor"] == pytest.approx(_EXPECTED["all_profit_factor"])
    assert everything["win_rate"] == pytest.approx(_EXPECTED["all_win_rate"])


def test_unscored_rows_are_excluded_from_cohorts_and_reported_separately(client, cohort):
    """A row the model never scored belongs to no decision — folding it into the
    baseline would contaminate the very comparison the endpoint exists to make."""
    body = _perf(client)

    assert body["unscored"] == {"total": 2, "resolved": 1, "pending": 1}
    cohort_total = sum(body["cohorts"][k]["total"] for k in (api.COHORT_TAKE, api.COHORT_SKIP))
    assert cohort_total == 6, "the 2 unscored rows must not appear in any cohort"
    # The unscored +1.0R would have lifted all_signals expectancy if it leaked in.
    assert body["cohorts"][api.COHORT_ALL]["expectancy"] == pytest.approx(
        _EXPECTED["all_expectancy"]
    )


def test_outcome_breakdown_uses_the_shared_metrics_buckets(client, cohort):
    take = _perf(client)["cohorts"][api.COHORT_TAKE]["outcome_breakdown"]

    # +2.0 -> full_win (>= 1.9), +1.5 -> partial, -1.0 -> loss
    assert take["full_win"] == 1
    assert take["partial"] == 1
    assert take["loss"] == 1
    assert take["breakeven"] == 0


def test_filter_effect_quantifies_the_lift(client, cohort):
    effect = _perf(client)["filter_effect"]

    assert effect["keep_rate"] == pytest.approx(_EXPECTED["keep_rate"])
    assert effect["expectancy_lift_vs_all"] == pytest.approx(
        _EXPECTED["take_expectancy"] - _EXPECTED["all_expectancy"]
    )
    assert effect["expectancy_lift_vs_skip"] == pytest.approx(
        _EXPECTED["take_expectancy"] - _EXPECTED["skip_expectancy"]
    )
    assert effect["profit_factor_lift_vs_all"] == pytest.approx(
        _EXPECTED["take_profit_factor"] - _EXPECTED["all_profit_factor"]
    )


def test_pending_rows_are_counted_but_never_measured(client, db, inst):
    """A cohort of only pending rows has no measurable metrics — null, not zero."""
    _row(db, inst, 0, DECISION_TAKE, rr=None)
    take = _perf(client)["cohorts"][api.COHORT_TAKE]

    assert take["total"] == 1 and take["pending"] == 1 and take["count"] == 0
    assert take["expectancy"] is None
    assert take["win_rate"] is None
    assert take["profit_factor"] is None


def test_empty_cohorts_report_null_not_zero(client):
    """No data must never masquerade as a measured zero."""
    body = _perf(client)

    for key in (api.COHORT_TAKE, api.COHORT_SKIP, api.COHORT_ALL):
        stats = body["cohorts"][key]
        assert stats["total"] == stats["count"] == stats["pending"] == 0
        assert stats["expectancy"] is None and stats["profit_factor"] is None
    assert body["filter_effect"]["keep_rate"] is None


def test_infinite_profit_factor_serialises_as_null(client, db, inst):
    """A cohort with wins and zero losses gives profit_factor = inf, which is honest
    in Python and meaningless in JSON — it must not be clamped to a number."""
    _row(db, inst, 0, DECISION_TAKE, rr=2.0)
    _row(db, inst, 1, DECISION_TAKE, rr=1.5)

    take = _perf(client)["cohorts"][api.COHORT_TAKE]

    assert take["count"] == 2
    assert take["profit_factor"] is None
    assert take["expectancy"] == pytest.approx(1.75)


def test_performance_respects_the_date_range(client, cohort):
    """Narrowing the window to the first two take rows (+2.0, -1.0) changes the math."""
    body = _perf(
        client,
        start=_T_BASE.isoformat(),
        end=(_T_BASE + timedelta(hours=4)).isoformat(),
    )
    take = body["cohorts"][api.COHORT_TAKE]

    assert take["count"] == 2
    assert take["expectancy"] == pytest.approx((2.0 - 1.0) / 2)
    assert take["profit_factor"] == pytest.approx(2.0 / 1.0)


def test_performance_scopes_to_the_requested_instrument(client, db, instrument, cohort):
    other = instrument(_OTHER_SYMBOL)
    _row(db, other, 30, DECISION_TAKE, rr=5.0)

    # The other pair's outsized win must not move this pair's numbers.
    assert _perf(client)["cohorts"][api.COHORT_TAKE]["expectancy"] == pytest.approx(
        _EXPECTED["take_expectancy"]
    )
    assert _perf(client, instrument=_OTHER_SYMBOL)["cohorts"][api.COHORT_TAKE][
        "expectancy"
    ] == pytest.approx(5.0)


def test_performance_ignores_the_backtest_corpus(client, cohort):
    """Only stage='shadow' rows may enter a cohort — the 7k backtest rows share the
    table and would swamp every number if the stage filter were dropped."""
    everything = _perf(client, instrument=None)["cohorts"][api.COHORT_ALL]

    assert everything["total"] == 6, "unfiltered by instrument, still shadow-only"


def test_app_exposes_both_shadow_routes():
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert _DECISIONS in paths and _PERFORMANCE in paths


# ── 4. ambiguous rows must not pollute the cohorts (QA HIGH 5) ──────────────
#
# The simulator's degraded path can write a FABRICATED flat outcome — exit at entry,
# rr_actual = 0.0, closed_at == opened_at — which labels as ``breakeven``. Folded into
# a cohort it silently drags expectancy toward zero: a measurement artefact that reads
# as a real result. These tests use exactly that fabricated shape.
_AMBIGUOUS_RR = 0.0
# Hand-computed with the ambiguous take row INCLUDED:
#   take rr = [+2.0, -1.0, +1.5, 0.0] -> mean 0.625 (vs 0.8333 clean), wins 2/4
#   all  rr = [+2.0, -1.0, +1.5, 0.0, -1.0, -1.0] -> mean 0.0833 (vs 0.1 clean)
_EXPECTED_WITH_AMBIGUOUS = {
    "take_expectancy": (2.0 - 1.0 + 1.5 + 0.0) / 4,   # 0.625
    "take_win_rate": 2 / 4,
    "take_profit_factor": (2.0 + 1.5) / 1.0,          # 3.5 — a 0.0 R is neither
    "all_expectancy": (2.0 - 1.0 + 1.5 + 0.0 - 1.0 - 1.0) / 6,   # 0.08333...
    "keep_rate": 4 / 6,
}


@pytest.fixture()
def cohort_with_ambiguous(db, inst, cohort):
    """The documented cohort PLUS one ambiguous take row shaped like the simulator's
    fabricated flat exit (rr 0.0, exit == entry, closed_at == opened_at)."""
    row = _row(db, inst, 10, DECISION_TAKE, rr=_AMBIGUOUS_RR)
    row.ambiguous_resolution = True
    row.exit_price = row.entry_price
    row.closed_at = row.opened_at
    db.commit()
    db.refresh(row)
    return cohort + [row]


def test_ambiguous_rows_are_excluded_from_every_cohort_by_default(
    client, cohort_with_ambiguous
):
    """The clean numbers must be IDENTICAL to the no-ambiguous-row fixture."""
    body = _perf(client)

    take = body["cohorts"][api.COHORT_TAKE]
    everything = body["cohorts"][api.COHORT_ALL]
    assert take["count"] == len(_TAKE_RRS)
    assert take["expectancy"] == pytest.approx(_EXPECTED["take_expectancy"])
    assert take["win_rate"] == pytest.approx(_EXPECTED["take_win_rate"])
    assert everything["expectancy"] == pytest.approx(_EXPECTED["all_expectancy"])
    assert everything["count"] == 5
    # The fabricated breakeven must not appear anywhere in the cohort breakdown.
    assert take["outcome_breakdown"].get("breakeven", 0) == 0


def test_ambiguous_rows_are_reported_in_their_own_block(client, cohort_with_ambiguous):
    """Excluded is not the same as hidden — a rising count is an M1-ingestion alarm."""
    block = _perf(client)["ambiguous"]

    assert block["included_in_cohorts"] is False
    assert block["stats"]["total"] == 1
    assert block["stats"]["count"] == 1
    assert block["stats"]["expectancy"] == pytest.approx(_AMBIGUOUS_RR)
    assert block["stats"]["outcome_breakdown"]["breakeven"] == 1


def test_include_ambiguous_folds_them_back_into_the_cohorts(client, cohort_with_ambiguous):
    """Hand-computed with the ambiguous row present — proves the flag really moves the
    numbers, and by exactly the documented amount."""
    body = _perf(client, include_ambiguous="true")

    take = body["cohorts"][api.COHORT_TAKE]
    everything = body["cohorts"][api.COHORT_ALL]
    assert take["count"] == len(_TAKE_RRS) + 1
    assert take["expectancy"] == pytest.approx(_EXPECTED_WITH_AMBIGUOUS["take_expectancy"])
    assert take["win_rate"] == pytest.approx(_EXPECTED_WITH_AMBIGUOUS["take_win_rate"])
    assert take["profit_factor"] == pytest.approx(_EXPECTED_WITH_AMBIGUOUS["take_profit_factor"])
    assert take["outcome_breakdown"]["breakeven"] == 1
    assert everything["expectancy"] == pytest.approx(_EXPECTED_WITH_AMBIGUOUS["all_expectancy"])
    assert body["filter_effect"]["keep_rate"] == pytest.approx(
        _EXPECTED_WITH_AMBIGUOUS["keep_rate"]
    )
    assert body["ambiguous"]["included_in_cohorts"] is True
    # Still reported separately, so the two views remain reconcilable.
    assert body["ambiguous"]["stats"]["total"] == 1


def test_ambiguous_dilution_is_measurable(client, cohort_with_ambiguous):
    """The point of the fix, stated as an inequality: one fabricated flat row visibly
    deflates expectancy, which is why it must not be in the default view."""
    clean = _perf(client)["cohorts"][api.COHORT_TAKE]["expectancy"]
    polluted = _perf(client, include_ambiguous="true")["cohorts"][api.COHORT_TAKE]["expectancy"]

    assert polluted < clean


def test_filter_effect_is_computed_from_the_clean_cohorts_by_default(
    client, cohort_with_ambiguous
):
    effect = _perf(client)["filter_effect"]

    assert effect["keep_rate"] == pytest.approx(_EXPECTED["keep_rate"])
    assert effect["expectancy_lift_vs_all"] == pytest.approx(
        _EXPECTED["take_expectancy"] - _EXPECTED["all_expectancy"]
    )


def test_decisions_filters_by_ambiguous(client, cohort_with_ambiguous):
    """``/decisions?ambiguous=`` isolates or excludes the untrustworthy labels."""
    only = _decisions(client, ambiguous="true")
    clean = _decisions(client, ambiguous="false")
    both = _decisions(client)

    assert only["total"] == 1
    assert all(item["ambiguous_resolution"] is True for item in only["items"])
    assert clean["total"] == len(cohort_with_ambiguous) - 1
    assert all(item["ambiguous_resolution"] is False for item in clean["items"])
    assert both["total"] == len(cohort_with_ambiguous)


def test_pending_rows_are_never_treated_as_ambiguous(client, cohort_with_ambiguous):
    """``ambiguous_resolution`` is written at resolution time, so excluding ambiguous
    rows must not quietly drop rows still awaiting an outcome."""
    pending = _decisions(client, status="pending")

    assert pending["total"] == 2
    assert all(item["ambiguous_resolution"] is False for item in pending["items"])
    assert _perf(client)["cohorts"][api.COHORT_TAKE]["pending"] == 1
