"""Shadow decision recorder — writes one observe-only ``trades`` row per live signal.

This module is the single write path for ``stage='shadow'`` rows. It is called by
``services.pipeline`` at candle close AFTER the rule engine has fired and AFTER the
live RiskEngine has rendered its verdict; it never influences either.

Safety posture (defense in depth)
---------------------------------
* Gated on ``settings.SHADOW_MODE_ENABLED`` — false → returns ``None``, writes nothing.
* Hard-fails (``RuntimeError``) if ``settings.ORDER_PLACEMENT_ENABLED`` is anything
  other than ``False``. ``ml.inference.load_model`` already refuses to load an
  unpromoted artifact under that flag; this is the second, independent check at the
  WRITE site, so a shadow row can never be produced by a process that is also
  permitted to trade. A plain ``assert`` is deliberately NOT used — asserts vanish
  under ``python -O``.
* Imports no broker module and calls no broker method. Position sizing is read off
  the RiskEngine result the caller already computed; nothing here sizes or orders.

Idempotency
-----------
The natural key of a shadow row is ``(instrument_id, opened_at, stage='shadow')`` —
``opened_at`` is the signal timestamp T, which is deterministic for a given decision
bar, so re-running the candle-close pipeline for the same bar must not double-write.
Enforced twice:

* **query-before-insert** in :func:`record_shadow_decision` (the normal path — cheap,
  and lets the caller distinguish "already recorded" from "written" via a ``None``
  return), and
* a **partial unique index** ``uq_trades_shadow_natural_key`` on
  ``(instrument_id, opened_at) WHERE stage = 'shadow'`` (migration
  ``2026_08_01_shadow_trade_unique``) as the durable backstop against a race between
  two pipeline invocations. An ``IntegrityError`` from that index is caught, rolled
  back and treated as "already recorded".

The index is partial so it constrains ONLY shadow rows — the existing
``stage='backtest'`` corpus legitimately repeats an ``(instrument_id, opened_at)``
pair across backtest re-runs and must not be constrained.

NaN-probability policy
----------------------
``inference.decide`` raises ``ValueError`` on a NaN probability by design (a NaN must
never silently become 'skip'). At this call site that exception — and any other
scoring failure — is CAUGHT and the row is still written, with
``ml_probability = NULL``, ``ml_decision = NULL``, ``ml_model_id`` set (so the failing
artifact is still identified) and the error string in
``signal_reasoning['shadow']['ml_error']``.

Rationale: dropping the row would silently punch a hole in the shadow corpus exactly
on the signals whose features were hardest to compute — a biased sample. Writing it
with a NULL decision keeps the observation (features + risk verdict + eventual Phase-3
outcome) while ``ml_decision IS NULL`` cleanly excludes it from any filter-performance
comparison. The live loop always survives.

NaN-FEATURE policy (distinct from the above)
--------------------------------------------
A NaN *probability* is a scoring failure. A NaN *feature* is worse and quieter: the
model scores perfectly happily on a crippled vector (XGBoost treats NaN as a learned
missing-value branch), so nothing fails and nothing warns — but the row is no longer
comparable to the M7 training corpus, where every ``FEATURE_KEYS_MODEL`` key is 100%
populated. That is exactly how a stale FRED feed silently degraded every live shadow
row for three weeks.

So every row records ``nan_model_features`` — the count of NaN model-core features —
in its shadow sub-dict, and a count above ``settings.SHADOW_MAX_NAN_MODEL_FEATURES``
raises a WARNING naming the offending keys. The row is STILL written: a hole in the
corpus is worse than a flagged row, and the stored count makes such rows
unambiguously identifiable and excludable downstream.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.domain.execution_mode import (
    EXECUTION_MODE_OBSERVE,
    EXECUTION_MODE_SANDBOX,
    normalise as normalise_execution_mode,
)
from app.config import Settings
from app.domain.timeframes import get_timeframe
from app.models.candle import Candle
from app.models.instrument import Instrument
from app.models.trade import Trade
from app.services.feature_builder import FEATURE_KEYS_MODEL, json_safe
from app.services.ml import inference as inf
from app.services.ml.inference import LoadedModel
from app.services.risk_engine import ValidatedSignal
from app.services.signal_engine.base import SignalOutput

logger = logging.getLogger(__name__)

# ── persisted literals (named, never inline) ─────────────────────────────────
STAGE_SHADOW = "shadow"
# A row whose signal was actually sent to a broker (practice account).
STAGE_SANDBOX = "sandbox"
# The rule engine authored the signal; the model only observed it. A future
# ML-authored signal would carry a different source — this is not "ml".
SIGNAL_SOURCE_RULE_BASED = "rule_based"
STOP_METHOD_ATR = "atr"
# Namespaced sub-dict inside signal_reasoning holding shadow-run metadata. The
# feature keys themselves stay FLAT at the top level so the S1 dataset extractor
# reads a shadow row exactly as it reads a backtest row.
REASONING_SHADOW_KEY = "shadow"
# Sub-key of that dict holding how many FEATURE_KEYS_MODEL entries were NaN. Named
# so a downstream query can exclude degraded rows without re-deriving the count.
NAN_MODEL_FEATURES_KEY = "nan_model_features"
# Companion sub-key naming WHICH model-core keys were NaN — the count says a row is
# degraded, this says whether it was one dead FRED series or a broad outage.
NAN_MODEL_FEATURE_KEYS_KEY = "nan_model_feature_keys"

# Mid is the decision series (Bid/Ask are execution-side only).
_PRICE_TYPE_MID = "M"
# T must be STRICTLY after the decision bar's close — feature_builder filters
# `timestamp <= T` and candles.timestamp is bar-OPEN, so landing exactly on the
# boundary would pull in the following (not-yet-decided) bar. Same convention and
# same epsilon the M7 runner uses (backtester/runner.py: `bar_close + 1 second`).
_SIGNAL_TIME_EPSILON = timedelta(seconds=1)


# ── risk verdict container ───────────────────────────────────────────────────
@dataclass(frozen=True)
class RiskAssessment:
    """The live RiskEngine's verdict on a signal, captured for the shadow row.

    The caller already runs ``RiskEngine.validate`` for the real signal path; this
    just carries that outcome across so the recorder never re-validates (which would
    duplicate the broker pip-value call).

    Attributes:
        passed: True when ``RiskEngine.validate`` returned a ``ValidatedSignal``.
        rejection_reason: the ``RiskValidationError`` code (e.g. ``"STOP_TOO_WIDE"``)
            when ``passed`` is False, else ``None``.
        units: position size the risk engine derived (0 when rejected — a rejected
            signal has no size).
        risk_amount: account currency at risk the sizing was based on.
        pip_value: broker-fetched pip value used for sizing, or ``None`` when the
            signal was rejected before sizing.
    """

    passed: bool
    rejection_reason: Optional[str]
    units: int
    risk_amount: float
    pip_value: Optional[float]

    @classmethod
    def approved(cls, validated: ValidatedSignal) -> "RiskAssessment":
        """Build from a successful ``RiskEngine.validate`` result."""
        return cls(
            passed=True,
            rejection_reason=None,
            units=int(validated.units),
            risk_amount=float(validated.risk_amount),
            pip_value=float(validated.pip_value),
        )

    @classmethod
    def rejected(cls, reason: str, risk_amount: float) -> "RiskAssessment":
        """Build from a ``RiskValidationError``.

        Args:
            reason: the error code the risk engine raised.
            risk_amount: what WOULD have been risked (``balance * RISK_PCT_PER_TRADE``)
                — recorded so a rejected row is still comparable to an approved one.
        """
        return cls(
            passed=False,
            rejection_reason=str(reason),
            units=0,
            risk_amount=float(risk_amount),
            pip_value=None,
        )


# ── signal-time derivation (live feature parity) ─────────────────────────────
def live_signal_time(
    db: Session, instrument: Instrument, granularity: str
) -> Optional[datetime]:
    """Derive T — the point-in-time instant for a LIVE signal's feature vector.

    The live rule engine's decision bar is the newest stored Mid candle for
    ``(instrument, granularity)`` (the broker client discards incomplete candles, so
    the newest stored bar is always a CLOSED bar). ``candles.timestamp`` is bar-OPEN,
    therefore::

        bar_close = newest_bar_open + timeframe.period
        T         = bar_close + 1 second      # strictly after the close

    This reproduces the M7 runner's convention exactly (``t_sig = bar_close +
    timedelta(seconds=1)``), which is what makes a live shadow feature row and a
    backtest training row comparable rather than merely similar.

    Args:
        db: SQLAlchemy session.
        instrument: the instrument whose decision bar is being resolved.
        granularity: the trading timeframe the signal fired on (e.g. ``"H4"``).

    Returns:
        Naive-UTC T, or ``None`` when no Mid candle exists yet for this
        (instrument, granularity) — in which case the engine could not have fired
        and there is nothing to record.

    Raises:
        ValueError: ``granularity`` is not in the Timeframe registry. Resolved BEFORE
            the candle query so an unregistered timeframe fails loud instead of
            masquerading as "no data".
    """
    period = timedelta(hours=get_timeframe(granularity).period_hours)
    newest = (
        db.query(Candle.timestamp)
        .filter(
            Candle.instrument_id == instrument.id,
            Candle.granularity == granularity,
            Candle.price_type == _PRICE_TYPE_MID,
        )
        .order_by(Candle.timestamp.desc())
        .first()
    )
    if newest is None:
        return None
    return _naive(newest[0]) + period + _SIGNAL_TIME_EPSILON


# ── public API ───────────────────────────────────────────────────────────────
def record_shadow_decision(
    db: Session,
    settings: Settings,
    instrument: Instrument,
    signal: SignalOutput,
    features: dict,
    loaded_model: LoadedModel,
    *,
    signal_time: datetime,
    risk: RiskAssessment,
    stage: Optional[str] = None,
) -> Optional[Trade]:
    """Score a live signal with the S1 artifact and persist ONE shadow ``Trade`` row.

    Both decisions are recorded: a ``'skip'`` is as informative as a ``'take'`` when
    measuring the filter. Outcome columns (``outcome``, ``rr_actual``, ``exit_price``,
    ``exit_reason``, ``closed_at``) are left NULL — Phase 3 resolves them.

    Args:
        db: SQLAlchemy session (committed on success).
        settings: config supplying ``SHADOW_MODE_ENABLED``, ``ORDER_PLACEMENT_ENABLED``
            and ``ML_DECISION_THRESHOLD``.
        instrument: the signalled instrument.
        signal: the rule engine's ``SignalOutput`` (direction/entry/stop/target/score).
        features: the dict returned by ``feature_builder.build_features`` for
            ``signal_time`` — model tier + gated tier + risk payload +
            ``feature_schema_version``. Persisted in full.
        loaded_model: the validated artifact from ``ml.inference.load_model``.
        signal_time: T, from :func:`live_signal_time` — becomes ``trades.opened_at``
            and is half of the idempotency natural key.
        risk: the live RiskEngine verdict for this signal.

    Returns:
        The persisted :class:`Trade`, or ``None`` when shadow mode is disabled or a
        row for this ``(instrument, signal_time)`` already exists.

    Raises:
        RuntimeError: ``settings.ORDER_PLACEMENT_ENABLED`` is not ``False``. Shadow
            observation is only meaningful — and only safe — in a process that is
            structurally forbidden from trading.
    """
    if not settings.SHADOW_MODE_ENABLED:
        return None
    _assert_order_placement_disabled(settings, stage)

    if _existing_shadow_row(db, instrument.id, signal_time, stage) is not None:
        logger.debug(
            "shadow: row already exists for %s @ %s — skipping duplicate write",
            instrument.symbol, signal_time,
        )
        return None

    nan_keys = _nan_model_features(features)
    _warn_if_degraded(settings, instrument, signal_time, nan_keys)
    decision, ml_error = _score_safely(loaded_model, features, settings)

    # Which stage this row belongs to. Derived HERE because this is the only point
    # that holds the decision, the risk verdict and the execution mode together —
    # and the stage must tell the truth about whether an order will be sent.
    #   sandbox mode + take + risk approved  -> 'sandbox' (an order follows)
    #   anything else                        -> 'shadow'  (none will)
    # An explicit `stage` argument still wins, so tests and backfills can pin it.
    if stage is None:
        stage = _derive_stage(settings, decision, risk)

    stop_distance = abs(signal.entry - signal.stop)
    reward = abs(signal.target - signal.entry)
    pip_size = instrument.pip_size or 0.0
    trade = Trade(
        instrument_id=instrument.id,
        direction=signal.direction,
        entry_price=signal.entry,
        exit_price=None,
        stop_price=signal.stop,
        tp_price=signal.target,
        units=risk.units,
        risk_amount=risk.risk_amount,
        expected_pip_loss=(stop_distance / pip_size) if pip_size else 0.0,
        actual_pip_loss=None,
        rr_entry=(reward / stop_distance) if stop_distance else 0.0,
        rr_actual=None,
        signal_source=SIGNAL_SOURCE_RULE_BASED,
        stage=stage,
        outcome=None,          # Phase 3
        exit_reason=None,      # Phase 3
        ambiguous_resolution=False,
        opened_at=signal_time,
        closed_at=None,        # Phase 3
        signal_reasoning=_build_reasoning(
            features, signal, settings, loaded_model, signal_time, risk, ml_error, nan_keys
        ),
        confluence_score=signal.confidence_score,
        session=_session_of(features),
        stop_method=STOP_METHOD_ATR,
        ml_probability=decision.probability if decision is not None else None,
        ml_decision=decision.decision if decision is not None else None,
        ml_model_id=loaded_model.model_id,
    )
    db.add(trade)
    try:
        db.commit()
    except IntegrityError:
        # Lost a race against a concurrent pipeline run — the partial unique index
        # did its job. Treat exactly like the query-before-insert hit.
        db.rollback()
        logger.debug(
            "shadow: concurrent write already recorded %s @ %s", instrument.symbol, signal_time
        )
        return None
    db.refresh(trade)

    logger.info(
        "shadow recorded: %s %s %s p=%s decision=%s risk=%s model=%s nan_model_features=%d",
        instrument.symbol, signal.granularity, signal.direction,
        f"{decision.probability:.4f}" if decision is not None else "n/a",
        decision.decision if decision is not None else f"ERROR({ml_error})",
        "pass" if risk.passed else f"reject:{risk.rejection_reason}",
        loaded_model.model_id, len(nan_keys),
    )
    return trade


# ── internals ────────────────────────────────────────────────────────────────
def _derive_stage(settings: Settings, decision, risk) -> str:
    """Pick the stage a row belongs to. See the call site for the reasoning."""
    mode = normalise_execution_mode(getattr(settings, "EXECUTION_MODE", ""))
    if mode != EXECUTION_MODE_SANDBOX:
        return STAGE_SHADOW
    if decision is None or decision.decision != "take":
        return STAGE_SHADOW
    if not risk.passed or risk.units <= 0:
        # A risk-rejected signal is never traded, whatever the model thinks, so no
        # order follows and 'shadow' is the honest label.
        return STAGE_SHADOW
    return STAGE_SANDBOX


def _assert_order_placement_disabled(settings: Settings, stage: str) -> None:
    """Second, independent safety check at the WRITE site (see module docstring).

    The invariant this protects is about the STAGE LABEL, not about orders in
    general: a ``stage='shadow'`` row asserts "this decision was observed and no
    order was placed". If that claim can be false, every shadow analysis is
    measuring something other than what it says.

    Originally that meant forbidding orders outright, because shadow was the only
    mode. With sandbox, a 'take' is written as ``stage='sandbox'`` (an order WAS
    placed) and a 'skip' stays ``stage='shadow'`` (none was) — so the claim stays
    true in both, and the check narrows to exactly the case that would break it:
    a shadow-labelled row written by a process permitted to trade.
    """
    if stage != STAGE_SHADOW:
        return
    mode = normalise_execution_mode(getattr(settings, "EXECUTION_MODE", ""))
    if mode == EXECUTION_MODE_OBSERVE:
        return
    # In sandbox/live a shadow-labelled row is only honest for a decision that
    # placed no order — i.e. a skip, or a risk-rejected signal. The caller routes
    # takes to stage='sandbox'; reaching here with a take means that routing broke.
    if settings.ORDER_PLACEMENT_ENABLED is True:
        raise RuntimeError(
            f"refusing to write a stage='{STAGE_SHADOW}' row while "
            f"EXECUTION_MODE={mode!r} and ORDER_PLACEMENT_ENABLED is True. A shadow "
            "row asserts that NO order was placed; a process permitted to trade must "
            "route taken signals to stage='sandbox' instead. This is a routing bug, "
            "not a configuration one."
        )


def _existing_shadow_row(
    db: Session, instrument_id: int, signal_time: datetime, stage: str = STAGE_SHADOW
) -> Optional[int]:
    """Natural-key guard: one shadow row per (instrument, signal_time)."""
    row = (
        db.query(Trade.id)
        .filter(
            Trade.instrument_id == instrument_id,
            Trade.stage == stage,
            Trade.opened_at == signal_time,
        )
        .first()
    )
    return None if row is None else int(row[0])


def _nan_model_features(features: dict) -> list[str]:
    """Which ``FEATURE_KEYS_MODEL`` entries are NaN (or absent) in this feature dict.

    A missing key counts as NaN: the model's encoder would supply NaN for it anyway,
    so an absent key and a NaN key degrade the vector identically and must be
    reported identically. Non-numeric values (the categorical ``instrument_category``
    / ``session`` strings) are NaN only if they are literally ``None``.

    Args:
        features: the dict returned by ``feature_builder.build_features``.

    Returns:
        The offending keys, in ``FEATURE_KEYS_MODEL`` order (stable, so two rows with
        the same degradation produce the same list).
    """
    degraded: list[str] = []
    for key in FEATURE_KEYS_MODEL:
        if key not in features:
            degraded.append(key)
            continue
        value = features[key]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            degraded.append(key)
    return degraded


def _warn_if_degraded(
    settings: Settings,
    instrument: Instrument,
    signal_time: datetime,
    nan_keys: list[str],
) -> None:
    """WARN when a live row carries more NaN model features than config tolerates.

    Deliberately does NOT block the write — see the module docstring's NaN-FEATURE
    policy. The warning is the operator's signal that an upstream feed (typically a
    macro series) has gone stale; the stored count is the analyst's filter.
    """
    if len(nan_keys) <= int(settings.SHADOW_MAX_NAN_MODEL_FEATURES):
        return
    logger.warning(
        "shadow: %s @ %s has %d/%d NaN model-core features (max %d) — row is RECORDED "
        "but is NOT comparable to the training corpus; check macro feed freshness. "
        "NaN keys: %s",
        instrument.symbol, signal_time, len(nan_keys), len(FEATURE_KEYS_MODEL),
        int(settings.SHADOW_MAX_NAN_MODEL_FEATURES), ", ".join(nan_keys),
    )


def _score_safely(
    loaded_model: LoadedModel, features: dict, settings: Settings
) -> tuple[Optional[inf.ShadowDecision], Optional[str]]:
    """Score without ever letting the live loop die.

    ``inference.decide`` raises ``ValueError`` on a NaN probability by design; other
    failures (a malformed feature value reaching the encoder, an unexpected model
    error) are equally non-fatal to the trading loop. Both are converted into
    ``(None, error_string)`` so the row is still written with a NULL decision and an
    explicit error marker — see the module docstring for the rationale.
    """
    try:
        return inf.score_and_decide(loaded_model, features, settings), None
    except Exception as exc:  # noqa: BLE001 — a scoring failure must never halt live
        logger.warning(
            "shadow: scoring failed for model=%s (%s: %s) — recording row with NULL "
            "probability/decision",
            loaded_model.model_id, type(exc).__name__, exc,
        )
        return None, f"{type(exc).__name__}: {exc}"


def _build_reasoning(
    features: dict,
    signal: SignalOutput,
    settings: Settings,
    loaded_model: LoadedModel,
    signal_time: datetime,
    risk: RiskAssessment,
    ml_error: Optional[str],
    nan_keys: list[str],
) -> dict:
    """Assemble ``signal_reasoning``: the FULL feature dict, flat, plus shadow metadata.

    The model / gated / payload keys and ``feature_schema_version`` stay at the TOP
    level so the S1 dataset extractor reads a shadow row byte-identically to a
    backtest row. Everything shadow-specific — including the RiskEngine verdict that
    makes a would-have-been-REJECTED signal distinguishable from an accepted one, and
    the model-feature NaN count that makes a degraded row excludable — lives under the
    single namespaced ``"shadow"`` key.
    """
    return json_safe(
        {
            **features,
            REASONING_SHADOW_KEY: {
                "signal_time": signal_time.isoformat(),
                "recorded_at": datetime.utcnow().isoformat(),
                "granularity": signal.granularity,
                # RiskEngine verdict — the discriminator Phase 3 filters on.
                "risk_passed": risk.passed,
                "risk_rejection_reason": risk.rejection_reason,
                "risk_units": risk.units,
                "risk_pip_value": risk.pip_value,
                # Feature-vector integrity — how many model-core inputs were missing.
                # 0 means this row is directly comparable to an M7 training row.
                NAN_MODEL_FEATURES_KEY: len(nan_keys),
                NAN_MODEL_FEATURE_KEYS_KEY: nan_keys,
                # ML provenance (mirrors the ml_* columns; kept here so a JSON-only
                # export of the corpus is self-describing).
                "ml_model_id": loaded_model.model_id,
                "ml_decision_threshold": float(settings.ML_DECISION_THRESHOLD),
                "ml_error": ml_error,
            },
        }
    )


def _session_of(features: dict) -> Optional[str]:
    value = features.get("session")
    return value if isinstance(value, str) and value else None


def _naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt
