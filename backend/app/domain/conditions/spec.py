"""What a signal condition IS — the shape every rule condition declares itself in.

Why a registry rather than inline scoring
-----------------------------------------
The condition set was hardcoded in two places that had to agree and had no mechanism
forcing them to: the engine's inline BUY/SELL scoring, and
``feature_builder._CONFLUENCE_TO_PAYLOAD``. The feature builder iterated its OWN dict
rather than the breakdown handed to it, so a sixth condition would have been silently
dropped from the feature row while still influencing the trade — the failure would have
shown up as an unexplained gap between what the rule did and what the model saw.

This mirrors ``app/domain/timeframes.py`` and ``app/domain/macro_series.py``: one
declarative table, validated at import, that every consumer derives from.

Roles
-----
``VOTE``   directional — evaluated separately for BUY and SELL, contributes to the score.
``GATE``   direction-independent — a hard veto; the same answer for both sides.

Both share ONE ``evaluate(ctx, direction)`` signature; a GATE simply ignores
``direction``. So ``role`` decides how a result is CONSUMED, never how it is COMPUTED,
and the engine needs no branching on type.

That distinction is not cosmetic. ``session`` and ``spread`` are direction-independent
but were scored as votes, which means they contributed the same 2 points to both sides —
enough, at a threshold of 2, for every bar to satisfy BUY and SELL simultaneously and be
discarded as ambiguous. Naming the role is what makes that expressible.

Declared requirements
---------------------
``settings_keys`` — every ``Settings`` attribute the condition reads. Validated at import
(a typo fails loudly rather than silently reading nothing), and it is the exact list
strategy identity must cover: two configurations differing in an RSI band are different
strategies, and today they hash the same.

``required_indicators`` — the indicator columns that must be non-NULL for the condition
to be evaluable. This is what lets a new indicator column ship (nullable, unbackfilled)
without every bar being vetoed, and what stops the engine demanding ``rsi14`` for signals
whose enabled conditions never read it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Sequence


class Role(str, Enum):
    VOTE = "vote"
    GATE = "gate"


@dataclass(frozen=True)
class ConditionContext:
    """Everything a condition may read, assembled ONCE per (instrument, bar).

    Conditions are pure functions of this object and never touch the database. That is
    what makes them cheap to evaluate twice (BUY and SELL) and testable without a
    session, and it keeps every query in one place where its point-in-time bounds can be
    reviewed as a unit.

    ``now_utc`` is the decision time: ``as_of`` in a backtest, ``utcnow()`` live. ONE
    field rather than two, so a condition cannot accidentally behave differently between
    the two paths — the class of bug that made ``c3_structure`` fire on 20% of backtest
    rows and 0% of live ones.
    """

    instrument: Any                      # app.models.instrument.Instrument
    granularity: str
    now_utc: datetime
    settings: Any                        # app.config.Settings
    bars: Sequence[Any]                  # trading-timeframe Mid candles, ascending
    trend_bars: Sequence[Any]            # SIGNAL_TREND_TIMEFRAME Mid candles, ascending
    indicators: Any                      # newest app.models.indicator.Indicator <= now_utc
    quote_bid: float
    quote_ask: float
    # Precomputed once because the trend condition is evaluated for both directions and
    # the SMA is over SIGNAL_TREND_SMA_PERIOD bars.
    trend_close: Optional[float] = None
    trend_sma: Optional[float] = None

    @property
    def latest_bar(self) -> Any:
        """The decision bar — the newest trading-timeframe candle at or before now_utc."""
        return self.bars[-1]


@dataclass(frozen=True)
class Condition:
    """One named, self-describing signal condition."""

    key: str
    """Identity, and the key it occupies in ``SignalOutput.score_breakdown``."""

    role: Role

    evaluate: Callable[[ConditionContext, Optional[str]], bool]
    """``(ctx, direction) -> bool``. ``direction`` is "BUY"/"SELL" for a VOTE and None
    for a GATE. Must never raise: an input it cannot read is False ("not evaluable"),
    never True — an unevaluable condition is not a satisfied one."""

    payload_key: str
    """The ``PAYLOAD_KEYS`` name this condition is recorded under, e.g. ``c1_trend``."""

    settings_keys: tuple[str, ...] = ()
    required_indicators: tuple[str, ...] = ()
    required_bars: int = 0

    legacy: bool = False
    """One of the original five behind ``confluence_score``.

    ``confluence_score`` is a MODEL-TIER feature, so its meaning must stay pinned to the
    same five conditions no matter how many are added — otherwise every corpus trained
    before the addition describes a different quantity under the same name. New
    conditions are recorded in the payload tier and promoted deliberately, with evidence.
    """

    description: str = ""
