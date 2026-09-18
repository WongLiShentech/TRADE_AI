"""The built-in signal conditions.

Each is a pure function of a :class:`ConditionContext`. Adding one here plus naming it in
``SIGNAL_CONDITIONS`` is the whole of "add a condition" — no engine edit, and no second
place to keep in sync.

A note on roles as declared here
--------------------------------
``session`` and ``spread`` are declared ``GATE`` because that is what they ARE: neither
looks at direction, so both answer identically for BUY and SELL. The engine does not yet
act on that — it still scores every enabled condition, preserving current behaviour — and
separating gates from votes is a deliberate, separately-reviewable change. The
declaration is the truth; the engine catching up to it is the next step.

That gap is the whole defect. Scored as votes, the two contribute the same 2 points to
both sides, so at a threshold of 2 every bar satisfies BUY and SELL at once and is
discarded as ambiguous — measured: a 2-of-5 backtest fired only on bars where session or
spread FAILED, selecting for worse execution (spread OK on 53.9% of its trades vs 80.3%).
"""
from __future__ import annotations

from typing import Optional

from app.services.session_classifier import classify_session
from app.domain.conditions.spec import Condition, ConditionContext, Role

_BUY = "BUY"


def _trend(ctx: ConditionContext, direction: Optional[str]) -> bool:
    """Is the higher-timeframe trend aligned with this direction?

    Daily close versus its own SMA. A blunt instrument: measured across the backtest
    corpus this fires on 93% of trades (100% live), so it separates almost nothing — it
    reports the SIGN of the trend and says nothing about its STRENGTH. That is the gap
    an ADX/±DI condition is meant to fill.
    """
    if ctx.trend_close is None or ctx.trend_sma is None:
        return False
    return ctx.trend_close > ctx.trend_sma if direction == _BUY else ctx.trend_close < ctx.trend_sma


def _rsi(ctx: ConditionContext, direction: Optional[str]) -> bool:
    """Is momentum inside this direction's configured band?

    The two bands are independently configurable and, as configured, OVERLAP (BUY 40-55,
    SELL 45-60). Inside the overlap this condition is effectively direction-blind and
    feeds the same ambiguity problem as the gates — see
    tests/test_rule_based_engine.py::test_an_rsi_in_the_band_overlap_satisfies_both_directions.
    """
    rsi = getattr(ctx.indicators, "rsi14", None)
    if rsi is None:
        return False
    s = ctx.settings
    if direction == _BUY:
        return s.SIGNAL_RSI_OVERSOLD <= rsi <= s.SIGNAL_RSI_OVERBOUGHT
    return s.SIGNAL_RSI_OVERSOLD_SELL <= rsi <= s.SIGNAL_RSI_OVERBOUGHT_SELL


def _structure(ctx: ConditionContext, direction: Optional[str]) -> bool:
    """Is price at the edge of its recent range, on the side this direction wants?

    Reads the TRAILING Donchian channel, which is knowable at its own bar close. It
    previously read ``swing_high``/``swing_low``, which come from a CENTRED window and
    are therefore decided by bars AFTER the one they sit on: a look-ahead in backtest
    (where indicators are recomputed over full history) and unpopulated in live. Never
    reintroduce a centred-window read here without routing it through a causal accessor.

    A NULL channel is False, not True: between a migration and its backfill the column is
    empty, and an unevaluable condition must never read as a satisfied one.
    """
    atr = getattr(ctx.indicators, "atr14", None)
    if atr is None:
        return False
    buffer = ctx.settings.SIGNAL_STRUCTURE_ATR_BUFFER * atr
    close = ctx.latest_bar.close
    if direction == _BUY:
        low = getattr(ctx.indicators, "donchian_low", None)
        return low is not None and (close - low) <= buffer
    high = getattr(ctx.indicators, "donchian_high", None)
    return high is not None and (high - close) <= buffer


def _session(ctx: ConditionContext, direction: Optional[str]) -> bool:
    """Is the decision time inside a tradeable session? Direction-independent."""
    allowed = {
        s.strip().lower()
        for s in ctx.settings.SIGNAL_SESSION_FILTER.split(",")
        if s.strip()
    }
    return classify_session(ctx.now_utc) in allowed


def _spread(ctx: ConditionContext, direction: Optional[str]) -> bool:
    """Is the quoted spread inside the cap? Direction-independent.

    Uses the QUOTE, not the candle: in a backtest that is the decision bar's Bid/Ask
    close, live it is the tick cache. A pip_size of 0 reads as an infinite spread and
    therefore False, rather than dividing by zero.
    """
    pip = ctx.instrument.pip_size
    if not pip:
        return False
    return ((ctx.quote_ask - ctx.quote_bid) / pip) < ctx.settings.SIGNAL_MAX_SPREAD_PIPS


TREND = Condition(
    key="trend", role=Role.VOTE, evaluate=_trend, payload_key="c1_trend", legacy=True,
    settings_keys=("SIGNAL_TREND_SMA_PERIOD", "SIGNAL_TREND_TIMEFRAME"),
    description="Higher-timeframe close vs its SMA, aligned with the direction.",
)

RSI = Condition(
    key="rsi", role=Role.VOTE, evaluate=_rsi, payload_key="c2_rsi", legacy=True,
    settings_keys=(
        "SIGNAL_RSI_OVERSOLD", "SIGNAL_RSI_OVERBOUGHT",
        "SIGNAL_RSI_OVERSOLD_SELL", "SIGNAL_RSI_OVERBOUGHT_SELL",
    ),
    required_indicators=("rsi14",),
    description="RSI inside the direction's configured band.",
)

STRUCTURE = Condition(
    key="structure", role=Role.VOTE, evaluate=_structure, payload_key="c3_structure", legacy=True,
    settings_keys=("SIGNAL_STRUCTURE_ATR_BUFFER", "SIGNAL_DONCHIAN_PERIOD"),
    required_indicators=("atr14",),
    description="Close within N x ATR of the trailing Donchian extreme on this side.",
)

SESSION = Condition(
    key="session", role=Role.GATE, evaluate=_session, payload_key="c4_session", legacy=True,
    settings_keys=("SIGNAL_SESSION_FILTER",),
    description="Decision time falls in a tradeable session. Direction-independent.",
)

SPREAD = Condition(
    key="spread", role=Role.GATE, evaluate=_spread, payload_key="c5_spread", legacy=True,
    settings_keys=("SIGNAL_MAX_SPREAD_PIPS",),
    description="Quoted spread within the cap. Direction-independent.",
)

BUILTIN: tuple[Condition, ...] = (TREND, RSI, STRUCTURE, SESSION, SPREAD)
