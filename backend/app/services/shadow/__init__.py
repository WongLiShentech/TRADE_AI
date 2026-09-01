"""Shadow observation — M8-Shadow Phase 2.

Shadow mode is the platform's *observe-only* deployment of the S1 ML signal
filter. Every time the LIVE rule engine fires a signal, the shadow path:

1. recomputes the LOCKED feature contract through the SAME chokepoint the M7
   training corpus used (``feature_builder.build_features``),
2. scores it with the serialized artifact via ``ml.inference.score_and_decide``,
3. persists ONE ``trades`` row with ``stage='shadow'`` carrying the full feature
   dict, the ML probability/decision/model-id, and the live RiskEngine verdict.

It places no orders, sizes no positions and alters no fire/no-fire logic — it is
purely additive to the live pipeline.

Phase 3 (:mod:`app.services.shadow.resolver`) closes the loop: a scheduled job walks
the pending queue (``stage='shadow' AND closed_at IS NULL``) and fills the outcome
columns Phase 2 leaves NULL, using the SAME triple-barrier simulator and the SAME
``label_outcome`` convention as the M7 backtest corpus. Both ``'take'`` and
``'skip'`` rows are resolved — the skips are the counterfactual that makes the
filter's value measurable.

Why BOTH decisions are recorded
-------------------------------
A ``'skip'`` row is exactly as valuable as a ``'take'`` row: the only way to
measure whether the filter helps is to compare how the trades it DECLINED fared
against the ones it accepted. Recording takes only would make the shadow corpus
survivorship-biased and the whole experiment unfalsifiable.
"""
from app.services.shadow.recorder import (
    NAN_MODEL_FEATURE_KEYS_KEY,
    NAN_MODEL_FEATURES_KEY,
    REASONING_SHADOW_KEY,
    STAGE_SHADOW,
    RiskAssessment,
    live_signal_time,
    record_shadow_decision,
)
from app.services.shadow.resolver import (
    min_bars_per_bucket,
    observable_bars,
    resolve_pending,
)

__all__ = [
    "NAN_MODEL_FEATURES_KEY",
    "NAN_MODEL_FEATURE_KEYS_KEY",
    "REASONING_SHADOW_KEY",
    "STAGE_SHADOW",
    "RiskAssessment",
    "live_signal_time",
    "min_bars_per_bucket",
    "observable_bars",
    "record_shadow_decision",
    "resolve_pending",
]
