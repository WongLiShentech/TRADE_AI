"""S1 — ML training & evaluation pipeline for the rule-engine trade corpus.

This package trains a single XGBoost classifier on the M7 backtest corpus
(``trades.stage == 'backtest'``) and evaluates an ML-FILTERED version of the
rule strategy on the SAME three-fold expanding walk-forward folds M7 used, so the
result is directly comparable to the rule baseline against the same promotion gate.

Design invariants (leakage firewall — see each module's docstring):

* Features come ONLY from ``feature_builder.FEATURE_KEYS_MODEL`` (the 20 locked
  model keys) extracted from each trade's ``signal_reasoning`` JSON — never a
  re-computation, never a hardcoded key list.
* The label is derived at training time: ``y = 1 if rr_actual >= ML_LABEL_THRESHOLD_R``.
* The one-hot encoder and the model are fit on the TRAINING SLICE ONLY within each
  fold (LEAK-6: nothing is ever fit on OOS data).
* The filter threshold is chosen PER FOLD on that fold's in-sample validation tail
  only — never on OOS.
* Numeric NaNs pass through to XGBoost natively (no imputation, no scaling).
"""
