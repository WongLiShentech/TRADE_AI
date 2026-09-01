"""Dependency-free constants shared between the S1 TRAINING and SERVING paths.

Why this module exists (RAM, not aesthetics)
--------------------------------------------
``ml/inference.py`` — which runs inside the always-on API/scheduler process —
needed exactly one string from ``ml/artifact.py``: the ``NOT_PROMOTED`` filename
marker. Importing it dragged ``artifact`` → ``shap_analysis`` → ``shap`` into the
serving process, for a measured **+96 MB RSS** that never executes a single line
of SHAP code. On a small always-on VM that is a meaningful fraction of the box.

So the constant lives here instead. This module must import NOTHING beyond the
standard library — that invariant is the whole point of the file. Anything the
serving path needs from the training path belongs here, not in ``artifact``.
"""
from __future__ import annotations

# Filename marker stamped onto every artifact whose walk-forward gate verdict is
# DO-NOT-PROMOTE. A non-promoted candidate is still serialized (its model card +
# SHAP are the post-mortem), but the marker + the metadata flag make it impossible
# for any serving/loader path to mistake it for a deployable model.
NOT_PROMOTED_MARKER = "NOT_PROMOTED"
