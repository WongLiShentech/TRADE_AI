"""A content digest over the exact rows and values a model was trained on.

Why this exists
---------------
"Trained on strategy 2" names a LIVE QUERY, not a dataset. The rows behind it grow,
get re-graded when new M1 arrives, and get rebuilt when the backtest re-runs. Two
models an hour apart can both claim the same corpus and have seen different data,
and nothing would report it — so "did my change help, or did the ground move?"
becomes unanswerable, which is the only question a retrain is asking.

What it must catch, and why (trade_id, rr_actual) is not enough
---------------------------------------------------------------
The obvious digest — row ids plus outcomes — misses the case that matters most.
Features are frozen into ``trades.signal_reasoning`` when the backtest runs, so a
leakage fix in ``feature_builder`` changes the stored VALUES while the row ids,
the feature NAMES, and every outcome stay identical. One corpus is leaky, the other
is clean, and a digest over ids and labels calls them the same data.

So the digest covers the whole frame: every feature column, ``rr_actual``,
``signal_time`` and ``holding_hours``, in contract order.

Hash ``rr_actual``, not the derived label
-----------------------------------------
``label_threshold_r`` is a RECIPE knob — it already enters the artifact's
``params_hash``. Hashing the derived ``y`` would mint a spurious second dataset over
byte-identical trades the moment the threshold moved. Nothing is lost: any label
change must come either from ``rr_actual`` moving (this digest catches it) or from
the threshold moving (already recorded twice elsewhere).

Bytes, not text — the trap this module exists to avoid
------------------------------------------------------
``str(np.float64)``, ``to_json`` float precision and ``to_csv`` formatting all vary
across pandas and numpy releases. A text-based digest therefore changes when you
upgrade a library, and the change is indistinguishable from "the corpus changed" —
a false alarm that trains you to ignore real ones. Everything here hashes raw
buffers, with three specific hazards handled:

* **NaN payload bits are not stable.** Different producers write different quiet-NaN
  payloads for the same "missing". Positions are hashed as a separate bitmask and
  the slots zeroed, so missingness is captured without depending on the bit pattern.
  This matters here more than most places: XGBoost takes NaN as a first-class value
  and six macro features are routinely NaN on live rows.
* **``-0.0`` and ``+0.0`` compare equal but have different bit patterns**, so they
  are normalised.
* **Datetimes are hashed as int64 nanoseconds**, never as formatted strings, which
  carry timezone and format drift.
"""
from __future__ import annotations

import hashlib
import struct
from typing import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

# Bumping this invalidates every stored fingerprint ON PURPOSE. Without it, a change
# to the hashing scheme is indistinguishable from a change to the data — every stored
# value mismatches at once with no explanation. Change the tag and the mismatch has a
# name.
_SCHEME = b"tradeai/dataset/v1"


def dataset_fingerprint(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Digest ``frame`` over ``columns``, in the order given.

    Args:
        frame: the dataset frame. Row order is significant — a re-simulated window
            that produced the same rows in a different order is a different dataset,
            because the walk-forward folds slice positionally.
        columns: columns to hash, in a fixed contract order. Passing the caller's
            order rather than ``frame.columns`` means a reordered frame cannot
            silently produce a different digest for identical data.

    Returns:
        40-character hex sha1.

    Raises:
        KeyError: if a named column is absent. Deliberately fatal — silently skipping
            a missing column would hash less data under the same scheme tag and
            produce a digest that collides with a genuinely smaller dataset.
    """
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"fingerprint columns absent from frame: {missing}")

    h = hashlib.sha1()
    h.update(_SCHEME)
    h.update(struct.pack("<Q", len(frame)))
    h.update(struct.pack("<Q", len(columns)))

    for col in columns:
        h.update(col.encode("utf-8"))
        h.update(b"\x00")
        series = frame[col]

        if is_datetime64_any_dtype(series):
            # int64 nanoseconds since epoch. NaT becomes INT64_MIN, which is a stable
            # sentinel — unlike NaN, it has one bit pattern.
            h.update(b"i8")
            arr = series.to_numpy(dtype="datetime64[ns]").view("int64")
            h.update(np.ascontiguousarray(arr).tobytes())

        elif is_numeric_dtype(series):
            h.update(b"f8")
            arr = series.to_numpy(dtype="float64", copy=True)
            mask = np.isnan(arr)
            arr[mask] = 0.0          # NaN payload bits are not portable
            arr[arr == 0.0] = 0.0    # -0.0 == 0.0 is True, so this normalises the sign
            h.update(np.ascontiguousarray(arr).tobytes())
            h.update(np.packbits(mask).tobytes())

        else:
            # Categoricals and object columns. Length-prefixed so that ("ab", "c")
            # and ("a", "bc") cannot hash alike.
            h.update(b"s")
            for value in series.tolist():
                raw = b"" if value is None or value is pd.NaT else str(value).encode("utf-8")
                h.update(struct.pack("<I", len(raw)))
                h.update(raw)

    return h.hexdigest()


def feature_keys_hash(keys: Sequence[str]) -> str:
    """Digest over the feature CONTRACT, independent of any data.

    Stored alongside a dataset so a fingerprint mismatch can be diagnosed. Adding a
    feature key changes every dataset fingerprint even though no stored row moved;
    without this, that reads as "all my data changed at once". With it, the report
    can say "the feature contract changed" instead.
    """
    h = hashlib.sha1()
    h.update(b"tradeai/feature-keys/v1")
    h.update(struct.pack("<Q", len(keys)))
    for k in keys:
        h.update(k.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()
