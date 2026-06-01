"""
Session classifier — pure function. Maps a UTC datetime to one of:
  asian | london | ny | overlap (London/NY).

Ranges are sourced from the SecondBrain CLAUDE.md "Session tracking" section:
  Asian:    00:00–08:00 UTC
  London:   07:00–16:00 UTC
  NY:       13:00–22:00 UTC
  Overlap:  12:00–16:00 UTC  (London/NY crossover)

Overlap takes priority when within its window. Outside all windows
(22:00–24:00 UTC) returns "asian" since the late evening NY session bleeds
into the next Asian open and there is no clean separator.
"""
from datetime import datetime


def classify_session(utc_dt: datetime) -> str:
    h = utc_dt.hour
    if 12 <= h < 16:
        return "overlap"
    if 0 <= h < 8:
        return "asian"
    if 7 <= h < 16:
        return "london"
    if 13 <= h < 22:
        return "ny"
    return "asian"
