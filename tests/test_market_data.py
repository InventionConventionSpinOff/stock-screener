# ==================================================================
#   tests/test_market_data.py
#   _detect_split_discontinuity should catch the specific signature of
#   a stock split landing between an existing cache and a fresh
#   incremental fetch: the SAME calendar date disagreeing on Close
#   between the two fetches, because Yahoo retroactively re-scales a
#   ticker's whole history once a new split happens.
# ==================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from modules.market_data import _detect_split_discontinuity


def _df(dates, closes):
    return pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Open": closes, "High": closes, "Low": closes, "Close": closes,
        "Volume": [1_000_000.0] * len(closes),
    })


def test_no_discontinuity_when_overlapping_closes_agree():
    existing = _df(["2026-01-01", "2026-01-02"], [100.0, 101.0])
    fresh = _df(["2026-01-02", "2026-01-03"], [101.0, 102.0])
    assert _detect_split_discontinuity(existing, fresh) is False


def test_discontinuity_detected_on_a_2_for_1_split():
    existing = _df(["2026-01-01", "2026-01-02"], [100.0, 101.0])
    # Fresh fetch now sees the whole history post-split — the SAME
    # 2026-01-02 close comes back roughly halved.
    fresh = _df(["2026-01-02", "2026-01-03"], [50.5, 51.0])
    assert _detect_split_discontinuity(existing, fresh) is True


def test_small_disagreement_is_not_a_split():
    # A trivial rounding/vendor discrepancy (<5%) must not false-
    # positive into a needless full re-backfill.
    existing = _df(["2026-01-01", "2026-01-02"], [100.0, 101.0])
    fresh = _df(["2026-01-02", "2026-01-03"], [101.5, 102.0])
    assert _detect_split_discontinuity(existing, fresh) is False


def test_no_overlap_means_no_discontinuity():
    existing = _df(["2026-01-01"], [100.0])
    fresh = _df(["2026-02-01"], [999.0])
    assert _detect_split_discontinuity(existing, fresh) is False


def test_empty_frames_are_safe():
    existing = _df(["2026-01-01"], [100.0])
    empty = pd.DataFrame()
    assert _detect_split_discontinuity(empty, existing) is False
    assert _detect_split_discontinuity(existing, empty) is False
    assert _detect_split_discontinuity(empty, empty) is False
