# ==================================================================
#   modules/priority.py
#   Rolls a ticker's PatternMatch list into a 1-5 star "Priority"
#   rating. Confluence-weighted, not a simple match count:
#     - Agreeing-direction matches compound, with diminishing weight
#       per additional match (2nd match adds less than the 1st, etc).
#     - Conflicting bullish/bearish matches on the same ticker actively
#       subtract from each other — a mixed signal should never
#       outscore a single clean one.
#   Per-match quality (0-1) is derived from whatever signal each
#   detector family already exposes in key_levels.
# ==================================================================
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ------------------------------------------------------------------
# Per-pattern quality extraction — how "clean" a given match is,
# 0.0 (barely qualifies) to 1.0 (textbook example).
# ------------------------------------------------------------------
def _quality_cup_handle(m):
    depth = m.key_levels.get("depth_pct", 15)
    handle_depth = m.key_levels.get("handle_depth_pct", 10)
    # Sweet spot: 15-35% cup depth, shallow handle.
    depth_score = max(0.0, 1 - abs(depth - 25) / 25)
    handle_score = max(0.0, 1 - handle_depth / 15)
    breakout_bonus = 0.15 if m.key_levels.get("breakout") else 0.0
    return min(1.0, 0.5 * depth_score + 0.35 * handle_score + breakout_bonus)


def _quality_bathtub(m):
    depth = m.key_levels.get("depth_pct", 15)
    return max(0.0, min(1.0, 1 - abs(depth - 20) / 30))


def _quality_flag(m):
    pole = abs(m.key_levels.get("pole_move_pct", 12))
    flag_range = m.key_levels.get("flag_range_pct", 6)
    pole_score = min(1.0, pole / 25)
    tightness_score = max(0.0, 1 - flag_range / 6)
    return min(1.0, 0.6 * pole_score + 0.4 * tightness_score)


def _quality_double(m):
    similarity = m.key_levels.get("similarity_pct", 3)
    return max(0.0, min(1.0, 1 - similarity / 3))


def _quality_golden_cross(m):
    # Freshness already gated by lookback_days in the detector; treat
    # any match as reasonably high quality, nudged by volume confirm.
    base = 0.6
    return min(1.0, base + (0.25 if m.volume_confirms else 0))


def _quality_breakout(m):
    mult = m.key_levels.get("volume_mult") or 1.8
    return max(0.0, min(1.0, (mult - 1.8) / 2 + 0.5))


def _quality_overextension(m):
    # Barely over the 100% threshold is a weaker risk signal than a
    # true parabolic 200%+ run — scale quality (and therefore how hard
    # it drags the star rating down) with how extreme the move is.
    gain = m.key_levels.get("gain_pct", 100)
    return max(0.0, min(1.0, 0.4 + (gain - 100) / 200))


QUALITY_FN_BY_PATTERN = {
    "Cup & Handle": _quality_cup_handle,
    "Bathtub": _quality_bathtub,
    "Flag": _quality_flag,
    "Double Top": _quality_double,
    "Double Bottom": _quality_double,
    "Golden Cross": _quality_golden_cross,
    "Death Cross": _quality_golden_cross,
    "Breakout": _quality_breakout,
    "Overextended": _quality_overextension,
}

# Each additional agreeing match still helps, but less than the last.
DIMINISHING_WEIGHTS = [1.0, 0.6, 0.35, 0.2, 0.1]


def _match_quality(m) -> float:
    fn = QUALITY_FN_BY_PATTERN.get(m.pattern)
    q = fn(m) if fn else 0.5
    if m.volume_confirms:
        q = min(1.0, q + 0.1)
    return q


def score_ticker(matches: list) -> float:
    """Returns a raw confluence score (can be negative for a
    conflicted ticker)."""
    if not matches:
        return 0.0

    bullish = sorted([m for m in matches if m.direction == "bullish"], key=_match_quality, reverse=True)
    bearish = sorted([m for m in matches if m.direction == "bearish"], key=_match_quality, reverse=True)

    def weighted_sum(group):
        total = 0.0
        for i, m in enumerate(group):
            weight = DIMINISHING_WEIGHTS[i] if i < len(DIMINISHING_WEIGHTS) else DIMINISHING_WEIGHTS[-1]
            total += _match_quality(m) * weight
        return total

    bull_score = weighted_sum(bullish)
    bear_score = weighted_sum(bearish)
    return bull_score - bear_score


def stars_for_score(score: float) -> int:
    """Maps a raw confluence score to a 1-5 star rating.

    Thresholds calibrated against a real 10,428-ticker run: with the
    DIMINISHING_WEIGHTS curve above, a single strong match tops out
    around 0.9-1.0, and even the best multi-pattern confluence that run
    topped out at 1.52 — so a 1.6 threshold for 5 stars was literally
    unreachable. 5 stars now requires genuine multi-pattern agreement
    (two-plus confirming patterns, not just one very clean one), not an
    impossible bar.
    """
    if score >= 1.3:
        return 5
    if score >= 1.0:
        return 4
    if score >= 0.6:
        return 3
    if score >= 0.2:
        return 2
    return 1


def rate_ticker(matches: list) -> dict:
    score = score_ticker(matches)
    return {"score": round(score, 2), "stars": stars_for_score(score)}


if __name__ == "__main__":
    from modules.pattern_types import PatternMatch

    strong_bull = [
        PatternMatch(pattern="Cup & Handle", direction="bullish", start_date=None, end_date=None,
                     key_levels={"depth_pct": 25, "handle_depth_pct": 5, "breakout": True}, volume_confirms=True),
        PatternMatch(pattern="Golden Cross", direction="bullish", start_date=None, end_date=None, volume_confirms=True),
    ]
    print("Strong bullish confluence:", rate_ticker(strong_bull))

    mixed = strong_bull + [
        PatternMatch(pattern="Double Top", direction="bearish", start_date=None, end_date=None,
                     key_levels={"similarity_pct": 0.5}, volume_confirms=True),
    ]
    print("Mixed signal:", rate_ticker(mixed))

    print("No matches:", rate_ticker([]))
