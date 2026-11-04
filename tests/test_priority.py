# ==================================================================
#   tests/test_priority.py
#   modules.priority's confluence scoring — the star rating behind
#   both the Candidates sheet and (via WeakBacktestPattern's sibling
#   logic) every downstream recommendation.
# ==================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.pattern_types import PatternMatch
from modules.priority import rate_ticker, score_ticker, stars_for_score


def _match(pattern, direction, volume_confirms=True, **key_levels):
    return PatternMatch(pattern=pattern, direction=direction, start_date=None, end_date=None,
                         key_levels=key_levels, volume_confirms=volume_confirms)


def test_no_matches_is_zero_score_one_star():
    r = rate_ticker([])
    assert r["score"] == 0.0
    assert r["stars"] == 1


def test_single_clean_match_scores_positively():
    matches = [_match("Golden Cross", "bullish")]
    r = rate_ticker(matches)
    assert r["score"] > 0


def test_conflicting_signals_score_lower_than_either_alone():
    bullish_only = [_match("Golden Cross", "bullish")]
    bearish_only = [_match("Death Cross", "bearish")]
    mixed = bullish_only + bearish_only

    bull_score = score_ticker(bullish_only)
    mixed_score = score_ticker(mixed)
    # A conflicting bearish match should pull the mixed score down
    # below the bullish-only score, not add to it.
    assert mixed_score < bull_score


def test_second_agreeing_match_adds_less_than_the_first():
    one = score_ticker([_match("Golden Cross", "bullish")])
    two = score_ticker([_match("Golden Cross", "bullish"), _match("Flag", "bullish", pole_move_pct=15)])
    # Diminishing returns: the second match's marginal contribution
    # should be smaller than the first match's own score.
    assert two > one
    assert (two - one) < one


def test_stars_for_score_thresholds_are_monotonic():
    scores = [-1.0, 0.0, 0.3, 0.7, 1.1, 1.5]
    stars = [stars_for_score(s) for s in scores]
    assert stars == sorted(stars)  # non-decreasing as score increases
    assert stars[0] == 1
    assert stars[-1] == 5


def test_volume_confirmation_increases_quality():
    confirmed = score_ticker([_match("Golden Cross", "bullish", volume_confirms=True)])
    unconfirmed = score_ticker([_match("Golden Cross", "bullish", volume_confirms=False)])
    assert confirmed > unconfirmed
