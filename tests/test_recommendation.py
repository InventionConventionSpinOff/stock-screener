# ==================================================================
#   tests/test_recommendation.py
#   modules.recommendation.recommend_buy's full verdict matrix. This
#   is the function deciding real BUY/WATCH/AVOID calls — a silent
#   regression here (e.g. a reordered condition flipping a verdict)
#   is exactly the kind of bug that's invisible until you notice a bad
#   recommendation weeks later.
# ==================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.recommendation import recommend_buy


def test_strong_confluence_clean_is_buy():
    r = recommend_buy(5, "Neutral", earnings_soon=False, sector_concentrated=False)
    assert r["verdict"] == "BUY"


def test_strong_confluence_downgraded_by_weak_backtest_history():
    # weak_backtest_history should win out over an otherwise-clean 5-star setup
    r = recommend_buy(5, "Neutral", earnings_soon=False, sector_concentrated=False,
                       weak_backtest_history=True)
    assert r["verdict"] == "WATCH"


def test_strong_confluence_downgraded_by_bearish_news():
    r = recommend_buy(5, "Bearish", earnings_soon=False)
    assert r["verdict"] == "WATCH"


def test_strong_confluence_downgraded_by_earnings_soon():
    r = recommend_buy(5, "Neutral", earnings_soon=True)
    assert r["verdict"] == "WATCH"


def test_strong_confluence_downgraded_by_sector_concentration():
    r = recommend_buy(4, "Neutral", earnings_soon=False, sector_concentrated=True)
    assert r["verdict"] == "WATCH"


def test_moderate_confluence_with_bullish_news_is_buy():
    r = recommend_buy(3, "Bullish", earnings_soon=False, sector_concentrated=False)
    assert r["verdict"] == "BUY"


def test_moderate_confluence_weak_backtest_is_avoid():
    r = recommend_buy(3, "Neutral", weak_backtest_history=True)
    assert r["verdict"] == "AVOID"


def test_moderate_confluence_bearish_news_is_avoid():
    r = recommend_buy(3, "Bearish")
    assert r["verdict"] == "AVOID"


def test_moderate_confluence_no_confirmation_is_watch():
    r = recommend_buy(3, "Neutral")
    assert r["verdict"] == "WATCH"


def test_weak_confluence_plain_is_watch():
    r = recommend_buy(2, "Neutral")
    assert r["verdict"] == "WATCH"


def test_weak_confluence_bearish_news_is_avoid():
    r = recommend_buy(2, "Bearish")
    assert r["verdict"] == "AVOID"


def test_weak_confluence_weak_backtest_is_avoid():
    r = recommend_buy(2, "Neutral", weak_backtest_history=True)
    assert r["verdict"] == "AVOID"


def test_every_verdict_has_a_nonempty_reason():
    for stars in (2, 3, 4, 5):
        for sentiment in (None, "Neutral", "Bullish", "Bearish"):
            r = recommend_buy(stars, sentiment)
            assert r["verdict"] in ("BUY", "WATCH", "AVOID")
            assert isinstance(r["reason"], str) and len(r["reason"]) > 0
