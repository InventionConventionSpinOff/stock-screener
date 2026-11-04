# ==================================================================
#   modules/recommendation.py
#   Turns a new-stock candidate's pattern confluence (modules/priority.py),
#   news sentiment (modules/news_sentiment.py), and earnings timing into
#   a single BUY / WATCH / AVOID call for the Candidates sheet.
#
#   This is a heuristic, not financial advice — same spirit as
#   modules/positions.py's Buy/Sell/Hold logic for held positions, just
#   applied to stocks you don't own yet.
# ==================================================================
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def recommend_buy(stars: int, sentiment_label: str = None, earnings_soon: bool = False,
                   sector_concentrated: bool = False, weak_backtest_history: bool = False) -> dict:
    """Returns {"verdict": "BUY"|"WATCH"|"AVOID", "reason": str}.

    stars: 1-5 pattern confluence rating from modules.priority.rate_ticker
    sentiment_label: "Bullish"/"Neutral"/"Bearish"/"No news"/None
    earnings_soon: True if earnings fall within SETTINGS["earnings_window_days"]
    sector_concentrated: True if this ticker's sector already makes up
        30%+ of your open positions' dollar value (see
        modules.sector_exposure) — adding more here means doubling down
        on concentration risk, not just picking a good chart pattern.
    weak_backtest_history: True if one of this ticker's matched patterns
        has a measured losing track record (see modules.backtest and
        modules.report_screener._pattern_backtest_is_weak) — e.g. Death
        Cross came back at ~38% win rate / negative average return
        across 300+ historical matches. Checked first, ahead of the
        same-day heuristics below, because it's the one factor here
        backed by actual outcomes rather than a plausible-sounding rule.
    """
    bearish_news = sentiment_label == "Bearish"
    bullish_news = sentiment_label == "Bullish"

    if stars >= 4:
        if weak_backtest_history:
            return {"verdict": "WATCH",
                    "reason": "Strong pattern confluence, but one of the matched patterns has a "
                              "measured losing track record in backtesting — treat with caution"}
        if bearish_news:
            return {"verdict": "WATCH",
                    "reason": "Strong pattern confluence, but news sentiment has turned bearish — wait for it to clear"}
        if earnings_soon:
            return {"verdict": "WATCH",
                    "reason": "Strong pattern confluence, but earnings are coming up soon — expect volatility"}
        if sector_concentrated:
            return {"verdict": "WATCH",
                    "reason": "Strong pattern confluence, but this sector is already 30%+ of your open positions — would increase concentration risk"}
        return {"verdict": "BUY",
                "reason": "Strong multi-pattern confluence with no bearish news conflict"}

    if stars == 3:
        if weak_backtest_history:
            return {"verdict": "AVOID",
                    "reason": "Pattern signal has a measured losing track record in backtesting — skip"}
        if bearish_news:
            return {"verdict": "AVOID",
                    "reason": "Pattern signal conflicts with bearish news sentiment — skip for now"}
        if bullish_news and not earnings_soon and not sector_concentrated:
            return {"verdict": "BUY",
                    "reason": "Solid pattern signal confirmed by positive news sentiment"}
        return {"verdict": "WATCH",
                "reason": "Decent pattern signal, but nothing else confirms it yet"}

    # stars == 2 (the weakest tier that still clears the Candidates bar)
    if weak_backtest_history and bearish_news:
        return {"verdict": "AVOID",
                "reason": "Weak pattern signal, a measured losing pattern track record, and bearish news — skip"}
    if weak_backtest_history:
        return {"verdict": "AVOID",
                "reason": "Weak pattern signal plus a measured losing pattern track record — skip"}
    if bearish_news:
        return {"verdict": "AVOID",
                "reason": "Weak pattern signal plus bearish news — skip"}
    return {"verdict": "WATCH",
            "reason": "Borderline pattern signal — not strong enough on its own to act on"}


if __name__ == "__main__":
    print(recommend_buy(5, "Neutral", earnings_soon=False))
    print(recommend_buy(5, "Bearish", earnings_soon=False))
    print(recommend_buy(5, "Neutral", earnings_soon=False, weak_backtest_history=True))
    print(recommend_buy(3, "Bullish", earnings_soon=False))
    print(recommend_buy(3, "Bearish", earnings_soon=False))
    print(recommend_buy(2, "Neutral", earnings_soon=False))
    print(recommend_buy(2, "Neutral", earnings_soon=False, weak_backtest_history=True))
