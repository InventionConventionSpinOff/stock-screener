# ==================================================================
#   modules/report_screener.py
#   Builds the daily Excel report: one dated file per day
#   (outputs/ScreenerReport_YYYY-MM-DD.xlsx, never overwritten across
#   days — a discovery log you can look back on).
#
#   Sheets: Summary, My Positions, My Options, Candidates, 5/4/3 Star
#   Prospects, Earnings Watch, News Sentiment, Sector Exposure, By
#   Pattern, ZigZag.
# ==================================================================
import os
import sys
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import PATHS, SETTINGS, DIR_OUTPUTS
from modules.priority import rate_ticker, _match_quality
from modules.earnings import in_earnings_window
from modules.swing_points import swing_sequence
from modules.news_sentiment import get_news_sentiment
from modules.recommendation import recommend_buy
from modules.sector_exposure import build_sector_exposure, get_sectors
from modules.backtest import load_pattern_stats_cache, load_pattern_stats_by_regime_cache, pattern_backtest_is_weak
from modules.ml_recommend import predict_win_probability, model_metadata

# The forward-day horizon shown as a single summary figure on the
# Candidates sheet (By Pattern shows every horizon the backtest has).
# Matches modules/backtest.py's DEFAULT_FORWARD_DAYS — 20 trading days
# is roughly a month, long enough for a pattern's thesis to play out
# without the number going stale.
_PRIMARY_HORIZON_DAYS = 20


# The 4 patterns analyze_market_regime.py found a real, held-out-
# reproducing Bull/Bear performance gap for (Bathtub/Cup & Handle
# better in Bear, Breakout/Death Cross worse in Bear) — see that
# script's docstring for the actual numbers. Golden Cross/Flag/
# Overextended showed no reproducible effect or too small a sample, so
# they deliberately get no regime-specific line here.
REGIME_SENSITIVE_PATTERNS = {"Bathtub", "Cup & Handle", "Breakout", "Death Cross"}

# Same sample floor analyze_market_regime.py required per regime-slice
# before trusting a number — below this, the regime-specific stat is
# just noise and showing it would be worse than not showing it.
_MIN_REGIME_SAMPLE = 100


def _format_pattern_edge(pattern: str, stats_df: pd.DataFrame,
                          regime_stats_df: pd.DataFrame = None,
                          current_regime: str = None) -> str:
    """One pattern's historical win-rate/avg-return at the primary
    horizon, as a short string for the Candidates sheet. Falls back to
    an explicit "no data yet" rather than silently omitting the
    pattern, since a missing number and a bad number look identical
    otherwise.

    For the 4 patterns with a validated regime effect
    (REGIME_SENSITIVE_PATTERNS), also appends the stat conditioned on
    TODAY's Bull/Bear regime when there's enough sample to trust it —
    e.g. Death Cross's blended 44.6% win rate hides that it's far
    worse specifically in Bear markets, which is exactly the regime a
    Death Cross match is more likely to show up in."""
    if stats_df.empty or "Pattern" not in stats_df.columns:
        return f"{pattern}: no backtest data yet"
    row = stats_df[stats_df["Pattern"] == pattern]
    win_col = f"WinRate_{_PRIMARY_HORIZON_DAYS}d_pct"
    ret_col = f"AvgReturn_{_PRIMARY_HORIZON_DAYS}d_pct"
    if row.empty or win_col not in row.columns or pd.isna(row.iloc[0][win_col]):
        return f"{pattern}: no backtest data yet"
    r = row.iloc[0]
    sign = "+" if r[ret_col] >= 0 else ""
    edge = f"{pattern}: {r[win_col]:.0f}%/{sign}{r[ret_col]:.1f}% (n={int(r['Sample Size'])})"

    if (pattern in REGIME_SENSITIVE_PATTERNS and current_regime in ("Bull", "Bear")
            and regime_stats_df is not None and not regime_stats_df.empty
            and "Pattern" in regime_stats_df.columns):
        rr = regime_stats_df[(regime_stats_df["Pattern"] == pattern)
                              & (regime_stats_df["Regime"] == current_regime)]
        if (not rr.empty and win_col in rr.columns and not pd.isna(rr.iloc[0][win_col])
                and rr.iloc[0]["Sample Size"] >= _MIN_REGIME_SAMPLE):
            rrow = rr.iloc[0]
            rsign = "+" if rrow[ret_col] >= 0 else ""
            edge += f" [{current_regime}: {rrow[win_col]:.0f}%/{rsign}{rrow[ret_col]:.1f}%]"
    return edge


# _pattern_backtest_is_weak used to live here; it's now
# modules.backtest.pattern_backtest_is_weak so modules/positions.py
# (held positions) can share the exact same gate as the Candidates
# sheet below (new-stock recommendations) without reaching across into
# this report-building module for it.


_SHEET_TAB_COLORS = {
    "Top Picks": "#C00000",
    "Summary": "#1F3864",
    "My Positions": "#BF8F00",
    "My Options": "#BF8F00",
    "Candidates": "#375623",
    "5 Star Prospects": "#375623",
    "4 Star Prospects": "#548235",
    "3 Star Prospects": "#A9D18E",
    "Earnings Watch": "#C55A11",
    "News Sentiment": "#7030A0",
    "Sector Exposure": "#0E7C7B",
    "By Pattern": "#2E75B6",
    "Forming Signals": "#E2A020",
    "Track Record": "#548235",
    "ZigZag": "#2E75B6",
}

# Columns that are dollar amounts — shown with a $ prefix and 2 decimals.
_CURRENCY_COLS = {"AvgCost", "LastPrice", "CurrentMark", "UnrealizedPL", "Strike", "MarketValue", "Entry"}
# Columns already expressed in percent units (e.g. 50.2 meaning "50.2%")
# — a literal "%" suffix, NOT Excel's native percent format (which
# would multiply by 100 again and show "5020%").
_LITERAL_PCT_SUFFIXES = ("pct", "%")
# Columns that are a 0-1 fraction where Excel's native "multiply by
# 100 and add %" percent format is the correct one.
_FRACTION_PCT_COLS = {"MLWinProbability"}
# Long free-text columns worth wrapping instead of truncating.
_WRAP_COLS = {"Notes", "Note", "RecommendationWhy", "Description", "Reason", "HistoricalEdge", "TopHeadline",
              "What it means"}


def _number_format_for_column(col_name: str) -> str:
    if col_name in _CURRENCY_COLS:
        return "$#,##0.00;[RED]-$#,##0.00"
    if col_name in _FRACTION_PCT_COLS:
        return "0.0%"
    lower = col_name.lower()
    if lower.endswith(_LITERAL_PCT_SUFFIXES):
        return '0.0"%"'
    return None


def _write_sheet(writer, df: pd.DataFrame, sheet_name: str, highlight_col: str = None,
                  warn_bool_col: str = None, scale_col: str = None, good_bad_col: str = None):
    if df is None or df.empty:
        pd.DataFrame({"Note": ["No data for this section on this run."]}).to_excel(
            writer, sheet_name=sheet_name, index=False
        )
        writer.sheets[sheet_name].set_tab_color(_SHEET_TAB_COLORS.get(sheet_name, "#808080"))
        return

    df.to_excel(writer, sheet_name=sheet_name, index=False)
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]

    header_fmt = workbook.add_format({
        "bold": True, "font_color": "white", "bg_color": "#1F3864",
        "border": 1,
    })
    for col_idx, col_name in enumerate(df.columns):
        worksheet.write(0, col_idx, col_name, header_fmt)

    n = len(df)
    worksheet.autofilter(0, 0, n, len(df.columns) - 1)
    # Keep the header row AND the leading identifier column (Ticker/
    # Underlying/Metric/Pattern — whatever the sheet is keyed on)
    # visible while scrolling through a wide sheet, rather than just
    # the header row.
    worksheet.freeze_panes(1, 1 if len(df.columns) > 2 else 0)

    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    has_wrapped = False
    for col_idx, col_name in enumerate(df.columns):
        series = df[col_name]
        lengths = series.map(lambda v: len(str(v)) if pd.notna(v) else 0)
        width = max([len(str(col_name))] + lengths.tolist()) + 2

        if col_name in _WRAP_COLS:
            worksheet.set_column(col_idx, col_idx, min(width, 60), wrap_fmt)
            has_wrapped = True
            continue

        number_fmt = _number_format_for_column(col_name)
        col_format = workbook.add_format({"num_format": number_fmt}) if number_fmt else None
        worksheet.set_column(col_idx, col_idx, min(width, 40), col_format)

    if has_wrapped:
        # A fixed, generous row height rather than trying to compute
        # exact wrapped-line counts — cheap insurance against the
        # wrapped text getting clipped, at the cost of some blank
        # space on rows that didn't need it.
        for row_idx in range(1, n + 1):
            worksheet.set_row(row_idx, 30)

    if highlight_col and highlight_col in df.columns:
        col_idx = df.columns.get_loc(highlight_col)
        green = workbook.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
        amber = workbook.add_format({"bg_color": "#FFEB9C", "font_color": "#9C6500"})
        red = workbook.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})
        if df[highlight_col].dtype == object:
            # Text verdicts (e.g. Recommendation: BUY/WATCH/AVOID) rather
            # than a numeric star rating — match by exact text instead of
            # a >=/==/<= threshold.
            worksheet.conditional_format(1, col_idx, n, col_idx, {
                "type": "text", "criteria": "containing", "value": "BUY", "format": green,
            })
            worksheet.conditional_format(1, col_idx, n, col_idx, {
                "type": "text", "criteria": "containing", "value": "WATCH", "format": amber,
            })
            worksheet.conditional_format(1, col_idx, n, col_idx, {
                "type": "text", "criteria": "containing", "value": "AVOID", "format": red,
            })
            # My Positions' "Action" column uses SELL/TRIM/HOLD/BUY MORE/
            # WATCH rather than Candidates' BUY/WATCH/AVOID — same
            # highlight_col mechanism covers both verdict vocabularies.
            worksheet.conditional_format(1, col_idx, n, col_idx, {
                "type": "text", "criteria": "containing", "value": "SELL", "format": red,
            })
            worksheet.conditional_format(1, col_idx, n, col_idx, {
                "type": "text", "criteria": "containing", "value": "TRIM", "format": red,
            })
        else:
            worksheet.conditional_format(1, col_idx, n, col_idx, {
                "type": "cell", "criteria": ">=", "value": 4, "format": green,
            })
            worksheet.conditional_format(1, col_idx, n, col_idx, {
                "type": "cell", "criteria": "==", "value": 3, "format": amber,
            })
            worksheet.conditional_format(1, col_idx, n, col_idx, {
                "type": "cell", "criteria": "<=", "value": 2, "format": red,
            })

    if warn_bool_col and warn_bool_col in df.columns:
        # A plain boolean flag (e.g. WeakBacktestPattern) — only the
        # "True" / cautionary case needs calling out, not both states.
        col_idx = df.columns.get_loc(warn_bool_col)
        red = workbook.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})
        worksheet.conditional_format(1, col_idx, n, col_idx, {
            "type": "cell", "criteria": "==", "value": True, "format": red,
        })

    if good_bad_col and good_bad_col in df.columns:
        # True/False both matter here (unlike warn_bool_col) — a
        # resolved BUY call that actually worked is worth seeing just
        # as clearly as one that didn't.
        col_idx = df.columns.get_loc(good_bad_col)
        green = workbook.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
        red = workbook.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})
        worksheet.conditional_format(1, col_idx, n, col_idx, {
            "type": "cell", "criteria": "==", "value": True, "format": green,
        })
        worksheet.conditional_format(1, col_idx, n, col_idx, {
            "type": "cell", "criteria": "==", "value": False, "format": red,
        })

    if scale_col and scale_col in df.columns:
        # A continuous 0-1 (or similar) score — a 3-color scale reads
        # faster at a glance than discrete thresholds, and doesn't
        # imply false precision about where the "real" cutoff is.
        col_idx = df.columns.get_loc(scale_col)
        worksheet.conditional_format(1, col_idx, n, col_idx, {
            "type": "3_color_scale",
            "min_color": "#F8696B", "mid_color": "#FFEB84", "max_color": "#63BE7B",
        })

    worksheet.set_tab_color(_SHEET_TAB_COLORS.get(sheet_name, "#808080"))


def _ml_model_status_text() -> str:
    meta = model_metadata()
    if meta is None:
        return "not trained"
    status = "active" if meta.get("is_useful") else "not used (AUC too low)"
    return f"{status} — {meta.get('horizon_days')}d, AUC={meta.get('test_auc')}"


def _current_regime_status_text() -> str:
    from modules.market_data import get_current_regime
    regime = get_current_regime()
    if regime == "Unknown":
        return "unknown — SPY not cached yet"
    return f"{regime} (SPY vs 200d MA)"


def _matches_by_ticker(matches: list) -> dict:
    out = {}
    for m in matches:
        out.setdefault(m.ticker, []).append(m)
    return out


def _build_candidates(tickers_df: pd.DataFrame, matches: list) -> pd.DataFrame:
    from modules.market_data import passes_liquidity_filter, passes_activity_filter, get_current_regime

    pattern_stats_df = load_pattern_stats_cache()
    regime_stats_df = load_pattern_stats_by_regime_cache()
    current_regime = get_current_regime()
    by_ticker = _matches_by_ticker(matches)
    rows = []
    excluded_for_quality = 0
    for ticker, group in by_ticker.items():
        rating = rate_ticker(group)
        if rating["stars"] < 2:
            continue  # 1-star tickers are noise, don't surface them
        if not passes_liquidity_filter(ticker) or not passes_activity_filter(ticker):
            excluded_for_quality += 1
            continue  # penny stock, too thinly traded, or trades flat/dead most days
        name = ""
        if tickers_df is not None and "Ticker" in tickers_df.columns:
            match_row = tickers_df[tickers_df["Ticker"] == ticker]
            if not match_row.empty and "Name" in match_row.columns:
                name = match_row.iloc[0]["Name"]
        distinct_patterns = sorted({m.pattern for m in group})
        ml_probs = [predict_win_probability(m.pattern, _match_quality(m), m.volume_confirms) for m in group]
        ml_probs = [p for p in ml_probs if p is not None]
        rows.append({
            "Ticker": ticker,
            "Name": name,
            "Priority": rating["stars"],
            "Score": rating["score"],
            "Patterns": ", ".join(distinct_patterns),
            "Directions": ", ".join(sorted({m.direction for m in group})),
            "Notes": " | ".join(m.note for m in group if m.note),
            "HistoricalEdge": " | ".join(
                _format_pattern_edge(p, pattern_stats_df, regime_stats_df, current_regime)
                for p in distinct_patterns
            ),
            "WeakBacktestPattern": any(pattern_backtest_is_weak(p, pattern_stats_df) for p in distinct_patterns),
            # Average of this ticker's matched patterns' individual ML
            # win-probability predictions. None until the model exists
            # and has cleared its own held-out usefulness bar — see
            # modules/ml_recommend.py.
            "MLWinProbability": round(sum(ml_probs) / len(ml_probs), 3) if ml_probs else None,
        })

    # Explicit columns even when rows is empty — an empty
    # pd.DataFrame([]) has NO columns at all, which breaks any later
    # `.merge(..., on="Ticker")` or column access against this result
    # with a KeyError instead of a harmless no-op.
    candidate_cols = ["Ticker", "Name", "Priority", "Score", "Patterns", "Directions", "Notes",
                      "HistoricalEdge", "WeakBacktestPattern", "MLWinProbability"]
    df = pd.DataFrame(rows, columns=candidate_cols)
    if not df.empty:
        df = df.sort_values(["Priority", "Score"], ascending=[False, False]).reset_index(drop=True)
    df.attrs["excluded_for_quality"] = excluded_for_quality
    return df


def _build_top_picks(candidates_df: pd.DataFrame, forming_matches: list,
                      positions_df: pd.DataFrame, options_report: pd.DataFrame) -> pd.DataFrame:
    """A condensed, single-sheet digest of just the highest-conviction
    items across the whole report — the handful of things actually
    worth reading today, for when 14 sheets is more than you need."""
    rows = []

    if candidates_df is not None and not candidates_df.empty and "Recommendation" in candidates_df.columns:
        top_buys = candidates_df[candidates_df["Recommendation"] == "BUY"].sort_values(
            ["Priority", "Score"], ascending=[False, False]).head(10)
        for _, r in top_buys.iterrows():
            rows.append({"Category": "Top BUY", "Ticker": r["Ticker"],
                         "Detail": f"{r['Priority']}-star — {r.get('RecommendationWhy', '')}"})

    held_tickers = set(positions_df["Ticker"]) if positions_df is not None and not positions_df.empty else set()
    if forming_matches and held_tickers:
        urgent_forming = [m for m in forming_matches
                           if m.ticker in held_tickers and m.key_levels.get("days_to_cross", 99) <= 7]
        for m in sorted(urgent_forming, key=lambda m: m.key_levels.get("days_to_cross", 99)):
            rows.append({"Category": "Urgent (held position)", "Ticker": m.ticker,
                         "Detail": f"{m.pattern} — {m.note}"})

    if options_report is not None and not options_report.empty:
        soon = options_report[options_report["DaysToExpiration"] <= 7]
        for _, r in soon.iterrows():
            rows.append({"Category": "Option expiring soon", "Ticker": r["Underlying"],
                         "Detail": f"{r['Type']} ${r['Strike']} expires in {r['DaysToExpiration']}d — {r['Note']}"})

    if not rows:
        return pd.DataFrame()
    category_order = {"Urgent (held position)": 0, "Option expiring soon": 1, "Top BUY": 2}
    df = pd.DataFrame(rows)
    df["_sort"] = df["Category"].map(category_order).fillna(9)
    return df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)


def _build_forming_signals(forming_matches: list) -> pd.DataFrame:
    """Leading-edge signals from modules/pattern_forming.py — setups
    still in progress, not yet complete. Deliberately kept separate
    from Candidates/By Pattern: these aren't backtested the way the
    completed patterns are, so mixing them in would overstate how much
    validation is actually behind them."""
    if not forming_matches:
        return pd.DataFrame()
    rows = []
    for m in forming_matches:
        rows.append({
            "Ticker": m.ticker,
            "Pattern": m.pattern,
            "Direction": m.direction,
            "Confidence": m.confidence,
            "Note": m.note,
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["Confidence", "Ticker"], ascending=[False, True]).reset_index(drop=True)


def _build_by_pattern(matches: list) -> pd.DataFrame:
    rows = []
    for m in matches:
        ml_prob = predict_win_probability(m.pattern, _match_quality(m), m.volume_confirms)
        rows.append({
            "Ticker": m.ticker,
            "Pattern": m.pattern,
            "Direction": m.direction,
            "Start Date": m.start_date,
            "End Date": m.end_date,
            "Volume Confirms": m.volume_confirms,
            "Note": m.note,
            # None until train_recommendation_model.py has been run AND
            # its own held-out test showed real predictive power (see
            # modules/ml_recommend.py) — never a number that just looks
            # confident without having earned it.
            "MLWinProbability": round(ml_prob, 3) if ml_prob is not None else None,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Pattern", "Ticker"]).reset_index(drop=True)

    # Attach this pattern's own historical win-rate/avg-return per
    # horizon from the backtest cache (modules/backtest.py) — every
    # match of the same pattern shares the same stats, since this is
    # a global per-pattern lookup, not a per-ticker one.
    stats_df = load_pattern_stats_cache()
    if not df.empty and not stats_df.empty:
        stat_cols = [c for c in stats_df.columns if c != "Pattern"]
        df = df.merge(stats_df[["Pattern"] + stat_cols].add_prefix("Hist")
                       .rename(columns={"HistPattern": "Pattern"}),
                       on="Pattern", how="left")
    return df


def _build_zigzag(tickers: list, portfolio_tickers: set, candidate_tickers: set) -> pd.DataFrame:
    from modules.market_data import get_history

    focus = [t for t in tickers if t in portfolio_tickers or t in candidate_tickers]
    rows = []
    for t in focus:
        hist = get_history(t)
        if hist.empty:
            continue
        legs = swing_sequence(hist, order=SETTINGS["swing_order"], max_legs=SETTINGS["zigzag_legs_per_ticker"])
        for _, leg in legs.iterrows():
            rows.append({"Ticker": t, "Date": leg["Date"], "Type": leg["Type"], "Price": round(leg["Price"], 2)})
    return pd.DataFrame(rows)


def build_report(tickers_df: pd.DataFrame, matches: list, earnings_df: pd.DataFrame,
                  run_stats: dict, custom_tickers: list = None, positions_df: pd.DataFrame = None,
                  options_df: pd.DataFrame = None, forming_matches: list = None) -> "Path":
    today = dt.datetime.now().strftime("%Y-%m-%d")
    out_path = DIR_OUTPUTS / f"ScreenerReport_{today}.xlsx"

    candidates_df = _build_candidates(tickers_df, matches)
    by_pattern_df = _build_by_pattern(matches)
    forming_df = _build_forming_signals(forming_matches)

    portfolio_tickers = set(positions_df["Ticker"]) if positions_df is not None and not positions_df.empty else set()
    candidate_tickers = set(candidates_df["Ticker"]) if not candidates_df.empty else set()

    # News sentiment is only fetched for this small, relevant set (held
    # positions + surfaced candidates) — not the whole scanned universe —
    # since yfinance's news feed isn't built for scanning thousands of
    # tickers a day. See modules/news_sentiment.py.
    sentiment_df = get_news_sentiment(sorted(portfolio_tickers | candidate_tickers))
    if not sentiment_df.empty:
        candidates_df = candidates_df.merge(
            sentiment_df[["Ticker", "SentimentLabel", "AvgSentiment"]], on="Ticker", how="left"
        )

    earnings_view = pd.DataFrame()
    earnings_soon_tickers = set()
    if earnings_df is not None and not earnings_df.empty:
        ev = earnings_df.copy()
        ev["InWindow"] = ev["NextEarningsDate"].apply(in_earnings_window)
        earnings_view = ev[ev["InWindow"]].drop(columns=["InWindow"]).sort_values("NextEarningsDate")
        earnings_soon_tickers = set(ev[ev["InWindow"]]["Ticker"])

    # Sector concentration across your OPEN positions — e.g. if 60% of
    # invested dollars already sit in Technology, a fresh "BUY" call on
    # another tech name is a different decision than it looks in
    # isolation. See modules/sector_exposure.py.
    sector_exposure_df = build_sector_exposure(positions_df, options_df)
    concentrated_sectors = set(
        sector_exposure_df[sector_exposure_df["Concentrated"]]["Sector"]
    ) if not sector_exposure_df.empty else set()
    candidate_sector_map = get_sectors(sorted(candidate_tickers)) if candidate_tickers else {}

    # Buy/Watch/Avoid call for new-stock candidates — combines pattern
    # confluence (Priority), news sentiment, earnings timing, and
    # whether this ticker's sector is already a concentrated position.
    # See modules/recommendation.py. Distinct from the My Positions
    # sheet's Buy More/Hold/Trim/Sell logic, which only applies to
    # stocks you already own.
    if not candidates_df.empty:
        if "SentimentLabel" not in candidates_df.columns:
            candidates_df["SentimentLabel"] = None
        recs = candidates_df.apply(
            lambda row: recommend_buy(
                row["Priority"], row.get("SentimentLabel"), row["Ticker"] in earnings_soon_tickers,
                candidate_sector_map.get(row["Ticker"]) in concentrated_sectors,
                row.get("WeakBacktestPattern", False),
            ),
            axis=1,
        )
        candidates_df["Recommendation"] = recs.apply(lambda r: r["verdict"])
        candidates_df["RecommendationWhy"] = recs.apply(lambda r: r["reason"])

    # Log today's BUY/WATCH calls and resolve any older ones that have
    # now had enough trading days pass — the live (not backtested)
    # track record. See modules/tracking.py.
    from modules.tracking import log_todays_calls, resolve_due_calls, track_record_summary
    log_todays_calls(candidates_df)
    track_log_df = resolve_due_calls()
    track_summary_df = track_record_summary(track_log_df)

    positions_report = pd.DataFrame()
    if positions_df is not None and not positions_df.empty:
        from modules.positions import build_positions_report
        positions_report = build_positions_report(positions_df, _matches_by_ticker(matches),
                                                    load_pattern_stats_cache())
        if not sentiment_df.empty:
            positions_report = positions_report.merge(
                sentiment_df[["Ticker", "SentimentLabel", "AvgSentiment"]], on="Ticker", how="left"
            )

    options_report = pd.DataFrame()
    if options_df is not None and not options_df.empty:
        from modules.options import build_options_report
        options_report = build_options_report(options_df, _matches_by_ticker(matches), log=print)

    zigzag_df = _build_zigzag(tickers_df["Ticker"].tolist() if tickers_df is not None else [],
                               portfolio_tickers, candidate_tickers)

    top_picks_df = _build_top_picks(candidates_df, forming_matches, positions_df, options_report)

    summary_rows = [
        {"Metric": "Run date", "Value": dt.datetime.now().strftime("%A %B %d, %Y  %H:%M")},
        {"Metric": "Tickers scanned", "Value": run_stats.get("tickers_scanned")},
        {"Metric": "Elapsed (sec)", "Value": run_stats.get("elapsed_sec")},
        {"Metric": "Candidates", "Value": len(candidates_df)},
        {"Metric": "5-star candidates", "Value": len(candidates_df[candidates_df["Priority"] == 5]) if not candidates_df.empty else 0},
        {"Metric": "Excluded (low quality)", "Value": candidates_df.attrs.get("excluded_for_quality", 0)},
        {"Metric": "BUY calls", "Value": len(candidates_df[candidates_df["Recommendation"] == "BUY"]) if not candidates_df.empty else 0},
        {"Metric": "Positions flagged SELL/TRIM", "Value": len(positions_report[positions_report["Action"].isin(["SELL", "TRIM"])]) if not positions_report.empty else 0},
        {"Metric": "Forming signals", "Value": len(forming_df) if not forming_df.empty else 0},
        {"Metric": "Open option positions", "Value": len(options_report) if not options_report.empty else 0},
        {"Metric": "Options expiring soon", "Value": len(options_report[options_report["DaysToExpiration"] <= 7]) if not options_report.empty else 0},
        {"Metric": "Earnings soon", "Value": len(earnings_view)},
        {"Metric": "Bullish / Bearish sentiment", "Value": (
            f"{len(sentiment_df[sentiment_df['SentimentLabel'] == 'Bullish'])} / "
            f"{len(sentiment_df[sentiment_df['SentimentLabel'] == 'Bearish'])}"
        ) if not sentiment_df.empty else "0 / 0"},
        {"Metric": "Largest sector exposure", "Value": (
            f"{sector_exposure_df.iloc[0]['Sector']} ({sector_exposure_df.iloc[0]['PortfolioPct']}%)"
            if not sector_exposure_df.empty else "(none)"
        )},
        {"Metric": "Custom tickers", "Value": ", ".join(custom_tickers) if custom_tickers else "(none)"},
        {"Metric": "ML model", "Value": _ml_model_status_text()},
        {"Metric": "Market regime", "Value": _current_regime_status_text()},
        {"Metric": "Live track record", "Value": (
            f"{int(track_summary_df.iloc[1]['Value'])} resolved / {int(track_summary_df.iloc[2]['Value'])} pending"
            if not track_summary_df.empty else "no calls logged yet"
        )},
    ]
    summary_df = pd.DataFrame(summary_rows)

    def _write_workbook(path):
        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            _write_sheet(writer, top_picks_df, "Top Picks")
            _write_sheet(writer, summary_df, "Summary")
            _write_sheet(writer, positions_report, "My Positions", highlight_col="Action",
                         warn_bool_col="WeakBacktestPattern")
            _write_sheet(writer, options_report, "My Options")
            _write_sheet(writer, candidates_df, "Candidates", highlight_col="Recommendation",
                         warn_bool_col="WeakBacktestPattern", scale_col="MLWinProbability")
            for stars in (5, 4, 3):
                tier_df = candidates_df[candidates_df["Priority"] == stars] if not candidates_df.empty else pd.DataFrame()
                _write_sheet(writer, tier_df, f"{stars} Star Prospects",
                             warn_bool_col="WeakBacktestPattern", scale_col="MLWinProbability")
            _write_sheet(writer, earnings_view, "Earnings Watch")
            _write_sheet(writer, sentiment_df.sort_values("AvgSentiment", ascending=False) if not sentiment_df.empty else sentiment_df,
                         "News Sentiment")
            _write_sheet(writer, sector_exposure_df, "Sector Exposure")
            _write_sheet(writer, by_pattern_df, "By Pattern", scale_col="MLWinProbability")
            _write_sheet(writer, forming_df, "Forming Signals", scale_col="Confidence")
            _write_sheet(writer, track_log_df, "Track Record", good_bad_col="Correct")
            _write_sheet(writer, zigzag_df, "ZigZag")

            # First sheet Excel opens to (writing order above already does
            # this, but set_first_sheet pins it even if a user has
            # previously reordered tabs in a file with this name).
            writer.sheets["Top Picks"].set_first_sheet()

    try:
        _write_workbook(out_path)
    except PermissionError:
        # Same-day rerun while the earlier report is still open in Excel
        # locks the dated filename — fall back to a timestamped one
        # rather than losing the whole run's results.
        out_path = DIR_OUTPUTS / f"ScreenerReport_{today}_{dt.datetime.now().strftime('%H%M%S')}.xlsx"
        _write_workbook(out_path)

    highlights = {
        "buy_calls": len(candidates_df[candidates_df["Recommendation"] == "BUY"]) if not candidates_df.empty else 0,
        "sell_trim": len(positions_report[positions_report["Action"].isin(["SELL", "TRIM"])]) if not positions_report.empty else 0,
        "candidates": len(candidates_df),
        "earnings_soon": len(earnings_view),
    }
    return out_path, highlights


if __name__ == "__main__":
    print("Run run_screener.py for a real report — this module has no standalone self-test "
          "(it needs live matches/earnings data to produce something meaningful).")
