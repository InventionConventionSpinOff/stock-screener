# ==================================================================
#   Stock Screener  —  run_screener.py
#   Daily chart-pattern & earnings discovery scan, plus a Buy/Sell/Hold
#   read on your own open Fidelity positions.
#
#   Usage:
#       python run_screener.py                      <- portfolio tickers only
#       python run_screener.py --universe full       <- portfolio + DJIA + S&P 500 + Nasdaq-100
#       python run_screener.py --universe all-us     <- EVERY US-listed ticker (~10,000+, SEC list)
#       python run_screener.py --universe fast       <- only tickers already known liquid from a
#                                                        past all-us run (much faster repeat scan)
#       python run_screener.py --universe portfolio  <- only your traded tickers (fast)
#       python run_screener.py --limit 10            <- cap ticker count (smoke testing)
#       python run_screener.py --add NVDA,PLTR        <- also screen extra tickers this run
#       python run_screener.py --refresh-universe    <- force re-scrape index constituent lists
#       python run_screener.py --check                <- config check only
#
#   You'll also be prompted interactively for extra one-off tickers
#   (skipped automatically when run headlessly, e.g. by the watcher).
#
#   This tool NEVER writes to your Fidelity account or any brokerage —
#   it only reads your exported Positions CSV and writes its own
#   output file(s)/logs under screener\.
#
#   Logs:
#       Console                            — live output while running
#       logs\ScreenerRun_YYYY-MM-DD.log     — saved after every run
# ==================================================================

import sys
import time
import traceback
import datetime as dt
import logging
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from config import SETTINGS, DIR_LOGS
from modules.run_lock import acquire_lock, release_lock


# ==================================================================
# LOGGING SETUP
# ==================================================================
def setup_logging():
    today = dt.datetime.now().strftime("%Y-%m-%d")
    log_file = DIR_LOGS / f"ScreenerRun_{today}.log"

    logger = logging.getLogger("STOCK_SCREENER")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    return logger, log_file


def prompt_for_custom_tickers(log) -> list:
    """Interactive-only: ask before the scan starts whether there are
    extra tickers to check today, on top of the regular universe and
    any --add tickers already passed on the command line. Skipped
    automatically when stdin isn't a real console (e.g. the scheduled
    watcher invoking this script headlessly)."""
    if not sys.stdin.isatty():
        return []
    try:
        raw = input("Enter extra tickers to check today (comma-separated), or press Enter to skip: ")
    except (EOFError, KeyboardInterrupt):
        return []
    tickers = _valid_tickers(raw)
    if tickers:
        info(log, f"Custom tickers requested: {', '.join(tickers)}")
    return tickers


def _valid_tickers(raw: str) -> list:
    """Splits a comma-separated string into valid-looking ticker
    symbols, silently dropping anything that isn't ticker-shaped
    (e.g. if a whole sentence gets pasted in by mistake)."""
    import re
    candidates = {t.strip().upper() for t in raw.split(",") if t.strip()}
    return sorted(t for t in candidates if re.match(r"^[A-Z][A-Z0-9.\-]{0,6}$", t))


def banner(log, text): log.info("=" * 60); log.info(f"  {text}"); log.info("=" * 60)
def section(log, text): log.info(""); log.info(f"── {text} {'─' * max(0, 52 - len(text))}")
def ok(log, text):   log.info(f"  ✓  {text}")
def warn(log, text): log.warning(f"  ⚠  {text}")
def err(log, text):  log.error(f"  ✗  {text}")
def info(log, text): log.info(f"     {text}")


# ==================================================================
# STEP 1 — Config check
# ==================================================================
def step_check_config(log) -> bool:
    section(log, "STEP 1 — Config & file check")
    from modules.ticker_universe import resolve_portfolio_csv_path

    csv_path = resolve_portfolio_csv_path()
    if not csv_path.exists():
        warn(log, f"No broker positions CSV found at: {csv_path}")
        warn(log, "Continuing without it — portfolio/Buy-Sell-Hold analysis will be skipped, "
                   "but new-stock scanning still works.")
        return True
    ok(log, f"Broker positions CSV found: {csv_path.name}")
    return True


# ==================================================================
# STEP 2 — Build ticker universe
# ==================================================================
def step_build_universe(log, universe_mode, limit, force_refresh, custom_tickers=None):
    section(log, "STEP 2 — Build ticker universe")
    from modules.ticker_universe import (
        build_full_universe, build_all_us_universe, build_cached_liquid_universe,
        get_traded_tickers, attach_names,
    )
    import pandas as pd

    if universe_mode == "portfolio":
        try:
            tickers = get_traded_tickers()
        except (FileNotFoundError, ValueError) as e:
            warn(log, f"Could not read portfolio tickers ({e}) — using an empty portfolio list.")
            tickers = []
        df = pd.DataFrame({
            "Ticker": tickers, "Source": "Portfolio", "Index": "",
        })
    elif universe_mode == "all-us":
        df = build_all_us_universe(force_refresh=force_refresh, log=lambda m: warn(log, m))
    elif universe_mode == "fast":
        df = build_cached_liquid_universe(log=lambda m: warn(log, m))
    else:
        df = build_full_universe(force_refresh=force_refresh, log=lambda m: warn(log, m))

    if limit:
        df = df.head(limit)

    if universe_mode == "portfolio":
        df = attach_names(df, log=lambda m: warn(log, m))

    if custom_tickers:
        new_rows = [t for t in custom_tickers if t not in set(df["Ticker"])]
        if new_rows:
            new_df = pd.DataFrame({
                "Ticker": new_rows, "Source": "Custom", "Index": "",
            })
            new_df = attach_names(new_df, log=lambda m: warn(log, m))
            df = pd.concat([df, new_df], ignore_index=True)

    counts = df["Source"].value_counts().to_dict()
    info(log, f"Universe mode      : {universe_mode}")
    info(log, f"Tickers total      : {len(df)}")
    for source, count in counts.items():
        info(log, f"  {source:<18} {count}")
    ok(log, "Ticker universe built")
    return df


# ==================================================================
# STEP 3 — Fetch market data
# ==================================================================
def step_fetch_market_data(log, tickers):
    section(log, "STEP 3 — Fetch market data")
    from modules.market_data import refresh_all_history

    t0 = time.time()
    result = refresh_all_history(tickers, log=lambda m: warn(log, m))
    elapsed = time.time() - t0

    info(log, f"Full backfills     : {result['full_backfill']}")
    info(log, f"Incremental updates: {result['incremental']}")
    if result["failed"]:
        warn(log, f"Failed to fetch {len(result['failed'])} ticker(s) this run "
                   f"(will retry next run): {result['failed'][:10]}"
                   f"{'...' if len(result['failed']) > 10 else ''}")
    ok(log, f"Market data refresh done in {elapsed:.1f}s")
    return result


# ==================================================================
# STEP 4 — Detect patterns
# ==================================================================
def step_detect_patterns(log, tickers):
    section(log, "STEP 4 — Detect patterns")
    from modules.market_data import get_history
    from modules.pattern_bathtub import detect_bathtub
    from modules.pattern_cup_handle import detect_cup_handle
    from modules.pattern_flag import detect_flag
    from modules.pattern_double import detect_double
    from modules.pattern_golden_cross import detect_golden_cross
    from modules.pattern_breakout import detect_breakout
    from modules.pattern_overextension import detect_overextension
    from modules.pattern_forming import detect_all_forming

    # detect_cup_handle is checked before detect_bathtub — a cup+handle
    # match implies the same underlying bowl shape would also match as a
    # plain bathtub, so bathtub is only used as the fallback to avoid
    # reporting one setup as two matches.
    #
    # detect_overextension isn't a "buy" pattern — it's a bearish risk
    # signal (ticker up 100%+ recently) that runs on every ticker,
    # including held positions, so it automatically drags down a
    # parabolic mover's star rating (modules/priority.py) and can
    # trigger a TRIM/SELL call on a held position that's run up huge
    # (modules/positions.py) without either of those files needing to
    # know anything about it specifically.
    other_detectors = [detect_flag, detect_double, detect_golden_cross, detect_breakout, detect_overextension]

    matches = []
    forming_matches = []
    for t in tickers:
        history = get_history(t)
        if history.empty:
            continue

        cup_match = detect_cup_handle(history)
        if cup_match:
            cup_match.ticker = t
            matches.append(cup_match)
        else:
            bathtub_match = detect_bathtub(history)
            if bathtub_match:
                bathtub_match.ticker = t
                matches.append(bathtub_match)

        for detect in other_detectors:
            match = detect(history)
            if match:
                match.ticker = t
                matches.append(match)

        # Leading-edge versions of the same setups above (still in
        # progress, not yet complete) — see modules/pattern_forming.py.
        # Kept in a separate list: these aren't backtested the way the
        # completed patterns are, so they don't feed Priority/stars or
        # the Recommendation verdict, just their own report sheet.
        for fm in detect_all_forming(history):
            fm.ticker = t
            forming_matches.append(fm)

    by_pattern = {}
    for m in matches:
        by_pattern.setdefault(m.pattern, []).append(m)
    for pattern_name, group in by_pattern.items():
        info(log, f"{pattern_name} matches: {len(group)}")
    for m in matches[:15]:
        info(log, f"  {m.ticker:<8} {m.pattern:<24} {m.note}")
    if len(matches) > 15:
        info(log, f"  ... and {len(matches) - 15} more (see report)")
    ok(log, "Pattern detection done")

    if forming_matches:
        info(log, f"Forming signals (not yet complete): {len(forming_matches)}")
    return matches, forming_matches


# ==================================================================
# STEP 5 — Check earnings
# ==================================================================
def step_check_earnings(log, tickers):
    section(log, "STEP 5 — Check upcoming earnings")
    from modules.earnings import refresh_earnings_cache, in_earnings_window

    earnings_df = refresh_earnings_cache(tickers, log=lambda m: log.debug(m))
    in_window = earnings_df[earnings_df["NextEarningsDate"].apply(in_earnings_window)]
    info(log, f"Tickers with earnings in the next {SETTINGS['earnings_window_days']} days: {len(in_window)}")
    ok(log, "Earnings check done")
    return earnings_df


# ==================================================================
# STEP 6 — Read open positions (for Buy/Sell/Hold)
# ==================================================================
def step_read_positions(log):
    section(log, "STEP 6 — Read open positions")
    from modules.ticker_universe import get_open_positions
    from modules.options import get_option_positions

    try:
        positions_df = get_open_positions()
    except (FileNotFoundError, ValueError) as e:
        warn(log, f"Could not read open positions ({e}) — skipping Buy/Sell/Hold analysis")
        positions_df = None
    else:
        info(log, f"Open positions: {len(positions_df)}")

    options_df = get_option_positions(log=lambda m: warn(log, m))
    if not options_df.empty:
        info(log, f"Open option positions: {len(options_df)} "
                  f"({options_df['Underlying'].nunique()} underlying ticker(s))")

    if positions_df is None and options_df.empty:
        return None, options_df
    ok(log, "Positions loaded")
    return positions_df, options_df


# ==================================================================
# STEP 7 — Build report
# ==================================================================
def step_build_report(log, tickers_df, matches, earnings_df, run_stats, custom_tickers=None,
                       positions_df=None, options_df=None, forming_matches=None):
    section(log, "STEP 7 — Build report")
    from modules.report_screener import build_report

    out_path, highlights = build_report(tickers_df, matches, earnings_df, run_stats,
                                         custom_tickers=custom_tickers, positions_df=positions_df,
                                         options_df=options_df, forming_matches=forming_matches)
    ok(log, f"Report saved → {out_path.name}")
    return out_path, highlights


# ==================================================================
# MAIN
# ==================================================================
def main():
    log, log_file = setup_logging()
    start_time = dt.datetime.now()

    banner(log, f"Stock Screener  —  {start_time.strftime('%A %B %d, %Y  %H:%M')}")
    info(log, f"Log file: {log_file}")

    args = [a.lower() for a in sys.argv[1:]]
    check_only = "--check" in args
    force_refresh = "--refresh-universe" in args
    universe_mode = "portfolio"
    if "--universe" in args:
        idx = args.index("--universe")
        if idx + 1 < len(sys.argv[1:]):
            universe_mode = sys.argv[1:][idx + 1].lower()
    limit = None
    if "--limit" in args:
        idx = args.index("--limit")
        if idx + 1 < len(sys.argv[1:]):
            limit = int(sys.argv[1:][idx + 1])

    cli_custom_tickers = []
    if "--add" in args:
        idx = args.index("--add")
        if idx + 1 < len(sys.argv[1:]):
            raw = sys.argv[1:][idx + 1]
            cli_custom_tickers = _valid_tickers(raw)

    info(log, f"MODE: {'CONFIG CHECK ONLY' if check_only else f'FULL SCREENER RUN (universe={universe_mode})'}")

    lock_path = ROOT_DIR / ".screener_run.lock"
    if not acquire_lock(lock_path, SETTINGS.get("lock_max_age_sec", 3600)):
        warn(log, "Another screener run appears to already be in progress (lock held) — skipping this run.")
        sys.exit(0)

    try:
        if not step_check_config(log):
            err(log, "Config check failed — fix the issues above and re-run.")
            sys.exit(1)
        ok(log, "Config check passed")

        if check_only:
            banner(log, "Config check complete — all good")
            sys.exit(0)

        interactive_custom = prompt_for_custom_tickers(log)
        custom_tickers = sorted(set(cli_custom_tickers) | set(interactive_custom))
        if cli_custom_tickers:
            info(log, f"--add tickers: {', '.join(cli_custom_tickers)}")

        tickers_df = step_build_universe(log, universe_mode, limit, force_refresh, custom_tickers)
        tickers = tickers_df["Ticker"].tolist()

        positions_df, options_df = step_read_positions(log)
        if positions_df is not None:
            # Make sure every held position gets a pattern scan even in
            # "portfolio"/limited/custom runs, so Buy/Sell/Hold always
            # has fresh data to work with.
            held = positions_df["Ticker"].tolist()
            missing = [t for t in held if t not in set(tickers)]
            tickers.extend(missing)
        if options_df is not None and not options_df.empty:
            # Same reasoning — an option's underlying needs a fresh
            # pattern scan even if it isn't itself a stock holding or
            # part of this run's universe.
            held_underlyings = options_df["Underlying"].unique().tolist()
            missing = [t for t in held_underlyings if t not in set(tickers)]
            tickers.extend(missing)

        step_fetch_market_data(log, tickers)
        matches, forming_matches = step_detect_patterns(log, tickers)
        earnings_df = step_check_earnings(log, tickers)

        elapsed = (dt.datetime.now() - start_time).total_seconds()
        run_stats = {"tickers_scanned": len(tickers), "elapsed_sec": round(elapsed, 1)}
        out_path, highlights = step_build_report(log, tickers_df, matches, earnings_df, run_stats,
                                                   custom_tickers, positions_df, options_df, forming_matches)

        from modules.notify import notify_success, build_urgent_alerts, notify_urgent_alerts
        notify_success(
            "Stock Screener — scan complete",
            f"{highlights['buy_calls']} new BUY call(s), {highlights['sell_trim']} position(s) flagged "
            f"SELL/TRIM, {highlights['candidates']} candidate(s) total.\n{out_path.name}",
        )
        notify_urgent_alerts(build_urgent_alerts(forming_matches, positions_df, options_df))

    except FileNotFoundError as e:
        err(log, f"File not found: {e}")
        log.debug(traceback.format_exc())
        sys.exit(1)
    except ImportError as e:
        err(log, f"Import error: {e}")
        err(log, "Make sure dependencies are installed: pip install -r requirements.txt")
        log.debug(traceback.format_exc())
        sys.exit(1)
    except Exception as e:
        err(log, f"Unexpected error: {type(e).__name__}: {e}")
        err(log, "Full traceback saved to log file — open it for details.")
        log.debug(traceback.format_exc())
        sys.exit(1)
    finally:
        release_lock(lock_path)

    elapsed = (dt.datetime.now() - start_time).total_seconds()
    banner(log, f"DONE  —  completed in {elapsed:.1f}s  —  log saved")
    info(log, f"→ {log_file}")
    info(log, "")


if __name__ == "__main__":
    main()
