# ==================================================================
#   modules/options.py
#   Reads option positions out of the same broker Positions CSV export
#   used for equities (modules/ticker_universe.py), and enriches them
#   with a live quote for that exact contract plus whatever the
#   underlying's own pattern scan already found for it.
#
#   Two symbol formats recognized, tried in order:
#     - OCC-packed: "-AAPL251219C00200000" (optional leading "-",
#       YYMMDD, no separators, 8-digit strike = dollars*1000).
#       Verified against a real Fidelity export. thinkorswim is known
#       to also use real OCC symbols (sometimes space-padded, e.g.
#       "AAPL  251219C00200000") — the space-stripping before this
#       regex already handles that padding, so thinkorswim should
#       work here too, but that's inferred, not yet confirmed against
#       a real thinkorswim export.
#     - Schwab-style human-readable: "AAPL 01/17/2025 150.00 C"
#       (MM/DD/YYYY, decimal strike, space-separated). Best-effort
#       from Schwab's publicly documented export format — NOT yet
#       verified against a real Schwab export, since none was
#       available when this was written.
#   If your actual export uses a different layout than either, tell me
#   a real example row (redact the dollar amounts if you want, I only
#   need the symbol format) and I'll adjust the regex — anything that
#   doesn't match either pattern is surfaced as "unparsed" rather than
#   silently dropped, specifically so this failure mode is visible
#   instead of quietly losing a position.
#
#   Equity-side column names (Quantity/Qty, cost basis, last
#   price/Mark) and thinkorswim's multi-section CSV layout (separate
#   Equities/Options/Futures tables stacked in one file, each with
#   its own header row) are handled by modules.ticker_universe's
#   read_broker_csv() — also best-effort/unverified for thinkorswim.
# ==================================================================
import os
import re
import sys
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

# Root (1-6 letters) + YYMMDD expiration + C/P + 8-digit strike
# (dollars * 1000, so 00200000 == $200.000). Leading "-" is Fidelity's
# own convention for marking a row as an option contract.
FIDELITY_OPTION_RE = re.compile(r"^-?([A-Z]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})$")

# Root + MM/DD/YYYY + decimal strike + C/P, space-separated.
SCHWAB_OPTION_RE = re.compile(r"^([A-Z]{1,6})\s+(\d{2})/(\d{2})/(\d{4})\s+(\d+(?:\.\d+)?)\s+([CP])$")


def parse_option_symbol(raw_symbol: str) -> dict:
    """Returns {"underlying", "expiration" (date), "option_type"
    ("CALL"/"PUT"), "strike" (float)}, or None if raw_symbol doesn't
    match either recognized broker layout."""
    if not isinstance(raw_symbol, str):
        return None
    raw = raw_symbol.strip()

    packed = raw.replace(" ", "")
    m = FIDELITY_OPTION_RE.match(packed)
    if m:
        underlying, yy, mm, dd, cp, strike_raw = m.groups()
        try:
            expiration = dt.date(2000 + int(yy), int(mm), int(dd))
        except ValueError:
            return None
        return {
            "underlying": underlying,
            "expiration": expiration,
            "option_type": "CALL" if cp == "C" else "PUT",
            "strike": int(strike_raw) / 1000.0,
        }

    normalized = re.sub(r"\s+", " ", raw.upper())
    m = SCHWAB_OPTION_RE.match(normalized)
    if m:
        underlying, mm, dd, yyyy, strike_raw, cp = m.groups()
        try:
            expiration = dt.date(int(yyyy), int(mm), int(dd))
        except ValueError:
            return None
        return {
            "underlying": underlying,
            "expiration": expiration,
            "option_type": "CALL" if cp == "C" else "PUT",
            "strike": float(strike_raw),
        }

    return None


def _clean_money(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(r"[$,]", "", regex=True).str.strip(),
        errors="coerce",
    )


def get_option_positions(log=print) -> pd.DataFrame:
    """Reads the same Fidelity Positions CSV as
    modules.ticker_universe.get_open_positions(), but returns the
    option rows: Underlying, OptionType, Strike, Expiration,
    DaysToExpiration, Quantity (contracts), AvgCost (per share, same
    convention Fidelity uses — multiply by 100 for per-contract $),
    LastPrice (per share, as last reported by Fidelity).

    Returns an empty DataFrame (not an error) if there's no CSV, no
    option rows, or neither — a portfolio with no options open isn't
    a failure state."""
    from modules.ticker_universe import resolve_portfolio_csv_path, read_broker_csv

    csv_path = resolve_portfolio_csv_path()
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    df = read_broker_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if "Symbol" not in df.columns:
        return pd.DataFrame()

    lower_map = {c.lower(): c for c in df.columns}
    qty_col = lower_map.get("quantity") or lower_map.get("qty")
    # Fidelity/thinkorswim's cost basis columns are already per-share
    # ("Trade Price" for thinkorswim); Schwab's "Cost Basis" is a
    # TOTAL, divided by quantity below instead of being treated the
    # same way (see modules/ticker_universe.py's get_open_positions
    # for the equivalent equity-side handling).
    cost_col = lower_map.get("average cost basis") or lower_map.get("cost basis per share") or lower_map.get("trade price")
    total_cost_col = None if cost_col else lower_map.get("cost basis")
    price_col = lower_map.get("last price") or lower_map.get("price") or lower_map.get("mark")
    desc_col = lower_map.get("description")

    rows = []
    unparsed = []
    for _, r in df.iterrows():
        raw_symbol = r.get("Symbol")
        if not isinstance(raw_symbol, str):
            continue
        parsed = parse_option_symbol(raw_symbol)
        if parsed is None:
            # Only worth flagging if it also isn't a normal equity
            # symbol, cash sweep, or one of the trailing summary/
            # disclaimer rows Fidelity appends — those are handled
            # elsewhere and aren't options-parsing failures.
            stripped = raw_symbol.strip()
            looks_like_equity = bool(re.match(r"^[A-Z][A-Z0-9.\-]{0,6}$", stripped))
            looks_like_option_ish = any(ch.isdigit() for ch in stripped) and any(ch.isalpha() for ch in stripped)
            if not looks_like_equity and looks_like_option_ish and stripped.lower() not in ("pending activity",):
                unparsed.append(stripped)
            continue

        qty = pd.to_numeric(str(r.get(qty_col, "")).replace(",", ""), errors="coerce") if qty_col else None
        if cost_col:
            avg_cost = _clean_money(pd.Series([r.get(cost_col)]))[0]
        elif total_cost_col and pd.notna(qty) and qty != 0:
            # Schwab total cost basis is for the whole contract lot —
            # /100 to get back to a per-share-equivalent figure before
            # dividing by quantity (1 contract = 100 shares).
            avg_cost = _clean_money(pd.Series([r.get(total_cost_col)]))[0] / (qty * 100)
        else:
            avg_cost = None
        last_price = _clean_money(pd.Series([r.get(price_col)]))[0] if price_col else None
        if pd.isna(qty) or qty == 0:
            continue

        rows.append({
            "Underlying": parsed["underlying"],
            "OptionType": parsed["option_type"],
            "Strike": parsed["strike"],
            "Expiration": parsed["expiration"],
            "DaysToExpiration": (parsed["expiration"] - dt.date.today()).days,
            "Quantity": qty,
            "AvgCost": avg_cost,
            "LastPrice": last_price,
            "Description": r.get(desc_col, "") if desc_col else "",
            "RawSymbol": raw_symbol.strip(),
        })

    if unparsed:
        log(f"  {len(unparsed)} position row(s) looked like options but didn't match the expected "
            f"OCC symbol format: {unparsed[:5]}{'...' if len(unparsed) > 5 else ''} — "
            f"share one of these with me and I'll fix modules/options.py's OPTION_SYMBOL_RE.")

    return pd.DataFrame(rows)


def _fetch_live_contract_quote(underlying: str, expiration: dt.date, option_type: str, strike: float) -> dict:
    """Looks up this exact contract's current quote via yfinance's
    option chain for that expiration. Returns {} on any failure
    (contract expired/delisted, no chain data, network issue) rather
    than raising — a missing live quote just means falling back to
    Fidelity's own LastPrice, not a broken report."""
    try:
        import yfinance as yf
        chain = yf.Ticker(underlying).option_chain(expiration.strftime("%Y-%m-%d"))
        table = chain.calls if option_type == "CALL" else chain.puts
        match = table[(table["strike"] - strike).abs() < 0.01]
        if match.empty:
            return {}
        row = match.iloc[0]
        return {
            "LiveMark": round(float(row["lastPrice"]), 2) if pd.notna(row["lastPrice"]) else None,
            "Bid": round(float(row["bid"]), 2) if pd.notna(row["bid"]) else None,
            "Ask": round(float(row["ask"]), 2) if pd.notna(row["ask"]) else None,
            "ImpliedVol": round(float(row["impliedVolatility"]) * 100, 1) if pd.notna(row["impliedVolatility"]) else None,
            "OpenInterest": int(row["openInterest"]) if pd.notna(row["openInterest"]) else None,
            "InTheMoney": bool(row["inTheMoney"]) if pd.notna(row["inTheMoney"]) else None,
        }
    except Exception:
        return {}


def _recommendation_note(option_type: str, in_the_money, days_to_exp: int, stars: int, direction_bias: str) -> str:
    """A short, honest-effort read combining the option's own risk
    profile (moneyness, time left) with the underlying's existing
    pattern-confluence signal — not a pricing/Greeks-based verdict,
    just a sanity check on whether the underlying's chart still
    supports the bet this contract is making."""
    expiring_soon = days_to_exp <= 7
    wants_up = option_type == "CALL"
    supports = (wants_up and direction_bias == "bullish") or (not wants_up and direction_bias == "bearish")
    conflicts = (wants_up and direction_bias == "bearish") or (not wants_up and direction_bias == "bullish")

    parts = []
    if in_the_money is True:
        parts.append("currently ITM")
    elif in_the_money is False:
        parts.append("currently OTM")
    if expiring_soon:
        parts.append(f"expires in {days_to_exp}d — little time left for the thesis to play out")
    if stars >= 3 and supports:
        parts.append(f"underlying's {stars}-star pattern signal supports this side")
    elif stars >= 3 and conflicts:
        parts.append(f"underlying's {stars}-star pattern signal is on the OTHER side of this bet — worth a look")
    elif stars < 2:
        parts.append("no strong pattern signal on the underlying either way right now")
    return "; ".join(parts) if parts else "no notable signal"


def build_options_report(options_df: pd.DataFrame, matches_by_ticker: dict, log=print) -> pd.DataFrame:
    """Enriches raw option positions with a live contract quote and
    the underlying's existing pattern-confluence read. matches_by_ticker
    is the same {ticker: [PatternMatch,...]} dict report_screener.py
    already builds for equities — reused here instead of re-scanning."""
    from modules.priority import rate_ticker

    if options_df.empty:
        return options_df

    rows = []
    for _, pos in options_df.iterrows():
        underlying = pos["Underlying"]
        quote = _fetch_live_contract_quote(underlying, pos["Expiration"], pos["OptionType"], pos["Strike"])

        mark = quote.get("LiveMark") if quote.get("LiveMark") is not None else pos.get("LastPrice")
        avg_cost = pos.get("AvgCost")
        pl_per_contract = (mark - avg_cost) * 100 if pd.notna(mark) and pd.notna(avg_cost) else None
        pl_total = pl_per_contract * pos["Quantity"] if pl_per_contract is not None else None

        ticker_matches = matches_by_ticker.get(underlying, [])
        rating = rate_ticker(ticker_matches)
        directions = {m.direction for m in ticker_matches}
        direction_bias = "bullish" if "bullish" in directions and "bearish" not in directions else (
            "bearish" if "bearish" in directions and "bullish" not in directions else "mixed/none")

        rows.append({
            "Underlying": underlying,
            "Type": pos["OptionType"],
            "Strike": pos["Strike"],
            "Expiration": pos["Expiration"],
            "DaysToExpiration": pos["DaysToExpiration"],
            "Quantity": pos["Quantity"],
            "AvgCost": round(avg_cost, 2) if pd.notna(avg_cost) else None,
            "CurrentMark": round(mark, 2) if pd.notna(mark) else None,
            "UnrealizedPL": round(pl_total, 2) if pl_total is not None else None,
            "ImpliedVol%": quote.get("ImpliedVol"),
            "OpenInterest": quote.get("OpenInterest"),
            "InTheMoney": quote.get("InTheMoney"),
            "UnderlyingPriority": rating["stars"],
            "UnderlyingDirection": direction_bias,
            "Note": _recommendation_note(pos["OptionType"], quote.get("InTheMoney"),
                                          pos["DaysToExpiration"], rating["stars"], direction_bias),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("DaysToExpiration").reset_index(drop=True)
    return out


if __name__ == "__main__":
    from modules.ticker_universe import resolve_portfolio_csv_path

    opts = get_option_positions()
    print(f"{len(opts)} option position(s) found in {resolve_portfolio_csv_path()}")
    if not opts.empty:
        print(opts)
        report = build_options_report(opts, {})
        print(report)
