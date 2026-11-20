# ==================================================================
#   modules/note_analysis.py
#   Checks whether a note you wrote on a held position (e.g. "hold
#   until Cup & Handle finishes") looks like its condition has been
#   met, using a small LOCAL language model via Ollama
#   (http://localhost:11434) — no cloud API, no API key, nothing
#   leaves the machine.
#
#   This is deliberately advisory, not authoritative: a 3B model will
#   sometimes misjudge an ambiguous note. It flags a possible match
#   for YOU to go check, the same way the plain pattern-change alert
#   would have, just with a better idea of which changes are actually
#   relevant to what you wrote. It never takes an action on its own.
#
#   If Ollama isn't installed or isn't running, every call here
#   degrades to status "UNAVAILABLE" — the note itself still displays
#   and still carries forward day to day; it just doesn't get an AI
#   judgment that day. The rest of the screener is completely
#   unaffected either way.
# ==================================================================
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from config import SETTINGS

OLLAMA_URL = SETTINGS["ollama_url"]
OLLAMA_MODEL = SETTINGS["ollama_model"]

# A health check (http GET) is near-instant if Ollama is up and hangs
# if it's not — kept short since this runs once per report, not once
# per ticker, and a slow failure here shouldn't stall the whole run.
HEALTH_TIMEOUT_SEC = 2
# The actual judgment call is a real (if small) model generation —
# give it real headroom, especially on a CPU-only machine.
REQUEST_TIMEOUT_SEC = 45

VALID_STATUSES = ("MET", "NOT_MET", "UNSURE")

_PROMPT_TEMPLATE = """You are helping a stock investor track a short note they wrote about a position they own. You are not giving financial advice — you are only judging whether the specific condition in their note appears to be satisfied today, based strictly on the facts listed below.

Ticker: {ticker}
Investor's note: "{note}"

Today's facts about this ticker — this is the ONLY information you have:
{context}

Judge ONLY whether the condition described in the note looks satisfied today, using ONLY the facts listed above. Do not invent a price, date, or any other fact that isn't listed above, even if the note mentions one — if the note refers to something you have no data for (e.g. a specific price level), say UNSURE rather than assuming its value. If a pattern the note names is listed as still FORMING rather than COMPLETED, that condition is NOT_MET yet, not UNSURE.

Respond in exactly this format and nothing else — two lines, no markdown:
STATUS: MET or NOT_MET or UNSURE
REASON: one short plain sentence
"""

# Best-effort deterministic check for notes that reference a plain
# gain/loss percentage threshold (e.g. "take profit once up 20%",
# "sell if down 10%") — handled directly in Python instead of trusting
# the local model's arithmetic. Added 2026-11-10 after a direct test
# showed llama3.2:3b call a 25% actual gain "NOT_MET" against a noted
# "up 20%" threshold — backwards, and a dangerous failure mode
# specifically because a wrong NOT_MET silently suppresses an alert
# that should have fired. Only handles the unambiguous case (exactly
# one percentage, a clear up/down direction word); anything messier
# falls through to the LLM.
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_UP_WORDS = ("up", "gain", "gains", "gained", "rise", "rises", "risen", "rose",
             "increase", "increased", "profit", "higher")
_DOWN_WORDS = ("down", "loss", "losses", "lost", "drop", "drops", "dropped",
               "fall", "falls", "fallen", "fell", "decline", "declined", "lower")


def _try_numeric_threshold_check(note: str, gain_pct) -> dict:
    """Returns a status dict if the note is an unambiguous plain
    percentage-gain/loss threshold, else None (defer to the LLM)."""
    if gain_pct is None:
        return None
    matches = _PCT_RE.findall(note)
    if len(matches) != 1:
        return None  # 0 or 2+ numbers isn't confidently a single simple threshold
    threshold = float(matches[0])
    lower = note.lower()
    is_up = any(w in lower for w in _UP_WORDS)
    is_down = any(w in lower for w in _DOWN_WORDS)
    if is_up == is_down:  # neither direction word present, or both -> ambiguous
        return None
    if is_up:
        met = gain_pct >= threshold
        direction = f"up-threshold of {threshold:.0f}%"
    else:
        met = gain_pct <= -threshold
        direction = f"down-threshold of {threshold:.0f}%"
    return {
        "status": "MET" if met else "NOT_MET",
        "explanation": f"Checked directly (not by the AI): actual {gain_pct:.1f}% vs. the note's {direction}.",
    }


def ollama_available() -> bool:
    """One cheap check per report run — not per ticker — to decide
    whether it's worth attempting any note analysis at all."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/version", timeout=HEALTH_TIMEOUT_SEC)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _build_context(action: str, reason: str, gain_pct, completed_patterns: list,
                    forming_patterns: list) -> str:
    lines = [
        f"- Current Buy/Hold/Sell recommendation: {action} ({reason})",
        f"- Unrealized gain/loss: {gain_pct}%",
        "- Patterns that COMPLETED today: " + (", ".join(completed_patterns) if completed_patterns else "none"),
        "- Patterns still FORMING (not yet complete): " + (", ".join(forming_patterns) if forming_patterns else "none"),
    ]
    return "\n".join(lines)


def check_note_condition(ticker: str, note: str, action: str, reason: str, gain_pct,
                          completed_patterns: list = None, forming_patterns: list = None) -> dict:
    """Returns {"status": "MET"|"NOT_MET"|"UNSURE"|"UNAVAILABLE", "explanation": str}.

    Caller is expected to have already checked ollama_available() once
    per run rather than relying on this failing fast per-ticker."""
    numeric_result = _try_numeric_threshold_check(note, gain_pct)
    if numeric_result is not None:
        return numeric_result

    context = _build_context(action, reason, gain_pct, completed_patterns or [], forming_patterns or [])
    prompt = _PROMPT_TEMPLATE.format(ticker=ticker, note=note, context=context)

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=REQUEST_TIMEOUT_SEC,
        )
        r.raise_for_status()
        text = r.json().get("response", "")
    except requests.exceptions.RequestException as e:
        return {"status": "UNAVAILABLE", "explanation": f"Ollama unreachable ({e.__class__.__name__})"}

    return _parse_response(text)


def _parse_response(text: str) -> dict:
    status = "UNSURE"
    explanation = text.strip()[:200] or "Model returned an empty response"
    for line in text.splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("STATUS:"):
            val = line.split(":", 1)[1].strip().upper()
            if val in VALID_STATUSES:
                status = val
        elif upper.startswith("REASON:"):
            explanation = line.split(":", 1)[1].strip()
    return {"status": status, "explanation": explanation}


# ------------------------------------------------------------------
# Forming-signal outlook — a different, lower-risk use of the same
# local model. Unlike check_note_condition (a MET/NOT_MET/UNSURE
# judgment that can suppress an alert if wrong), this just asks the
# model to paraphrase facts we already computed into one readable
# sentence — summarizing provided text is what small models are
# actually good at, as opposed to the arithmetic/judgment calls that
# proved unreliable above. Forming signals don't drive any BUY/HOLD/
# SELL decision (see modules/report_screener.py's _build_forming_signals
# docstring), so a so-so sentence here is cosmetic, not a risk.
# ------------------------------------------------------------------
_FORMING_PROMPT_TEMPLATE = """Write one short, plain sentence describing this still-forming (not yet complete) chart pattern for an investor. Use ONLY the facts listed below — do not invent a price target, a completion date, or any fact not listed.

Ticker: {ticker}
Pattern forming: {pattern} ({direction})
Detection confidence: {confidence:.0%}
Pattern detail: {note}
If this pattern completes, its historical track record is: {historical_edge}

Respond with exactly one plain sentence, no markdown, no disclaimers, no "I" statements:
"""


def summarize_forming_signal(ticker: str, pattern: str, direction: str, confidence: float,
                              note: str, historical_edge: str = "no backtest data yet") -> str:
    """Returns a one-sentence AI paraphrase, or "" if Ollama isn't
    reachable or the call fails — caller should have already checked
    ollama_available() once per report rather than per row."""
    prompt = _FORMING_PROMPT_TEMPLATE.format(
        ticker=ticker, pattern=pattern, direction=direction, confidence=confidence or 0.0,
        note=note or "(no detail)", historical_edge=historical_edge,
    )
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=REQUEST_TIMEOUT_SEC,
        )
        r.raise_for_status()
        text = r.json().get("response", "")
    except requests.exceptions.RequestException:
        return ""
    return text.strip().splitlines()[0][:300] if text.strip() else ""
