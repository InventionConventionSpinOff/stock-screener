# ==================================================================
#   tests/test_forming_signals.py
#   modules.report_screener._build_forming_signals — the AI Outlook
#   column, which paraphrases already-computed facts for the
#   highest-confidence forming signals only (capped, since an all-us
#   scan can produce hundreds of these).
# ==================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import report_screener
from modules.pattern_types import PatternMatch


def _forming(ticker, pattern="Cup & Handle", confidence=0.8, note="cup forming"):
    return PatternMatch(pattern=pattern, direction="bullish", start_date=None, end_date=None,
                         ticker=ticker, confidence=confidence, note=note)


def test_empty_input_returns_empty_dataframe():
    df = report_screener._build_forming_signals([])
    assert df.empty


def test_ai_outlook_column_present_but_blank_without_ollama(monkeypatch):
    from modules import note_analysis
    monkeypatch.setattr(note_analysis, "ollama_available", lambda: False)
    df = report_screener._build_forming_signals([_forming("AAPL")])
    assert "AI Outlook" in df.columns
    assert df.iloc[0]["AI Outlook"] == ""


def test_ai_outlook_populated_when_ollama_available(monkeypatch):
    from modules import note_analysis
    monkeypatch.setattr(note_analysis, "ollama_available", lambda: True)
    monkeypatch.setattr(note_analysis, "summarize_forming_signal",
                         lambda *a, **k: "AAPL's Cup & Handle is 80% formed.")
    df = report_screener._build_forming_signals([_forming("AAPL")])
    assert df.iloc[0]["AI Outlook"] == "AAPL's Cup & Handle is 80% formed."


def test_outlook_capped_to_top_n_by_confidence(monkeypatch):
    from modules import note_analysis
    monkeypatch.setattr(note_analysis, "ollama_available", lambda: True)
    calls = []

    def _fake_summarize(ticker, *a, **k):
        calls.append(ticker)
        return f"outlook for {ticker}"
    monkeypatch.setattr(note_analysis, "summarize_forming_signal", _fake_summarize)
    monkeypatch.setitem(report_screener.SETTINGS, "forming_signals_ai_outlook_top_n", 2)

    matches = [_forming(f"T{i}", confidence=0.5 + i * 0.01) for i in range(5)]
    df = report_screener._build_forming_signals(matches)

    assert len(calls) == 2  # only the top 2 by confidence got an LLM call
    populated = df[df["AI Outlook"] != ""]
    assert len(populated) == 2
    # the two populated rows should be the two highest-confidence ones
    assert set(populated["Ticker"]) == set(df.sort_values("Confidence", ascending=False)["Ticker"].head(2))


def test_a_failed_summarize_call_leaves_outlook_blank_not_crashing(monkeypatch):
    from modules import note_analysis
    monkeypatch.setattr(note_analysis, "ollama_available", lambda: True)
    monkeypatch.setattr(note_analysis, "summarize_forming_signal", lambda *a, **k: "")
    df = report_screener._build_forming_signals([_forming("AAPL")])
    assert df.iloc[0]["AI Outlook"] == ""
