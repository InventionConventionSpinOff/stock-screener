# ==================================================================
#   tests/test_note_analysis.py
#   modules.note_analysis — the local-LLM (Ollama) note-condition
#   checker. Mocks the HTTP calls so this suite runs with no Ollama
#   installed and no model downloaded.
# ==================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

from modules import note_analysis


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._json_data


def test_ollama_available_true_on_200(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(200))
    assert note_analysis.ollama_available() is True


def test_ollama_available_false_on_connection_error(monkeypatch):
    def _raise(*a, **k):
        raise requests.exceptions.ConnectionError("no server")
    monkeypatch.setattr(requests, "get", _raise)
    assert note_analysis.ollama_available() is False


def test_check_note_condition_parses_met_status(monkeypatch):
    fake_text = "STATUS: MET\nREASON: Cup & Handle completed today.\n"
    monkeypatch.setattr(requests, "post",
                         lambda *a, **k: _FakeResponse(200, {"response": fake_text}))
    result = note_analysis.check_note_condition(
        "AAPL", "hold until Cup & Handle finishes", "HOLD", "no strong signal", 3.2,
        completed_patterns=["Cup & Handle"], forming_patterns=[])
    assert result == {"status": "MET", "explanation": "Cup & Handle completed today."}


def test_check_note_condition_parses_not_met_status(monkeypatch):
    fake_text = "STATUS: NOT_MET\nREASON: Still forming, no completion yet.\n"
    monkeypatch.setattr(requests, "post",
                         lambda *a, **k: _FakeResponse(200, {"response": fake_text}))
    result = note_analysis.check_note_condition(
        "AAPL", "hold until Cup & Handle finishes", "HOLD", "no strong signal", 3.2,
        completed_patterns=[], forming_patterns=["Cup & Handle"])
    assert result["status"] == "NOT_MET"


def test_check_note_condition_unreachable_returns_unavailable(monkeypatch):
    def _raise(*a, **k):
        raise requests.exceptions.ConnectionError("no server")
    monkeypatch.setattr(requests, "post", _raise)
    result = note_analysis.check_note_condition("AAPL", "some note", "HOLD", "reason", 1.0)
    assert result["status"] == "UNAVAILABLE"


def test_check_note_condition_timeout_returns_unavailable(monkeypatch):
    def _raise(*a, **k):
        raise requests.exceptions.Timeout("too slow")
    monkeypatch.setattr(requests, "post", _raise)
    result = note_analysis.check_note_condition("AAPL", "some note", "HOLD", "reason", 1.0)
    assert result["status"] == "UNAVAILABLE"


def test_malformed_response_without_status_line_defaults_to_unsure():
    result = note_analysis._parse_response("I'm not sure what you mean by that note.")
    assert result["status"] == "UNSURE"


def test_invalid_status_value_falls_back_to_unsure():
    result = note_analysis._parse_response("STATUS: MAYBE\nREASON: ambiguous\n")
    assert result["status"] == "UNSURE"


def test_response_missing_reason_line_still_parses_status():
    result = note_analysis._parse_response("STATUS: MET\n")
    assert result["status"] == "MET"


# ------------------------------------------------------------------
# Deterministic percentage-threshold check — added after a live test
# showed the LLM get a plain "25% actual vs. 20% threshold" comparison
# backwards. These bypass the model entirely for simple cases.
# ------------------------------------------------------------------
def test_up_threshold_met_when_gain_exceeds_it():
    r = note_analysis._try_numeric_threshold_check("take profit once up 20%", 25.0)
    assert r["status"] == "MET"


def test_up_threshold_not_met_when_gain_is_below_it():
    r = note_analysis._try_numeric_threshold_check("take profit once up 20%", 12.0)
    assert r["status"] == "NOT_MET"


def test_down_threshold_met_when_loss_exceeds_it():
    r = note_analysis._try_numeric_threshold_check("sell if down 10%", -15.0)
    assert r["status"] == "MET"


def test_down_threshold_not_met_when_loss_is_smaller():
    r = note_analysis._try_numeric_threshold_check("sell if down 10%", -3.0)
    assert r["status"] == "NOT_MET"


def test_ambiguous_note_with_no_direction_word_defers_to_llm():
    assert note_analysis._try_numeric_threshold_check("review at 20%", 25.0) is None


def test_note_with_two_percentages_defers_to_llm():
    assert note_analysis._try_numeric_threshold_check("sell if up 20% or down 10%", 25.0) is None


def test_note_with_no_percentage_defers_to_llm():
    assert note_analysis._try_numeric_threshold_check("hold until Cup & Handle finishes", 5.0) is None


def test_check_note_condition_uses_numeric_shortcut_without_calling_ollama(monkeypatch):
    def _fail(*a, **k):
        raise AssertionError("should not call the LLM for an unambiguous numeric note")
    monkeypatch.setattr(requests, "post", _fail)
    result = note_analysis.check_note_condition(
        "NVDA", "take profit once up 20%", "TRIM", "up a lot", 25.0)
    assert result["status"] == "MET"
