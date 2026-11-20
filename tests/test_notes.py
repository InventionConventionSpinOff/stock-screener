# ==================================================================
#   tests/test_notes.py
#   modules.notes.load_previous_notes — reading a user-typed "My Note"
#   column back out of the most recently saved report so it carries
#   forward into the next run.
# ==================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from modules import notes


def _write_fake_report(dir_path: Path, filename: str, rows: list, sheet_name: str = "My Positions"):
    path = dir_path / filename
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    return path


def test_no_reports_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(notes, "DIR_OUTPUTS", tmp_path)
    assert notes.load_previous_notes() == {}


def test_reads_note_column_from_most_recent_report(tmp_path, monkeypatch):
    monkeypatch.setattr(notes, "DIR_OUTPUTS", tmp_path)
    _write_fake_report(tmp_path, "ScreenerReport_2026-11-09.xlsx", [
        {"Ticker": "AAPL", "My Note": "hold until Cup & Handle finishes"},
        {"Ticker": "MSFT", "My Note": ""},
    ])
    result = notes.load_previous_notes()
    assert result == {"AAPL": "hold until Cup & Handle finishes"}


def test_picks_the_newest_dated_report_not_an_older_one(tmp_path, monkeypatch):
    monkeypatch.setattr(notes, "DIR_OUTPUTS", tmp_path)
    _write_fake_report(tmp_path, "ScreenerReport_2026-11-04.xlsx", [
        {"Ticker": "AAPL", "My Note": "stale note"},
    ])
    _write_fake_report(tmp_path, "ScreenerReport_2026-11-09.xlsx", [
        {"Ticker": "AAPL", "My Note": "fresh note"},
    ])
    assert notes.load_previous_notes() == {"AAPL": "fresh note"}


def test_timestamped_same_day_rerun_wins_over_plain_dated_file(tmp_path, monkeypatch):
    monkeypatch.setattr(notes, "DIR_OUTPUTS", tmp_path)
    _write_fake_report(tmp_path, "ScreenerReport_2026-11-09.xlsx", [
        {"Ticker": "AAPL", "My Note": "earlier run today"},
    ])
    _write_fake_report(tmp_path, "ScreenerReport_2026-11-09_143000.xlsx", [
        {"Ticker": "AAPL", "My Note": "later rerun today"},
    ])
    assert notes.load_previous_notes() == {"AAPL": "later rerun today"}


def test_missing_my_positions_sheet_falls_back_to_older_report(tmp_path, monkeypatch):
    monkeypatch.setattr(notes, "DIR_OUTPUTS", tmp_path)
    _write_fake_report(tmp_path, "ScreenerReport_2026-11-04.xlsx", [
        {"Ticker": "AAPL", "My Note": "older but usable"},
    ])
    _write_fake_report(tmp_path, "ScreenerReport_2026-11-09.xlsx", [
        {"Ticker": "AAPL", "Note": "no My Note column in this one"},
    ])
    assert notes.load_previous_notes() == {"AAPL": "older but usable"}


def test_ignores_files_not_matching_report_naming_pattern(tmp_path, monkeypatch):
    monkeypatch.setattr(notes, "DIR_OUTPUTS", tmp_path)
    (tmp_path / "SomeOtherFile.xlsx").write_text("not a real workbook")
    assert notes.load_previous_notes() == {}
