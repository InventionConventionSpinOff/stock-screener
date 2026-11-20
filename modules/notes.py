# ==================================================================
#   modules/notes.py
#   Lets you type a free-text note on a held position directly into
#   the "My Note" column of the "My Positions" sheet in the Excel
#   report (e.g. "hold until Cup & Handle finishes") and have it
#   survive into tomorrow's report.
#
#   The saved report file is the ONLY place a note lives — there's no
#   separate notes database — so this works by reading the note back
#   out of the most recently saved report before building a new one.
#   Since reports are never overwritten (see report_screener.py), the
#   note you typed into yesterday's file and saved is still sitting
#   there for today's run to pick up.
# ==================================================================
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DIR_OUTPUTS

NOTE_COLUMN = "My Note"

_REPORT_NAME_RE = re.compile(r"^ScreenerReport_(\d{4}-\d{2}-\d{2})(?:_(\d{6}))?\.xlsx$")


def _report_files_newest_first() -> list:
    """Every saved ScreenerReport_*.xlsx, newest first. A same-day
    rerun's timestamp-suffixed file (see report_screener.py's
    PermissionError fallback) sorts ahead of that day's plain-dated
    one, since it's the one that actually has your latest edits."""
    files = []
    for p in DIR_OUTPUTS.glob("ScreenerReport_*.xlsx"):
        m = _REPORT_NAME_RE.match(p.name)
        if not m:
            continue
        files.append((m.group(1), m.group(2) or "", p))
    files.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [f[2] for f in files]


def load_previous_notes() -> dict:
    """Returns {ticker: note_text} read from the most recent saved
    report's My Positions sheet. Falls back to progressively older
    reports if the newest one can't be read (e.g. still open and
    locked in Excel). Returns {} if no report exists yet, or none of
    them are readable."""
    import openpyxl

    for path in _report_files_newest_first():
        try:
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            try:
                if "My Positions" not in wb.sheetnames:
                    continue
                ws = wb["My Positions"]
                rows = ws.iter_rows(values_only=True)
                header = next(rows, None)
                if not header or "Ticker" not in header or NOTE_COLUMN not in header:
                    continue
                ticker_idx = header.index("Ticker")
                note_idx = header.index(NOTE_COLUMN)
                notes = {}
                for row in rows:
                    ticker = row[ticker_idx] if ticker_idx < len(row) else None
                    note = row[note_idx] if note_idx < len(row) else None
                    if ticker and note and str(note).strip():
                        notes[str(ticker).strip()] = str(note).strip()
                return notes
            finally:
                wb.close()
        except (OSError, PermissionError, KeyError, ValueError):
            continue  # this file is locked/corrupt/pre-dates the My Note column — try an older one

    return {}
