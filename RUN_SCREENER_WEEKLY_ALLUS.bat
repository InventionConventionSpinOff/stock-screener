@echo off
cd /d "%~dp0"

echo ============================================================
echo   Stock Screener - Weekly All-US Scan (~10,000+ tickers)
echo ============================================================
echo.
echo This scans every US-listed stock, not just your holdings and the
echo major indexes. It takes 30-45+ minutes (yfinance rate-limits this
echo many tickers) - that's expected, not a hang. Failed tickers retry
echo automatically on the NEXT run of this script, so coverage improves
echo over time even if today's run doesn't finish every single one.
echo.
echo Meant to be run about once a week (see Windows Task Scheduler if
echo you want this automatic) - for daily use, run RUN_SCREENER.bat
echo instead, which only covers your holdings + the major indexes and
echo finishes in under a minute.
echo.

call "%~dp0_ensure_python.bat"
if errorlevel 1 (
    pause
    exit /b 1
)

echo Installing required packages (first run may take a minute)...
"%PYEXE%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Dependency install failed - see the error above.
    pause
    exit /b 1
)

echo.
echo Scanning every US-listed stock - this will take a while...
echo.
"%PYEXE%" run_screener.py --universe all-us
set SCREENER_EXIT=%errorlevel%

echo.
if %SCREENER_EXIT% neq 0 (
    echo Screener exited with an error - see above for details.
) else (
    echo Done. Open the "outputs" folder to see today's report.
)
pause
