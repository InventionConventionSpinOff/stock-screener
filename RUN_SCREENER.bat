@echo off
cd /d "%~dp0"

echo ============================================================
echo   Stock Screener - Setup and Run
echo ============================================================
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
echo Scanning your holdings + DJIA/S&P 500/Nasdaq-100 (~518 tickers)...
echo (Run RUN_SCREENER_WEEKLY_ALLUS.bat separately to scan every
echo  US-listed stock (~10,000+) - that one takes 30-45+ minutes.)
echo.
"%PYEXE%" run_screener.py --universe full
set SCREENER_EXIT=%errorlevel%

echo.
if %SCREENER_EXIT% neq 0 (
    echo Screener exited with an error - see above for details.
) else (
    echo Done. Open the "outputs" folder to see today's report.
)
pause
