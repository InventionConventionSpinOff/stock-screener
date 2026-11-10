@echo off
cd /d "%~dp0"

echo ============================================================
echo   Stock Screener - Custom Tickers
echo ============================================================
echo.
echo Scans your regular holdings PLUS whichever extra tickers you
echo give it here - handy for checking on something you're watching
echo without waiting for a full all-US run.
echo.

set TICKERS=%~1
if "%TICKERS%"=="" goto prompt
shift
:collect_args
if "%~1"=="" goto run
set TICKERS=%TICKERS%,%~1
shift
goto collect_args

:prompt
set /p TICKERS=Enter ticker(s) to check (comma or space separated):
if "%TICKERS%"=="" (
    echo No tickers entered - nothing to do.
    pause
    exit /b 1
)
set TICKERS=%TICKERS: =,%

:run
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
echo Scanning your holdings + requested tickers: %TICKERS%
echo.
"%PYEXE%" run_screener.py --universe portfolio --add %TICKERS%
set SCREENER_EXIT=%errorlevel%

echo.
if %SCREENER_EXIT% neq 0 (
    echo Screener exited with an error - see above for details.
) else (
    echo Done. Open the "outputs" folder to see today's report.
)
pause
