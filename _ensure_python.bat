@echo off
rem ==================================================================
rem   _ensure_python.bat
rem   Called by RUN_SCREENER.bat / RUN_SCREENER_WEEKLY_ALLUS.bat via
rem   "call". If Python is already on PATH, does nothing. Otherwise
rem   silently downloads and installs it for the current user only
rem   (no admin rights needed, no UAC prompt), then points PYEXE at
rem   the fresh install directly by its known install path - PATH
rem   itself only updates for brand-new processes, not this
rem   already-running one, so using the full path lets the rest of
rem   THIS SAME run continue without the user needing to close and
rem   reopen anything.
rem
rem   Sets PYEXE to the python command/path the caller should use.
rem   Exits with errorlevel 1 if Python still isn't available after
rem   the install attempt (e.g. no internet connection).
rem ==================================================================

where python >nul 2>nul
if not errorlevel 1 (
    set "PYEXE=python"
    exit /b 0
)

echo Python was not found on this computer - installing it automatically.
echo This only happens once; it takes a minute or two.
echo.

set "PY_INSTALLER_URL=https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe"
set "PY_INSTALLER=%TEMP%\python_setup_for_screener.exe"

echo Downloading Python...
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri '%PY_INSTALLER_URL%' -OutFile '%PY_INSTALLER%' } catch { exit 1 }"
if not exist "%PY_INSTALLER%" (
    echo.
    echo Could not download Python automatically - no internet connection,
    echo or something is blocking the download. Please install it yourself:
    echo   1. Go to https://python.org/downloads
    echo   2. Run the installer
    echo   3. Check the box "Add python.exe to PATH" before clicking Install
    echo   4. Run this file again
    exit /b 1
)

echo Installing Python - this takes a minute or two, please wait...
"%PY_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0
del "%PY_INSTALLER%" >nul 2>nul

set "PYEXE=%LocalAppData%\Programs\Python\Python312\python.exe"
if not exist "%PYEXE%" (
    echo.
    echo Python installation did not finish correctly. Please install it
    echo yourself from https://python.org/downloads - check the box
    echo "Add python.exe to PATH" - then run this file again.
    exit /b 1
)

echo Python installed successfully.
echo.
exit /b 0
