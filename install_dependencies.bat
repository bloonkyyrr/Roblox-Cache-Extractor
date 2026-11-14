@echo off
setlocal

cd /d "%~dp0"
set "PYTHON_CMD="

where py >nul 2>nul
if not errorlevel 1 (
    py --version >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py"
)

if not defined PYTHON_CMD (
    where python >nul 2>nul
    if not errorlevel 1 (
        python --version >nul 2>nul
        if not errorlevel 1 set "PYTHON_CMD=python"
    )
)

if not defined PYTHON_CMD (
    echo Python was not found. Install Python 3.7 or newer and try again.
    goto :failed
)

echo Checking pip...
%PYTHON_CMD% -m pip --version >nul 2>nul
if errorlevel 1 (
    echo pip is missing. Bootstrapping pip with ensurepip...
    %PYTHON_CMD% -m ensurepip --upgrade
    if errorlevel 1 goto :failed
)

echo Installing dependencies from requirements.txt...
%PYTHON_CMD% -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto :failed

echo.
echo Dependencies installed successfully.
pause
exit /b 0

:failed
echo.
echo Dependency installation failed. Review the messages above and try again.
pause
exit /b 1
