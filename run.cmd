@echo off
if "%~1"=="" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0quality.py" analyze
) else (
    "%~dp0.venv\Scripts\python.exe" "%~dp0quality.py" %*
)
exit /b %errorlevel%
