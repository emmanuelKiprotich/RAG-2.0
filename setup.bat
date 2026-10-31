@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 (
    echo.
    echo Setup stopped with an error. Review the message above.
    pause
)
endlocal
