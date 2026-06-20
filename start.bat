@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set PORT=8000
title Embodied RAG QA

echo ============================================
echo    Embodied Intelligence RAG QA System
echo ============================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Run from project root, create venv first.
    pause
    exit /b 1
)

echo Starting server... model load takes ~60-90s on CPU.
echo Browser opens automatically at http://localhost:%PORT% (just refresh if it shows refused).
echo To STOP: close this window or press Ctrl+C.
echo.

start "" cmd /c "ping -n 95 127.0.0.1 >nul & start http://localhost:%PORT%"

.venv\Scripts\python.exe -m uvicorn app.api.main:app --port %PORT%

echo.
echo Server stopped. Press any key to close.
pause >nul
