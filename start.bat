@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PORT=8000
title 具身智能RAG问答系统

echo ============================================
echo    具身智能行业研究 RAG 问答系统
echo ============================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到 .venv 虚拟环境。
    echo 请确认本脚本位于项目根目录，且已创建 .venv（见 README 快速开始）。
    echo.
    pause
    exit /b 1
)

echo 正在启动服务，首次加载模型约 30-60 秒...
echo 就绪后（出现 Application startup complete）会自动打开浏览器：
echo     http://localhost:%PORT%
echo.
echo 【停止服务】关闭本窗口，或在窗口内按 Ctrl+C。
echo.

REM 后台等待约 50 秒后自动打开浏览器（不阻塞服务启动）
start "" cmd /c "ping -n 50 127.0.0.1 >nul & start http://localhost:%PORT%"

REM 前台运行服务：本窗口保持打开 = 服务运行中
.venv\Scripts\python.exe -m uvicorn app.api.main:app --port %PORT%

echo.
echo 服务已停止。按任意键关闭窗口。
pause >nul
