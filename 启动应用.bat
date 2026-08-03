@echo off
chcp 65001 >nul
title DistillationLab 一键安装与启动
echo ============================================
echo  DistillationLab - 一键安装依赖并启动
echo  首次运行会自动创建环境并下载依赖，请耐心等待
echo ============================================
echo.

where python >nul 2>nul
if %errorlevel%==0 (
    python "%~dp0launcher.py"
) else (
    echo [错误] 未找到 Python。
    echo 请先安装 Python 3.10+（勾选 Add python.exe to PATH）：
    echo   https://www.python.org/downloads/
    echo.
    pause
)
