@echo off
chcp 65001 >nul
title DistillationLab Debug Mode
echo ============================================
echo  DistillationLab - Debug Mode
echo  If the program crashes, this window stays open
echo  Check startup_error.log for details
echo ============================================
echo.

start /wait DistillationLab.exe

if errorlevel 1 (
    echo.
    echo ============================================
    echo  Program crashed (exit code: %errorlevel%)
    echo  Check startup_error.log for details
    echo ============================================
) else (
    echo.
    echo  Program exited normally.
)

echo.
pause
