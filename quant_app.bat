@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -STA -WindowStyle Normal -File "%~dp0quant_app_tray.ps1"
