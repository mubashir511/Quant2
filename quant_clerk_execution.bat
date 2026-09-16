@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" clerk_execution_job.py >> "%~dp0clerk_execution_bat_output.txt" 2>&1
