@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" mega_analysis_job.py >> "%~dp0mega_analysis_bat_output.txt" 2>&1
