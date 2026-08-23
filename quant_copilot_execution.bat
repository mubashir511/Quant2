@echo off
cd /d "%~dp0"
"C:\Users\mubas\AppData\Local\Programs\Python\Python314\python.exe" copilot_execution_job.py >> "%~dp0copilot_execution_bat_output.txt" 2>&1
