@echo off
cd /d "%~dp0"
"C:\Users\mubas\AppData\Local\Programs\Python\Python314\python.exe" mega_analysis_job.py >> "%~dp0mega_analysis_bat_output.txt" 2>&1
