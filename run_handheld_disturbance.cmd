@echo off
setlocal
cd /d "%~dp0"
"D:\acados_env\python.exe" -X utf8 "%~dp0run_handheld_disturbance.py" %*
endlocal
