@echo off
title AI Road Damage Detection - Streamlit App
cd /d "%~dp0"
echo Starting AI Road Damage Detection Web App...
set YOLO_CONFIG_DIR=%CD%\.ultralytics
call .venv\Scripts\python.exe -m streamlit run app.py
pause
