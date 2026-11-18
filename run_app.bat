@echo off
cd /d "%~dp0"
if not exist .venv (echo Once setup.bat calistir. & pause & exit /b 1)
call .venv\Scripts\activate
python desktop_app.py
