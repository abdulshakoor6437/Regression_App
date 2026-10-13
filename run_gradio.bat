@echo off
cd /d "%~dp0"
echo ========================================================
echo        Starting RegressIQ Gradio Dashboard
echo ========================================================
if not exist "insurance_model.joblib" (
    echo Training models...
    .venv\Scripts\python.exe train_models.py
)

echo Opening Gradio App at http://127.0.0.1:7860 ...
start http://127.0.0.1:7860
.venv\Scripts\python.exe gradio_app.py
pause
