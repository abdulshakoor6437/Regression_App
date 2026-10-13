@echo off
cd /d "%~dp0"
echo ========================================================
echo        Starting RegressIQ Studio (Flask Web App)
echo ========================================================
if not exist ".venv\Scripts\python.exe" (
    echo Python environment not found. Initializing...
    "C:\Users\Abdul Rauf\.local\bin\uv.exe" venv --python 3.11
    "C:\Users\Abdul Rauf\.local\bin\uv.exe" pip install --python .venv\Scripts\python.exe -r requirements.txt
)

if not exist "insurance_model.joblib" (
    echo Training models...
    .venv\Scripts\python.exe train_models.py
)

echo Opening Web Browser at http://127.0.0.1:5000 ...
start http://127.0.0.1:5000
.venv\Scripts\python.exe app.py
pause
