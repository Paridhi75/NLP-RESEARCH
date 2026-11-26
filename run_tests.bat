@echo off
setlocal
if exist "C:\Users\Lenovo\AppData\Local\Python\pythoncore-3.14-64" (
    set "PATH=C:\Users\Lenovo\AppData\Local\Python\pythoncore-3.14-64;C:\Users\Lenovo\AppData\Local\Python\pythoncore-3.14-64\Scripts;%PATH%"
)
if exist "C:\Users\Lenovo\AppData\Local\Python\bin" (
    set "PATH=C:\Users\Lenovo\AppData\Local\Python\bin;%PATH%"
)
set "TRANSFORMERS_OFFLINE=1"
set "HF_HUB_OFFLINE=1"
echo =======================================================
echo Running Unit and System Tests
echo =======================================================
python -m unittest tests/test_minilm_l12.py
pause
endlocal
