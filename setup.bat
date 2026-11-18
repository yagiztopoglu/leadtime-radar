@echo off
cd /d "%~dp0"
echo [1/3] Sanal ortam olusturuluyor...
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
echo [2/3] Paketler kuruluyor...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt || goto :hata
echo [3/3] Sentetik veri, modeller ve grafikler uretiliyor...
python 01_generate_data.py || goto :hata
python 02_features_baseline.py || goto :hata
python 03_train_quantile.py || goto :hata
python 04_risk_engine.py || goto :hata
python 05_make_charts.py || goto :hata
echo.
echo Kurulum tamam. Uygulamayi acmak icin run_app.bat dosyasina cift tikla.
pause
exit /b 0
:hata
echo.
echo HATA: Yukaridaki mesaji kopyalayip bak.
pause
exit /b 1
