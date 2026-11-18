@echo off
cd /d "%~dp0"
if not exist .venv (echo Once setup.bat calistir. & pause & exit /b 1)
if not exist real_data\olist_data\olist_orders_dataset.csv (
  echo Olist CSV dosyalari bulunamadi. real_data\olist_data\BURAYA_OLIST_CSV_KOY.txt dosyasina bak.
  pause & exit /b 1
)
call .venv\Scripts\activate
cd real_data
python 01b_import_olist.py --data olist_data || goto :hata
python 02_olist_experiment.py || goto :hata
python 03_olist_calibration.py || goto :hata
python 04_olist_charts.py || goto :hata
cd ..
echo Gercek veri deneyi tamam. Sonuclar uygulamanin "Dogrulama" sekmesinde ve docs/ klasorunde.
pause
exit /b 0
:hata
cd ..
echo HATA: Yukaridaki mesaji kopyalayip bak.
pause
exit /b 1
