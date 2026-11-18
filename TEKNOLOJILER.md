# Kullanılan teknolojiler

| Araç | Ne için |
|---|---|
| Python 3.11 / 3.12 | Tüm proje |
| SQLite | ERP benzeri tablolar (tedarikçi, parça, sipariş, stok) ve SQL ile özellik üretimi |
| pandas, NumPy | Veri hazırlama |
| LightGBM | Quantile gradient boosting (P50 / P85 / P95 teslimat süresi) |
| scikit-learn | Metrikler ve yardımcı araçlar |
| joblib | Eğitilmiş modelleri kaydetme |
| matplotlib | README ve uygulamadaki grafikler |
| CustomTkinter, Pillow | Masaüstü arayüz |
| Anthropic API (opsiyonel) | Satın almacı için doğal dilde aksiyon raporu. Anahtar yoksa kural tabanlı rapor üretilir |
| python-dotenv | API anahtarını `.env` dosyasından okuma |

## Kurulum notları

- Windows'ta `setup.bat` her şeyi kurar ve modelleri üretir; uygulama `run_app.bat` ile açılır.
- `models/` ve `leadtime.db` repoda yoktur, `setup.bat` yeniden üretir. Kütüphane sürümü değişirse modeller yeniden eğitilmelidir.
- LLM raporu için `.env.example` dosyasını `.env` olarak kopyalayıp `ANTHROPIC_API_KEY` yazın. `.env` GitHub'a gönderilmez.
- Olist verisi lisans ve boyut nedeniyle repoda yoktur; `real_data/olist_data/BURAYA_OLIST_CSV_KOY.txt` dosyasına bakın.
