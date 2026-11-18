"""
LeadTime Radar - Adim 2: Feature engineering (SQL) + zaman bazli split + BASELINE

Amac:
  1) Modelin kullanacagi ozellikleri (feature) SQL ile uretmek
  2) Veriyi zamana gore train/test diye ayirmak
  3) "ERP'nin sabit termini ne kadar yaniliyor?" sorusunu SAYIYA dokmek
     (modelimizin degeri, bu sayiyla kiyaslanarak olculecek)

Calistirma:  python 02_features_baseline.py
Gereken:     once 01_generate_data.py calismis olmali (leadtime.db)
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).with_name("leadtime.db")
SPLIT_DATE = "2025-01-01"   # bundan oncesi = egitim, sonrasi = test

con = sqlite3.connect(DB_PATH)

# Alt sorgu aramalarini hizlandirmak icin indeks
con.execute("""CREATE INDEX IF NOT EXISTS idx_po_supplier_deliv
               ON purchase_orders(supplier_id, actual_delivery_date)""")

# ---------------------------------------------------------------------------
# 1) FEATURE ENGINEERING (SQL)
# ---------------------------------------------------------------------------
# KRITIK KURAL: Bir siparisin ozelligi, o siparis verildigi ANDA bilinebilecek
# seylerden uretilmeli. "Tedarikcinin son 90 gunluk ortalama gecikmesi" icin
# sadece order_date'ten ONCE TESLIM ALINMIS siparislere bakariz.
# (Henuz gelmemis siparisin gecikmesini bilemeyiz - bilirsek hile olur.)
FEATURE_SQL = """
SELECT
    po.po_id,
    po.order_date,
    po.supplier_id,
    po.material_id,
    s.country,
    m.category,
    po.quantity,
    po.quantity / m.typical_qty                    AS qty_ratio,
    po.erp_lead_time_days,
    CAST(strftime('%m', po.order_date) AS INTEGER) AS order_month,
    CAST(strftime('%w', po.order_date) AS INTEGER) AS order_weekday,

    -- Tedarikci yuku: son 14 gunde bu tedarikciye verilen siparis sayisi
    (SELECT COUNT(*) FROM purchase_orders p2
      WHERE p2.supplier_id = po.supplier_id
        AND p2.order_date <  po.order_date
        AND p2.order_date >= date(po.order_date, '-14 day')) AS supplier_orders_14d,

    -- Tedarikcinin son 90 gundeki ORTALAMA gecikmesi (sadece teslim alinmislar)
    (SELECT AVG(p3.delay_days) FROM purchase_orders p3
      WHERE p3.supplier_id = po.supplier_id
        AND p3.actual_delivery_date <  po.order_date
        AND p3.actual_delivery_date >= date(po.order_date, '-90 day')) AS supplier_avg_delay_90d,

    -- Ayni sey, gecikmenin oynakligi (standart sapma icin ortalama kareler)
    (SELECT AVG(p4.delay_days * p4.delay_days) FROM purchase_orders p4
      WHERE p4.supplier_id = po.supplier_id
        AND p4.actual_delivery_date <  po.order_date
        AND p4.actual_delivery_date >= date(po.order_date, '-90 day')) AS supplier_delay_sq_90d,

    -- Kac teslimata bakarak hesapladik (guvenilirlik gostergesi)
    (SELECT COUNT(*) FROM purchase_orders p5
      WHERE p5.supplier_id = po.supplier_id
        AND p5.actual_delivery_date <  po.order_date
        AND p5.actual_delivery_date >= date(po.order_date, '-90 day')) AS supplier_n_90d,

    -- HEDEF (modelin tahmin edecegi sey) ve ERP ile karsilastirma
    po.actual_lead_time_days,
    po.delay_days
FROM purchase_orders po
JOIN materials m  ON m.material_id  = po.material_id
JOIN suppliers s  ON s.supplier_id  = po.supplier_id
ORDER BY po.order_date
"""
df = pd.read_sql(FEATURE_SQL, con, parse_dates=["order_date"])

# Standart sapma = karekok(E[x^2] - (E[x])^2)
var = df["supplier_delay_sq_90d"] - df["supplier_avg_delay_90d"] ** 2
df["supplier_std_delay_90d"] = np.sqrt(var.clip(lower=0))
df = df.drop(columns=["supplier_delay_sq_90d"])

# Verinin basinda gecmis yok -> bos kalan ozellikleri ilk 90 gunde atariz
df = df.dropna(subset=["supplier_avg_delay_90d"]).reset_index(drop=True)
df["supplier_std_delay_90d"] = df["supplier_std_delay_90d"].fillna(0)

# Kaydet: sonraki adimlar bu tablodan okuyacak
df.to_sql("features", con, if_exists="replace", index=False)
con.commit()
print(f"Feature tablosu: {len(df)} satir, {df.shape[1]} sutun")

# ---------------------------------------------------------------------------
# 2) ZAMAN BAZLI SPLIT
# ---------------------------------------------------------------------------
train = df[df["order_date"] < SPLIT_DATE]
test = df[df["order_date"] >= SPLIT_DATE]
print(f"Egitim: {len(train)} siparis ({train['order_date'].min().date()} -> {train['order_date'].max().date()})")
print(f"Test:   {len(test)} siparis ({test['order_date'].min().date()} -> {test['order_date'].max().date()})")

# ---------------------------------------------------------------------------
# 3) BASELINE: ERP'nin sabit termini ne kadar yaniliyor? (TEST verisinde)
# ---------------------------------------------------------------------------
def report(name, pred, actual):
    err = actual - pred                      # + = tahminden GEC geldi
    mae = np.abs(err).mean()
    return {
        "yontem": name,
        "MAE (gun)": round(mae, 2),
        "ort. sapma (gun)": round(err.mean(), 2),
        "tahminden gec gelen %": round((err > 0).mean() * 100, 1),
        "5+ gun gec gelen %": round((err > 5).mean() * 100, 1),
    }

y = test["actual_lead_time_days"]

# Baseline A: ERP sabit termini (bugunku durum)
pred_erp = test["erp_lead_time_days"]

# Baseline B: "Akilli Excel" - ERP + tedarikcinin son 90 gun ortalama gecikmesi
# Bunu da kuruyoruz ki modelimiz sadece "salak" bir rakibi yenmis olmasin.
pred_naive = test["erp_lead_time_days"] + test["supplier_avg_delay_90d"]

results = pd.DataFrame([
    report("A) ERP sabit termin", pred_erp, y),
    report("B) ERP + tedarikci 90g ort. gecikme", pred_naive, y),
])
print("\n=== BASELINE SONUCLARI (test donemi) ===")
print(results.to_string(index=False))

# Tedarikci bazinda: ERP'nin en cok yanildigi yerler
t = test.assign(err=y - pred_erp)
by_sup = t.groupby("supplier_id").agg(
    siparis=("po_id", "count"),
    ort_gecikme=("err", "mean"),
    gec_gelen_pct=("err", lambda s: (s > 0).mean() * 100),
    p85_gecikme=("err", lambda s: s.quantile(0.85)),
).round(1)
print("\n=== TEDARIKCI BAZINDA (ERP termini vs gerceklesen) ===")
print(by_sup.to_string())

# Mevsim etkisi: aya gore ortalama gecikme
by_month = t.groupby("order_month")["err"].mean().round(1)
print("\n=== AYA GORE ORTALAMA GECIKME (gun) ===")
print(by_month.to_string())

con.close()
