"""
LeadTime Radar - Adim 1: Sentetik satinalma (PO) verisi uretici + SQLite sema

Amac: Gercek sirket verisine erisemedigimiz icin, ERP'deki gibi gorunen
ama icinde "gizli kurallar" olan bir veri seti uretiriz. Model sonra bu
gizli kurallari veriden kendi kendine ogrenmeye calisir.

Calistirma:  python 01_generate_data.py
Cikti:       leadtime.db  (SQLite)
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_SUPPLIERS = 12
N_MATERIALS = 80
N_PO = 12000
START, END = "2022-01-01", "2025-06-30"
DB_PATH = Path(__file__).with_name("leadtime.db")

rng = np.random.default_rng(SEED)  # ayni seed = her calistirmada ayni veri

# ---------------------------------------------------------------------------
# 1) TEDARIKCILER: her birinin "karakteri" var (modelin ogrenecegi gizli kisim)
# ---------------------------------------------------------------------------
countries = ["TR", "TR", "TR", "DE", "DE", "IT", "CN", "CN", "PL", "ES", "TR", "IN"]
suppliers = pd.DataFrame({
    "supplier_id": range(1, N_SUPPLIERS + 1),
    "name": [f"Tedarikci-{i:02d}" for i in range(1, N_SUPPLIERS + 1)],
    "country": countries,
})
# Gizli parametreler: DB'ye YAZMAYIZ. Gercek hayatta da bunlari kimse bilmez.
hidden = pd.DataFrame({
    "supplier_id": suppliers["supplier_id"],
    "bias": rng.uniform(0.82, 1.40, N_SUPPLIERS),         # ortalama ne kadar gec getirir
    "sigma": rng.uniform(0.08, 0.35, N_SUPPLIERS),        # ne kadar dengesiz/tahmin edilemez
    "load_sens": rng.uniform(0.0, 0.25, N_SUPPLIERS),     # yogunluktan ne kadar etkilenir
    "season_sens": rng.uniform(0.2, 1.2, N_SUPPLIERS),    # tatil/yaz/yilbasindan etkilenme
}).set_index("supplier_id")
# Uzak ulkeler (CN, IN) daha dengesiz olsun - gercekci
far = suppliers.set_index("supplier_id")["country"].isin(["CN", "IN"])
hidden.loc[far, "sigma"] += 0.10

# ---------------------------------------------------------------------------
# 2) MALZEMELER: her malzeme bir tedarikciye bagli, ERP'de SABIT termin var
# ---------------------------------------------------------------------------
categories = ["Elektronik", "Mekanik", "Hammadde", "Ambalaj", "Baglanti Elemani"]
materials = pd.DataFrame({
    "material_id": range(1, N_MATERIALS + 1),
    "name": [f"Parca-{i:03d}" for i in range(1, N_MATERIALS + 1)],
    "category": rng.choice(categories, N_MATERIALS),
    "supplier_id": rng.integers(1, N_SUPPLIERS + 1, N_MATERIALS),
    # ERP'deki "Planned Delivery Time" - tek bir sabit sayi:
    "erp_lead_time_days": rng.choice([7, 10, 14, 21, 30, 45], N_MATERIALS,
                                     p=[.15, .20, .25, .20, .12, .08]),
    "typical_qty": np.round(rng.lognormal(6, 0.8, N_MATERIALS)).clip(10),
    "daily_consumption": np.round(rng.lognormal(3.5, 0.6, N_MATERIALS)).clip(1),
})

# ---------------------------------------------------------------------------
# 3) SIPARISLER (PO): tarih, miktar, hangi malzeme
# ---------------------------------------------------------------------------
dates = pd.to_datetime(rng.choice(pd.date_range(START, END), N_PO)).sort_values()
po = pd.DataFrame({
    "po_id": range(1, N_PO + 1),
    "order_date": dates,
    "material_id": rng.integers(1, N_MATERIALS + 1, N_PO),
})
po = po.merge(materials[["material_id", "supplier_id", "erp_lead_time_days", "typical_qty"]],
              on="material_id")
po = po.sort_values("order_date").reset_index(drop=True)
po["po_id"] = range(1, N_PO + 1)

# Siparis miktari: tipik miktarin etrafinda dalgalanir
po["quantity"] = np.round(po["typical_qty"] * rng.lognormal(0, 0.4, N_PO)).clip(1)
po["qty_ratio"] = po["quantity"] / po["typical_qty"]

# Tedarikci yuku (proxy): o tedarikciye son 14 gunde verilen siparis sayisi.
# Sadece gecmise bakar -> modelde "gelecegi gormek" (leakage) olmaz.
def recent_orders(g):
    d = g["order_date"].values
    left = np.searchsorted(d, d - np.timedelta64(14, "D"), side="left")
    return pd.Series(np.arange(len(d)) - left, index=g.index)

po["supplier_recent_orders_14d"] = (
    po.groupby("supplier_id", group_keys=False).apply(recent_orders)
)
sup_mean_load = po.groupby("supplier_id")["supplier_recent_orders_14d"].transform("mean")
sup_std_load = po.groupby("supplier_id")["supplier_recent_orders_14d"].transform("std").replace(0, 1)
po["load_z"] = (po["supplier_recent_orders_14d"] - sup_mean_load) / sup_std_load

# ---------------------------------------------------------------------------
# 4) GERCEK TESLIMAT SURESI: gizli kurallar burada devreye girer
# ---------------------------------------------------------------------------
h = hidden.loc[po["supplier_id"]].reset_index(drop=True)
month = po["order_date"].dt.month

# Mevsimsellik: Temmuz-Agustos (yaz tatili) ve Aralik (yil sonu) yavaslar
season_base = np.where(month.isin([7, 8]), 0.18, np.where(month == 12, 0.22, 0.0))
season = 1 + season_base * h["season_sens"]

# Buyuk siparis daha uzun surer (sadece tipik miktarin ustunde)
qty_effect = 1 + 0.12 * np.clip(po["qty_ratio"] - 1, 0, None)

# Tedarikci yogunsa gecikir
load_effect = 1 + h["load_sens"] * np.clip(po["load_z"], -1, 3) * 0.5

# Rastgele dalgalanma (log-normal: hep pozitif, saga carpik - gercekci)
noise = rng.lognormal(0, h["sigma"])

mult = h["bias"] * season * qty_effect * load_effect * noise
actual = po["erp_lead_time_days"] * mult

# Nadir buyuk aksamalar (grev, gumruk, makine arizasi): %4 ihtimalle +7..20 gun
disruption = np.where(rng.random(N_PO) < 0.04, rng.uniform(7, 20, N_PO), 0)
po["actual_lead_time_days"] = np.round(actual + disruption).clip(1).astype(int)

po["promised_date"] = po["order_date"] + pd.to_timedelta(po["erp_lead_time_days"], unit="D")
po["actual_delivery_date"] = po["order_date"] + pd.to_timedelta(po["actual_lead_time_days"], unit="D")
po["delay_days"] = po["actual_lead_time_days"] - po["erp_lead_time_days"]

# ---------------------------------------------------------------------------
# 5) SQLite'a yaz (ERP export'una benzer 3 tablo)
# ---------------------------------------------------------------------------
SCHEMA = """
DROP TABLE IF EXISTS purchase_orders;
DROP TABLE IF EXISTS materials;
DROP TABLE IF EXISTS suppliers;

CREATE TABLE suppliers (
    supplier_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    country     TEXT
);
CREATE TABLE materials (
    material_id        INTEGER PRIMARY KEY,
    name               TEXT NOT NULL,
    category           TEXT,
    supplier_id        INTEGER REFERENCES suppliers(supplier_id),
    erp_lead_time_days INTEGER,   -- ERP'deki sabit planlanan termin
    typical_qty        REAL,
    daily_consumption  REAL
);
CREATE TABLE purchase_orders (
    po_id                INTEGER PRIMARY KEY,
    material_id          INTEGER REFERENCES materials(material_id),
    supplier_id          INTEGER REFERENCES suppliers(supplier_id),
    order_date           TEXT,
    quantity             REAL,
    erp_lead_time_days   INTEGER,
    promised_date        TEXT,
    actual_delivery_date TEXT,
    actual_lead_time_days INTEGER,
    delay_days           INTEGER
);
CREATE INDEX idx_po_supplier_date ON purchase_orders(supplier_id, order_date);
"""

po_out = po[["po_id", "material_id", "supplier_id", "order_date", "quantity",
             "erp_lead_time_days", "promised_date", "actual_delivery_date",
             "actual_lead_time_days", "delay_days"]].copy()
for c in ["order_date", "promised_date", "actual_delivery_date"]:
    po_out[c] = po_out[c].dt.strftime("%Y-%m-%d")

with sqlite3.connect(DB_PATH) as con:
    con.executescript(SCHEMA)
    suppliers.to_sql("suppliers", con, if_exists="append", index=False)
    materials.to_sql("materials", con, if_exists="append", index=False)
    po_out.to_sql("purchase_orders", con, if_exists="append", index=False)

# ---------------------------------------------------------------------------
# 6) Saglik kontrolu: veri mantikli mi?
# ---------------------------------------------------------------------------
print(f"Veritabani: {DB_PATH}")
print(f"Siparis sayisi: {len(po_out)}")
print(f"ERP termini ile ayni veya erken gelen: %{(po['delay_days'] <= 0).mean() * 100:.1f}")
print(f"Gec gelen siparisler: %{(po['delay_days'] > 0).mean() * 100:.1f}")
print(f"Ortalama gecikme: {po['delay_days'].mean():.1f} gun | "
      f"Medyan: {po['delay_days'].median():.0f} | "
      f"En kotu: {po['delay_days'].max()} gun")
print("\nTedarikci bazinda ortalama gecikme (gun):")
print(po.groupby("supplier_id")["delay_days"].mean().round(1).to_string())
