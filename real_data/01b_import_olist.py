"""
LeadTime Radar - GERCEK VERI DOGRULAMASI, Adim 1: Olist verisini iceri aktar

Olist = Brezilya'daki bir e-ticaret pazaryerinin gercek (anonimlestirilmis) siparisleri.
Bizim projedeki karsiliklari:
    satici (seller)                         -> tedarikci
    order_estimated_delivery_date           -> "ERP'deki sabit termin" (pazaryerinin verdigi soz)
    order_delivered_customer_date           -> gerceklesen teslimat
    siparis edilen urun sayisi              -> miktar

DURUST NOT: Bu bir SATIN ALMA verisi degil, MUSTERIYE TESLIMAT verisi.
Problemin yapisi ayni ("sabit soz vs gercek dagilim") ama CV/README'de
"gercek e-ticaret teslimat verisinde dogruladim" diye yazilmali.

Calistirma (PowerShell):
    python 01b_import_olist.py --data "C:\\...\\olist_klasoru"

Klasorde su dosyalar olmali (Kaggle zip'inden):
    olist_orders_dataset.csv, olist_order_items_dataset.csv, olist_customers_dataset.csv,
    olist_sellers_dataset.csv, olist_products_dataset.csv, olist_geolocation_dataset.csv,
    product_category_name_translation.csv
Cikti: leadtime_olist.db  (bu scriptin yanina)
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(HERE / "olist_data"), help="Olist CSV klasoru")
ap.add_argument("--min-orders", type=int, default=100,
                help="Bu sayidan az siparisi olan saticilar atilir (tedarikci havuzu 'tekrar eden' olsun)")
args = ap.parse_args()
D = Path(args.data)

NEEDED = ["olist_orders_dataset.csv", "olist_order_items_dataset.csv", "olist_customers_dataset.csv",
          "olist_sellers_dataset.csv", "olist_products_dataset.csv", "olist_geolocation_dataset.csv",
          "product_category_name_translation.csv"]
missing = [f for f in NEEDED if not (D / f).exists()]
if missing:
    sys.exit(f"Eksik dosya(lar) {D} icinde: {missing}\n--data ile dogru klasoru ver.")

# ---------------------------------------------------------------------------
# 1) OKU
# ---------------------------------------------------------------------------
orders = pd.read_csv(D / "olist_orders_dataset.csv",
                     parse_dates=["order_purchase_timestamp", "order_delivered_customer_date",
                                  "order_estimated_delivery_date"])
items = pd.read_csv(D / "olist_order_items_dataset.csv")
cust = pd.read_csv(D / "olist_customers_dataset.csv")
sell = pd.read_csv(D / "olist_sellers_dataset.csv")
prod = pd.read_csv(D / "olist_products_dataset.csv")
trans = pd.read_csv(D / "product_category_name_translation.csv")
geo = pd.read_csv(D / "olist_geolocation_dataset.csv")

funnel = [("Toplam siparis", len(orders))]

# ---------------------------------------------------------------------------
# 2) TEMIZLE / FILTRELE (her adimda kac siparis kaldigini gosteriyoruz)
# ---------------------------------------------------------------------------
# a) Sadece teslim edilmis ve teslim tarihi olan siparisler (hedefi bilmeliyiz)
o = orders[(orders["order_status"] == "delivered")].dropna(subset=["order_delivered_customer_date"])
funnel.append(("Teslim edilmis", len(o)))

# b) Tek saticili siparisler: teslim tarihi siparis bazinda; birden fazla satici varsa
#    hangi saticinin gecikmeye yol actigini bilemeyiz.
n_sellers = items.groupby("order_id")["seller_id"].nunique()
single = n_sellers[n_sellers == 1].index
o = o[o["order_id"].isin(single)]
funnel.append(("Tek saticili", len(o)))

# c) 2016 cok seyrek (~270 siparis): 2017-01-01 ve sonrasi
o = o[o["order_purchase_timestamp"] >= "2017-01-01"]
funnel.append(("2017 ve sonrasi", len(o)))

# ---------------------------------------------------------------------------
# 3) SIPARIS-SATICI SEVIYESINDE OZELLIKLER (1 satir = 1 "PO")
# ---------------------------------------------------------------------------
prod = prod.merge(trans, on="product_category_name", how="left")
prod["category"] = prod["product_category_name_english"].fillna("unknown")
prod["product_weight_g"] = prod["product_weight_g"].fillna(prod["product_weight_g"].median())

it = items.merge(prod[["product_id", "category", "product_weight_g"]], on="product_id", how="left")
it["category"] = it["category"].fillna("unknown")
first_cat = it.sort_values("order_item_id").groupby("order_id")["category"].first()   # ilk urunun kategorisi
agg = it.groupby(["order_id", "seller_id"]).agg(
    quantity=("order_item_id", "count"),
    price=("price", "sum"),
    freight_value=("freight_value", "sum"),
    weight_g=("product_weight_g", "sum"),
).reset_index()
agg["category"] = agg["order_id"].map(first_cat)

df = o.merge(agg, on="order_id").merge(cust[["customer_id", "customer_zip_code_prefix", "customer_state"]],
                                       on="customer_id").merge(
    sell[["seller_id", "seller_zip_code_prefix", "seller_state", "seller_city"]], on="seller_id")

# ---------------------------------------------------------------------------
# 4) MESAFE: posta kodu oneki merkezleri arasi (haversine, km)
# ---------------------------------------------------------------------------
geo = geo[geo["geolocation_lat"].between(-34, 6) & geo["geolocation_lng"].between(-74, -34)]  # Brezilya disi hatalari at
cent = geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean()

def coords(zip_series):
    return (zip_series.map(cent["geolocation_lat"]).to_numpy(float),
            zip_series.map(cent["geolocation_lng"]).to_numpy(float))

lat1, lon1 = coords(df["seller_zip_code_prefix"])
lat2, lon2 = coords(df["customer_zip_code_prefix"])
p1, p2 = np.radians(lat1), np.radians(lat2)
a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
df["distance_km"] = 2 * 6371 * np.arcsin(np.sqrt(a))          # bulunamazsa NaN kalir (model NaN'i kaldirir)
df["same_state"] = (df["seller_state"] == df["customer_state"]).astype(int)

# ---------------------------------------------------------------------------
# 5) HEDEF: takvim gunu cinsinden sureler
# ---------------------------------------------------------------------------
buy_day = df["order_purchase_timestamp"].dt.normalize()
df["erp_lead_time_days"] = (df["order_estimated_delivery_date"].dt.normalize() - buy_day).dt.days
df["actual_lead_time_days"] = (df["order_delivered_customer_date"].dt.normalize() - buy_day).dt.days
df["delay_days"] = df["actual_lead_time_days"] - df["erp_lead_time_days"]
df = df[(df["erp_lead_time_days"] >= 0) & (df["actual_lead_time_days"] >= 0)]

# ---------------------------------------------------------------------------
# 6) TEDARIKCI HAVUZU: yeterince siparisi olan saticilar
# ---------------------------------------------------------------------------
cnt = df.groupby("seller_id").size()
keep = cnt[cnt >= args.min_orders].index
df = df[df["seller_id"].isin(keep)].copy()
funnel.append((f"Satici >= {args.min_orders} siparis ({len(keep)} satici)", len(df)))

df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
sup = (df.groupby("seller_id").agg(seller_state=("seller_state", "first"), seller_city=("seller_city", "first"),
                                   n_orders=("order_id", "count")).reset_index().sort_values("n_orders", ascending=False))
sup["supplier_id"] = range(1, len(sup) + 1)
sup["name"] = sup["supplier_id"].map(lambda i: f"Satici-{i:03d}")
df = df.merge(sup[["seller_id", "supplier_id"]], on="seller_id")
df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
df["po_id"] = np.arange(1, len(df) + 1)

po = pd.DataFrame({
    "po_id": df["po_id"], "order_id": df["order_id"], "supplier_id": df["supplier_id"],
    "category": df["category"], "customer_state": df["customer_state"], "seller_state": df["seller_state"],
    "order_ts": df["order_purchase_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
    "delivery_ts": df["order_delivered_customer_date"].dt.strftime("%Y-%m-%d %H:%M:%S"),
    "promised_date": df["order_estimated_delivery_date"].dt.strftime("%Y-%m-%d"),
    "quantity": df["quantity"], "price": df["price"].round(2), "freight_value": df["freight_value"].round(2),
    "weight_g": df["weight_g"], "distance_km": df["distance_km"].round(1), "same_state": df["same_state"],
    "erp_lead_time_days": df["erp_lead_time_days"], "actual_lead_time_days": df["actual_lead_time_days"],
    "delay_days": df["delay_days"],
})

# ---------------------------------------------------------------------------
# 7) SQLITE'A YAZ
# ---------------------------------------------------------------------------
DB = HERE / "leadtime_olist.db"
with sqlite3.connect(DB) as con:
    con.execute("DROP TABLE IF EXISTS purchase_orders")
    con.execute("DROP TABLE IF EXISTS suppliers")
    sup[["supplier_id", "seller_id", "name", "seller_state", "seller_city", "n_orders"]].to_sql(
        "suppliers", con, index=False)
    po.to_sql("purchase_orders", con, index=False)
    con.execute("CREATE INDEX idx_po_supplier ON purchase_orders(supplier_id, order_ts)")

print("=== SUZGEC (kac siparis kaldi) ===")
for name, n in funnel:
    print(f"{name:45s}: {n:7d}")
print(f"\nVeritabani: {DB}")
print(f"Tarih araligi: {po['order_ts'].min()[:10]} -> {po['order_ts'].max()[:10]}")
print(f"Mesafe hesaplanamayan siparis: {po['distance_km'].isna().mean() * 100:.1f}%")
d = po["delay_days"]
print("\n=== PAZARYERININ SOZU (ERP termini karsiligi) vs GERCEK ===")
print(f"Ortalama soz: {po['erp_lead_time_days'].mean():.1f} gun | Ortalama gerceklesen: {po['actual_lead_time_days'].mean():.1f} gun")
print(f"Ortalama sapma (gerceklesen - soz): {d.mean():.1f} gun | medyan {d.median():.0f}")
print(f"Sozden GEC gelen: %{(d > 0).mean() * 100:.1f} | 5+ gun gec: %{(d > 5).mean() * 100:.1f}")
