"""
LeadTime Radar - Adim 4: Risk motoru

Amac:
  Modelin tahminini (P50/P85) STOK bilgisiyle birlestirip her ACIK siparis icin
  "stok bitmeden gelir mi?" sorusunu yanitlamak -> Dusuk / Orta / Yuksek risk.

Calistirma:  python 04_risk_engine.py
Gereken:     01, 02, 03 numarali scriptler calismis olmali (models/ klasoru dahil).
"""
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DB_PATH = HERE / "leadtime.db"
MODEL_DIR = HERE / "models"
AS_OF = pd.Timestamp("2025-06-16")   # "bugun" (simulasyon tarihi)
SAFETY_DAYS = 4                      # guvenlik stogu = 4 gunluk tuketim
rng = np.random.default_rng(123)

con = sqlite3.connect(DB_PATH)

# ---------------------------------------------------------------------------
# 1) ENVANTER TABLOSU (ERP'deki stok raporu gibi: parca basina eldeki stok)
# ---------------------------------------------------------------------------
mat = pd.read_sql("SELECT material_id, name, supplier_id, daily_consumption, erp_lead_time_days FROM materials", con)
# Gercekci stok: planlamaci stogu ERP terminine gore tutar (stok ~ termin kadar gun yeter),
# ama bazen fazla bazen az olur -> log-normal dalgalanma.
cover_days = np.clip(mat["erp_lead_time_days"] * rng.lognormal(np.log(1.1), 0.35, len(mat)), 2, 90)
mat["stock_on_hand"] = np.round(mat["daily_consumption"] * cover_days)
mat["safety_stock"] = mat["daily_consumption"] * SAFETY_DAYS
mat["as_of"] = AS_OF.strftime("%Y-%m-%d")
mat[["material_id", "stock_on_hand", "safety_stock", "as_of"]].to_sql(
    "inventory", con, if_exists="replace", index=False)

# ---------------------------------------------------------------------------
# 2) ACIK SIPARISLER: AS_OF tarihinde verilmis ama henuz gelmemis olanlar
# ---------------------------------------------------------------------------
df = pd.read_sql("SELECT * FROM features ORDER BY order_date", con, parse_dates=["order_date"])
for c in ["supplier_id", "category", "country"]:      # Adim 3'teki ile ayni tipler
    df[c] = df[c].astype("category")

deliv = pd.read_sql("SELECT po_id, actual_delivery_date FROM purchase_orders", con,
                    parse_dates=["actual_delivery_date"])
df = df.merge(deliv, on="po_id")
open_po = df[(df["order_date"] <= AS_OF) & (df["actual_delivery_date"] > AS_OF)].copy()
print(f"As-of tarihi: {AS_OF.date()} | Acik siparis sayisi: {len(open_po)}")

# ---------------------------------------------------------------------------
# 3) MODEL TAHMINLERI (siparis verildigi anki bilgiyle)
# ---------------------------------------------------------------------------
FEATURES = ["erp_lead_time_days", "quantity", "qty_ratio", "order_month",
            "order_weekday", "supplier_orders_14d", "supplier_avg_delay_90d",
            "supplier_std_delay_90d", "supplier_n_90d", "supplier_id", "category", "country"]
preds = {}
for q in [50, 85, 95]:
    m = joblib.load(MODEL_DIR / f"quantile_{q}.joblib")
    preds[q] = m.predict(open_po[FEATURES])
P = np.maximum.accumulate(np.column_stack([preds[50], preds[85], preds[95]]), axis=1)
open_po["p50_delay"], open_po["p85_delay"], open_po["p95_delay"] = P[:, 0], P[:, 1], P[:, 2]

# ---------------------------------------------------------------------------
# 4) RISK HESABI (hepsi "bugunden kac gun sonra" cinsinden)
# ---------------------------------------------------------------------------
open_po = open_po.merge(mat[["material_id", "name", "daily_consumption",
                             "stock_on_hand", "safety_stock"]], on="material_id")
elapsed = (AS_OF - open_po["order_date"]).dt.days

def days_from_today(delay_col):
    # Beklenen varis = siparis tarihi + ERP termini + tahmin edilen gecikme
    arrival = open_po["erp_lead_time_days"] + open_po[delay_col]
    return np.maximum(np.ceil(arrival - elapsed), 1)   # en erken yarin

open_po["erp_arrival_days"] = np.maximum(open_po["erp_lead_time_days"] - elapsed, 0)
open_po["p50_arrival_days"] = days_from_today("p50_delay")
open_po["p85_arrival_days"] = days_from_today("p85_delay")

# Stok kac gun yeter?
open_po["days_to_zero"] = open_po["stock_on_hand"] / open_po["daily_consumption"]
open_po["days_to_safety"] = (open_po["stock_on_hand"] - open_po["safety_stock"]) / open_po["daily_consumption"]

# Stoksuz kalinacak gun sayisi: ERP'ye gore vs modelin P85 tahminine gore
open_po["shortage_days_erp"] = np.maximum(open_po["erp_arrival_days"] - open_po["days_to_zero"], 0)
open_po["shortage_days_p85"] = np.maximum(open_po["p85_arrival_days"] - open_po["days_to_zero"], 0)
open_po["extra_units_needed"] = np.ceil(open_po["shortage_days_p85"] * open_po["daily_consumption"])

def classify(r):
    if r["p50_arrival_days"] > r["days_to_zero"]:
        return "Yuksek"   # tipik senaryoda bile stok biter
    if r["p85_arrival_days"] > r["days_to_safety"]:
        return "Orta"     # makul kotu senaryoda guvenlik stogune dalinir / stok biter
    return "Dusuk"        # P85'te bile guvenlik stogu korunur
open_po["risk"] = open_po.apply(classify, axis=1)

# ---------------------------------------------------------------------------
# 5) GERIYE DONUK TEST (backtest): simulasyonda gercek sonucu biliyoruz
# ---------------------------------------------------------------------------
actual_days = (open_po["actual_delivery_date"] - AS_OF).dt.days
open_po["actual_stockout"] = (actual_days > open_po["days_to_zero"]).astype(int)
open_po["erp_flag"] = (open_po["erp_arrival_days"] > open_po["days_to_zero"]).astype(int)
open_po["model_flag"] = open_po["risk"].isin(["Orta", "Yuksek"]).astype(int)

print("\n=== RISK DAGILIMI ===")
print(open_po["risk"].value_counts().reindex(["Yuksek", "Orta", "Dusuk"]).to_string())

print("\n=== BACKTEST: Gercekte stoksuz kalan siparis orani (risk sinifina gore) ===")
bt = open_po.groupby("risk").agg(siparis=("po_id", "count"),
                                 gercek_stoksuz_pct=("actual_stockout", lambda s: s.mean() * 100)).round(1)
print(bt.reindex(["Yuksek", "Orta", "Dusuk"]).to_string())

n_real = open_po["actual_stockout"].sum()
print(f"\nGercekte stoksuz kalan siparis: {n_real} / {len(open_po)}")
for name, col in [("ERP termini", "erp_flag"), ("LeadTime Radar (Orta+Yuksek)", "model_flag")]:
    caught = ((open_po[col] == 1) & (open_po["actual_stockout"] == 1)).sum()
    flagged = open_po[col].sum()
    print(f"{name:30s}: {flagged:3d} siparisi isaretledi, gercek stoksuzlugun {caught}/{n_real} tanesini yakaladi "
          f"(yakalama %{caught / max(n_real, 1) * 100:.0f})")

# ---------------------------------------------------------------------------
# 6) KAYDET (arayuz bu tablodan okuyacak)
# ---------------------------------------------------------------------------
cols = ["po_id", "supplier_id", "material_id", "name", "order_date", "quantity",
        "erp_lead_time_days", "p50_delay", "p85_delay", "p95_delay",
        "erp_arrival_days", "p50_arrival_days", "p85_arrival_days",
        "stock_on_hand", "safety_stock", "daily_consumption",
        "days_to_zero", "days_to_safety", "shortage_days_erp", "shortage_days_p85",
        "extra_units_needed", "risk", "actual_delivery_date", "actual_stockout"]
out = open_po[cols].copy()
out["supplier_id"] = out["supplier_id"].astype(int)
out["order_date"] = out["order_date"].dt.strftime("%Y-%m-%d")
out["actual_delivery_date"] = out["actual_delivery_date"].dt.strftime("%Y-%m-%d")
out = out.round(1)
out.to_sql("open_orders_risk", con, if_exists="replace", index=False)
con.commit()

print("\n=== EN RISKLI 5 SIPARIS ===")
top = out.sort_values(["shortage_days_p85"], ascending=False).head(5)
print(top[["po_id", "supplier_id", "name", "erp_arrival_days", "p85_arrival_days",
           "days_to_zero", "shortage_days_p85", "risk"]].to_string(index=False))
con.close()
print("\nTablo 'open_orders_risk' yazildi.")
