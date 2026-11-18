"""
LeadTime Radar - Adim 3: Quantile modelleri (P50 / P85 / P95) + kalibrasyon testi

Amac:
  Tek bir sayi degil, bir ARALIK tahmin etmek:
    "Bu siparis ERP'de 14 gun, ama %50 ihtimalle 17, %85 ihtimalle 22 gunde gelir."
  Sonra bu araligin SOZUNU TUTUP TUTMADIGINI (coverage) test etmek.

Calistirma:  pip install lightgbm scikit-learn joblib
             python 03_train_quantile.py
Gereken:     01 ve 02 numarali scriptler calismis olmali.
"""
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    BACKEND = "LightGBM"
except ImportError:  # lightgbm kurulu degilse ayni mantikla sklearn'e dus
    from sklearn.ensemble import HistGradientBoostingRegressor
    BACKEND = "sklearn HistGradientBoosting (lightgbm kurulu degil)"

HERE = Path(__file__).parent
DB_PATH = HERE / "leadtime.db"
MODEL_DIR = HERE / "models"
MODEL_DIR.mkdir(exist_ok=True)
SPLIT_DATE = "2025-01-01"
QUANTILES = [0.50, 0.85, 0.95]

# ---------------------------------------------------------------------------
# 1) VERIYI YUKLE
# ---------------------------------------------------------------------------
con = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM features ORDER BY order_date", con,
                 parse_dates=["order_date"])

NUM_FEATURES = ["erp_lead_time_days", "quantity", "qty_ratio", "order_month",
                "order_weekday", "supplier_orders_14d", "supplier_avg_delay_90d",
                "supplier_std_delay_90d", "supplier_n_90d"]
CAT_FEATURES = ["supplier_id", "category", "country"]
FEATURES = NUM_FEATURES + CAT_FEATURES

# Kategorik sutunlari 'category' tipine cevir (model bunlari etiket olarak gorsun,
# "tedarikci 12 > tedarikci 3" gibi sayisal anlam yuklemesin)
for c in CAT_FEATURES:
    df[c] = df[c].astype("category")

# HEDEF: gecikme (gun) = gerceklesen - ERP termini.
# Neden termin yerine gecikmeyi tahmin ediyoruz? 7 gunluk ve 45 gunluk parcalar
# ayni olcekte kiyaslanir; sonra "ERP termini + tahmin edilen gecikme" deriz.
TARGET = "delay_days"

train = df[df["order_date"] < SPLIT_DATE]
test = df[df["order_date"] >= SPLIT_DATE].copy()
X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]
print(f"Backend: {BACKEND}")
print(f"Egitim: {len(train)} | Test: {len(test)}\n")

# ---------------------------------------------------------------------------
# 2) HER QUANTILE ICIN AYRI MODEL EGIT
# ---------------------------------------------------------------------------
def make_model(q):
    if BACKEND == "LightGBM":
        return lgb.LGBMRegressor(
            objective="quantile", alpha=q,
            n_estimators=300, learning_rate=0.05,
            num_leaves=15, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_iter=300, learning_rate=0.05,
        max_leaf_nodes=15, min_samples_leaf=40,
        categorical_features="from_dtype", random_state=42,
    )

models = {}
for q in QUANTILES:
    m = make_model(q)
    m.fit(X_train, y_train)
    models[q] = m
    joblib.dump(m, MODEL_DIR / f"quantile_{int(q * 100)}.joblib")

pred = pd.DataFrame({f"p{int(q * 100)}": models[q].predict(X_test) for q in QUANTILES},
                    index=test.index)
# Quantile'lar birbirini gecmesin (P50 <= P85 <= P95)
pred = pd.DataFrame(np.maximum.accumulate(pred.values, axis=1),
                    columns=pred.columns, index=pred.index)

# ---------------------------------------------------------------------------
# 3) DEGERLENDIRME
# ---------------------------------------------------------------------------
def pinball(y, p, q):
    d = y - p
    return np.mean(np.maximum(q * d, (q - 1) * d))

# --- 3a) Nokta tahmini: P50 (medyan) vs baseline'lar (MAE, gun) ---
mae_erp = np.abs(y_test - 0).mean()                              # ERP: gecikme=0 varsayar
mae_naive = np.abs(y_test - test["supplier_avg_delay_90d"]).mean()
mae_model = np.abs(y_test - pred["p50"]).mean()
print("=== 1) NOKTA TAHMINI (MAE, gun - dusuk iyi) ===")
print(f"A) ERP sabit termin            : {mae_erp:.2f}")
print(f"B) ERP + tedarikci 90g ortalama: {mae_naive:.2f}")
print(f"C) Model P50                   : {mae_model:.2f}")

# --- 3b) ARALIK KALITESI: coverage ---
# P85 dogruysa, siparislerin ~%85'i P85 tahmininden ONCE veya o gun gelmeli.
print("\n=== 2) KALIBRASYON / COVERAGE (hedef: P50->%50, P85->%85, P95->%95) ===")
for q in QUANTILES:
    cov = (y_test <= pred[f"p{int(q * 100)}"]).mean() * 100
    print(f"Model P{int(q * 100)}: siparislerin %{cov:.1f}'i tahmin icinde geldi (hedef %{q * 100:.0f})")
cov_erp = (y_test <= 0).mean() * 100
cov_naive = (y_test <= test["supplier_avg_delay_90d"]).mean() * 100
print(f"Kiyas -> ERP termini: %{cov_erp:.1f} | Baseline B ortalamasi: %{cov_naive:.1f}")

# --- 3c) Pinball loss (quantile tahminlerin 'notu', dusuk iyi) ---
print("\n=== 3) PINBALL LOSS (dusuk iyi) ===")
for q in [0.85, 0.95]:
    key = f"p{int(q * 100)}"
    pl_model = pinball(y_test, pred[key], q)
    pl_erp = pinball(y_test, np.zeros(len(y_test)), q)
    pl_naive = pinball(y_test, test["supplier_avg_delay_90d"], q)
    print(f"P{int(q * 100)}: ERP={pl_erp:.2f} | Baseline B={pl_naive:.2f} | Model={pl_model:.2f}")

# --- 3d) Isinda bir kontrol: tedarikci bazinda P85 coverage ---
test["p50"], test["p85"], test["p95"] = pred["p50"], pred["p85"], pred["p95"]
by_sup = test.groupby("supplier_id", observed=True).apply(
    lambda g: pd.Series({
        "siparis": len(g),
        "P85_coverage_%": (g[TARGET] <= g["p85"]).mean() * 100,
        "ort_P85_gecikme_gun": g["p85"].mean(),
        "gercek_ort_gecikme": g[TARGET].mean(),
    }), include_groups=False
).round(1)
print("\n=== 4) TEDARIKCI BAZINDA P85 COVERAGE ===")
print(by_sup.to_string())

# --- 3e) Hangi ozellik ne kadar onemli? (P85 modeli icin) ---
if BACKEND == "LightGBM":
    imp = pd.Series(models[0.85].booster_.feature_importance(importance_type="gain"),
                    index=FEATURES)
else:
    from sklearn.inspection import permutation_importance
    r = permutation_importance(models[0.85], X_test, y_test, n_repeats=3,
                               random_state=42, n_jobs=1)
    imp = pd.Series(r.importances_mean, index=FEATURES)
imp = (imp / imp.sum() * 100).sort_values(ascending=False).round(1)
print("\n=== 5) OZELLIK ONEMI (%, P85 modeli) ===")
print(imp.to_string())

# ---------------------------------------------------------------------------
# 4) ORNEK CUMLELER + TAHMINLERI KAYDET
# ---------------------------------------------------------------------------
test["erp"] = test["erp_lead_time_days"]
print("\n=== 6) ORNEK TAHMINLER (rastgele 5 test siparisi) ===")
for _, r in test.sample(5, random_state=7).iterrows():
    print(f"PO {int(r.po_id)} | Tedarikci {int(r.supplier_id)} | ERP: {int(r.erp)} gun -> "
          f"P50: {r.erp + r.p50:.0f} gun, P85: {r.erp + r.p85:.0f} gun, "
          f"P95: {r.erp + r.p95:.0f} gun | Gerceklesen: {int(r.actual_lead_time_days)} gun")

out = test[["po_id", "order_date", "supplier_id", "erp", "p50", "p85", "p95",
            "actual_lead_time_days", TARGET]].copy()
out["order_date"] = out["order_date"].dt.strftime("%Y-%m-%d")
out.to_sql("predictions_test", con, if_exists="replace", index=False)
con.commit()
con.close()
print(f"\nModeller kaydedildi: {MODEL_DIR}")
print("Tahminler 'predictions_test' tablosuna yazildi.")
