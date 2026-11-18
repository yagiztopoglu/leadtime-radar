"""
LeadTime Radar - GERCEK VERI DOGRULAMASI, Adim 2: baseline + quantile model (Olist)

Sentetik pipeline'daki 02 + 03 numarali scriptlerin Olist (gercek) veri versiyonu.
Ayni fikir:  soz (ERP termini) vs gerceklesen  ->  P50/P85/P95 tahmini  ->  kalibrasyon testi.

Calistirma:  python 02_olist_experiment.py
Gereken:     once 01b_import_olist.py calismis olmali (leadtime_olist.db)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sqlite3

import numpy as np
import pandas as pd

from olist_common import BACKEND, DB, FEATURES, SPLIT, load_features, make_model, pinball, update_results

po = load_features()

# ---------------------------------------------------------------------------
# ZAMAN BAZLI SPLIT (+ "temizleme boslugu")
# ---------------------------------------------------------------------------
# Egitim: split'ten ONCE verilmis VE split'ten once TESLIM EDILMIS siparisler.
# (Mart sonunda verilip Nisan'da gelen siparisin sonucu, split aninda henuz bilinmiyordu.)
train = po[(po["order_ts"] < SPLIT) & (po["delivery_ts"] < SPLIT)]
test = po[po["order_ts"] >= SPLIT].copy()
print(f"Backend: {BACKEND}")
print(f"Egitim: {len(train)} siparis ({train['order_ts'].min().date()} -> {train['order_ts'].max().date()})")
print(f"Test:   {len(test)} siparis ({test['order_ts'].min().date()} -> {test['order_ts'].max().date()})")
y = test["delay_days"].to_numpy(float)

# ---------------------------------------------------------------------------
# BASELINE'LAR
# ---------------------------------------------------------------------------
train_mean_delay = train["delay_days"].mean()
pred_A = np.zeros(len(test))                                                    # A) pazaryerinin sozu
pred_B = test["seller_avg_delay_90d"].fillna(train_mean_delay).to_numpy()       # B) soz + satici 90g ort. sapma
seller_median_lead = train.groupby("supplier_id", observed=True)["actual_lead_time_days"].median()
static_lead = test["supplier_id"].map(seller_median_lead).astype(float).fillna(train["actual_lead_time_days"].median())
pred_C = (static_lead - test["erp_lead_time_days"]).to_numpy()                  # C) satici bazli STATIK termin (egitim medyani)

# ---------------------------------------------------------------------------
# QUANTILE MODELLERI
# ---------------------------------------------------------------------------
q_match = round(float((train["delay_days"] <= 0).mean()), 3)    # pazaryeri sozunun egitimdeki tutma orani
QS = sorted({0.50, 0.85, 0.95, q_match})
preds = {q: make_model(q).fit(train[FEATURES], train["delay_days"]).predict(test[FEATURES]) for q in QS}
models85 = make_model(0.85).fit(train[FEATURES], train["delay_days"])
P = np.maximum.accumulate(np.column_stack([preds[q] for q in QS]), axis=1)     # quantile'lar birbirini gecmesin
pq = {q: P[:, i] for i, q in enumerate(QS)}

# ---------------------------------------------------------------------------
# DEGERLENDIRME
# ---------------------------------------------------------------------------
res = {"n_train": int(len(train)), "n_test": int(len(test))}
print("\n=== 1) NOKTA TAHMINI (MAE, gun - dusuk iyi) ===")
rows = [("A) Pazaryeri sozu (ERP termini)", "A", pred_A), ("B) Soz + satici 90g ort. sapma", "B", pred_B),
        ("C) Satici bazli statik termin (egitim medyani)", "C", pred_C), ("D) Model P50", "D", pq[0.5])]
res["mae"] = {}
for name, key, p in rows:
    mae = float(np.abs(y - p).mean())
    res["mae"][key] = round(mae, 2)
    print(f"{name:50s} MAE={mae:5.2f} | ort. sapma={np.mean(y - p):+6.2f} | tahminden gec gelen %{(y > p).mean() * 100:5.1f}")

print("\n=== 2) KALIBRASYON: 'P85' dediginde gercekten %85 mi? ===")
res["coverage"] = {}
for q in [0.50, 0.85, 0.95]:
    c = float((y <= pq[q]).mean() * 100)
    res["coverage"][f"P{int(q * 100)}"] = round(c, 1)
    print(f"Model P{int(q * 100)}: siparislerin %{c:.1f}'i tahmin icinde geldi (hedef %{q * 100:.0f})")
res["coverage"]["platform"] = round(float((y <= 0).mean() * 100), 1)
print(f"Kiyas -> pazaryeri sozu: %{res['coverage']['platform']}'ini tuttu")

print("\n=== 3) PINBALL LOSS (dusuk iyi) ===")
for q in [0.85, 0.95]:
    print(f"P{int(q * 100)}: Pazaryeri sozu={pinball(y, np.zeros(len(y)), q):.2f} | "
          f"Baseline B={pinball(y, pred_B, q):.2f} | Baseline C={pinball(y, pred_C, q):.2f} | Model={pinball(y, pq[q], q):.2f}")

print("\n=== 4) GEC KALACAK SIPARISLERI ONCEDEN ISARETLEME (risk bayragi) ===")
late = y > 0
res["late_rate"] = round(float(late.mean() * 100), 1)
res["flags"] = {}
print(f"Testte soze gore gec gelen: %{late.mean() * 100:.1f} ({late.sum()} siparis). Pazaryeri sozu hicbirini onceden isaretlemez.")
for q in [0.85, 0.95]:
    flag = pq[q] > 0
    recall = float((flag & late).sum() / max(late.sum(), 1) * 100)
    res["flags"][f"P{int(q * 100)}"] = {"flagged_pct": round(float(flag.mean() * 100), 1),
                                         "late_in_flagged": round(float(late[flag].mean() * 100), 1),
                                         "late_in_unflagged": round(float(late[~flag].mean() * 100), 1),
                                         "recall": round(recall, 0)}
    print(f"Bayrak: P{int(q * 100)} > 0 -> {flag.mean() * 100:.1f}% siparis isaretlendi | "
          f"isaretlenenlerde gec orani %{late[flag].mean() * 100:.1f} (isaretlenmeyenlerde %{late[~flag].mean() * 100:.1f}) | "
          f"gec siparislerin %{recall:.0f}'ini yakaladi")

print("\n=== 5) OZELLIK ONEMI (%, P85 modeli) ===")
if BACKEND == "LightGBM":
    imp = pd.Series(models85.booster_.feature_importance(importance_type="gain"), index=FEATURES)
else:
    from sklearn.inspection import permutation_importance
    sub = test.sample(4000, random_state=1)
    r = permutation_importance(models85, sub[FEATURES], sub["delay_days"], n_repeats=2,
                               random_state=42, n_jobs=1, scoring="neg_mean_absolute_error")
    imp = pd.Series(r.importances_mean, index=FEATURES)
imp = (imp / imp.abs().sum() * 100).sort_values(ascending=False).round(1)
res["importance_top"] = {k: float(v) for k, v in imp.head(6).items()}
print(imp.to_string())

update_results("baseline_model", res)
out = test[["po_id", "supplier_id", "order_ts", "erp_lead_time_days", "actual_lead_time_days", "delay_days"]].copy()
out["supplier_id"] = out["supplier_id"].astype(int)
out["order_ts"] = out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
out["p50"], out["p85"], out["p95"] = pq[0.5], pq[0.85], pq[0.95]
out["pA"], out["pB"], out["pC"] = pred_A, pred_B, pred_C
con = sqlite3.connect(DB)
out.to_sql("predictions_test", con, if_exists="replace", index=False)
con.commit(); con.close()
print("\nTahminler 'predictions_test' tablosuna, ozet 'results.json' dosyasina yazildi.")
