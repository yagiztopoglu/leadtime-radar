"""
LeadTime Radar - GERCEK VERI, Adim 3: Kalibrasyon kaymasini DUZELTME (conformal)

SORUN (02 numarali scriptte gorduk): Gercek dunya zamanla degisiyor. Teslimatlar test doneminde
egitim donemine gore hizlandi; model "eski dunyayi" ogrendigi icin P50 %50 yerine %62 tuttu,
P85 coverage aydan aya %77 ile %96 arasinda oynadi.

COZUM: Split conformal kalibrasyon.
  1) Modeli ESKI donemde egit (2017).
  2) Daha YENI bir doneme (2018 Ocak-Mart, "kalibrasyon kumesi") bak: model P85 dedi ama gercekte
     siparislerin yuzde kaci o sureden once geldi? Fark kadar tahmini kaydir (offset).
  3) KAYAN versiyon: test doneminde her hafta basinda, son 8 haftada teslim edilmis siparislerle
     offset'i yeniden hesapla (uretimde de boyle yapilir: sadece o ana kadar BILINEN sonuclar).

Calistirma:  python 03_olist_calibration.py
Gereken:     01b ve 02 numarali scriptler calismis olmali.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sqlite3

import numpy as np
import pandas as pd

from olist_common import BACKEND, DB, FEATURES, SPLIT, load_features, make_model, pinball, update_results

FIT_END = pd.Timestamp("2018-01-01")      # egitim: 2017 | kalibrasyon: 2018 Ocak-Mart | test: Nisan-Agustos
WINDOW_DAYS = 56                           # kayan kalibrasyon penceresi (8 hafta)
MIN_N = 300                                # pencerede en az bu kadar siparis yoksa statik offset kullan

po = load_features()
fit = po[(po["order_ts"] < FIT_END) & (po["delivery_ts"] < FIT_END)]
cal = po[(po["order_ts"] >= FIT_END) & (po["order_ts"] < SPLIT) & (po["delivery_ts"] < SPLIT)].copy()
test = po[po["order_ts"] >= SPLIT].copy()
print(f"Backend: {BACKEND}")
print(f"Egitim (2017): {len(fit)} | Kalibrasyon (2018 Oca-Mar): {len(cal)} | Test (Nis-Agu 2018): {len(test)}")

q_match = round(float((cal["delay_days"] <= 0).mean()), 3)    # pazaryeri sozunun kalibrasyon kumesindeki tutma orani
QS = sorted({0.50, 0.85, 0.95, q_match})
name = {q: ("pM" if q == q_match and q not in (0.5, 0.85, 0.95) else f"p{int(q * 100)}") for q in QS}

# ---------------------------------------------------------------------------
# 1) HAM MODEL (sadece 2017'de egitildi)
# ---------------------------------------------------------------------------
def monotone(d):
    M = np.maximum.accumulate(np.column_stack([d[q] for q in QS]), axis=1)
    return {q: M[:, i] for i, q in enumerate(QS)}

raw_cal, raw_test = {}, {}
for q in QS:
    m = make_model(q).fit(fit[FEATURES], fit["delay_days"])
    raw_cal[q], raw_test[q] = m.predict(cal[FEATURES]), m.predict(test[FEATURES])
raw_cal, raw_test = monotone(raw_cal), monotone(raw_test)

# ---------------------------------------------------------------------------
# 2) CONFORMAL OFFSET
# ---------------------------------------------------------------------------
def offset(scores, q):
    """Skorlarin (gercek - tahmin) sonlu-ornek duzeltmeli q-kuantili. Tahmine eklenince coverage ~q olur."""
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * q) / n)
    return float(np.quantile(scores, level))

y_cal = cal["delay_days"].to_numpy(float)
static_off = {q: offset(y_cal - raw_cal[q], q) for q in QS}
static_test = monotone({q: raw_test[q] + static_off[q] for q in QS})
print("\nStatik offset (gun):", {name[q]: round(static_off[q], 2) for q in QS})

# Kayan: her test haftasinin basinda, o ana kadar TESLIM EDILMIS son 8 haftalik siparislerle
pool = pd.concat([cal.assign(**{f"r{name[q]}": raw_cal[q] for q in QS}),
                  test.assign(**{f"r{name[q]}": raw_test[q] for q in QS})], ignore_index=True)
week = test["order_ts"].dt.to_period("W").dt.start_time.to_numpy()
roll = {q: np.zeros(len(test)) for q in QS}
n_fallback = 0
for w in np.unique(week):
    w = pd.Timestamp(w)
    win = pool[(pool["delivery_ts"] >= w - pd.Timedelta(days=WINDOW_DAYS)) & (pool["delivery_ts"] < w)]
    rows = week == w.to_datetime64()
    for q in QS:
        if len(win) >= MIN_N:
            off = offset(win["delay_days"].to_numpy(float) - win[f"r{name[q]}"].to_numpy(float), q)
        else:
            off, n_fallback = static_off[q], n_fallback + 1
        roll[q][rows] = raw_test[q][rows] + off
roll_test = monotone(roll)
print(f"Kayan kalibrasyon: {len(np.unique(week))} hafta, yedek (statik) kullanilan: {n_fallback}")

# ---------------------------------------------------------------------------
# 3) KARSILASTIRMA
# ---------------------------------------------------------------------------
y = test["delay_days"].to_numpy(float)
erp = test["erp_lead_time_days"].to_numpy(float)
variants = [("Ham model", raw_test), ("Statik conformal", static_test), ("Kayan conformal (8 hafta)", roll_test)]
res = {"fit_n": int(len(fit)), "cal_n": int(len(cal)), "test_n": int(len(test)), "variants": {}}

print("\n=== 1) KALIBRASYON: hedef vs gerceklesen coverage (%) ===")
print(f"{'':28s} {'P50':>6s} {'P85':>6s} {'P95':>6s}   (pazaryeri sozu: %{(y <= 0).mean() * 100:.1f})")
for vname, d in variants:
    cov = {k: float((y <= d[q]).mean() * 100) for k, q in [("P50", 0.5), ("P85", 0.85), ("P95", 0.95)]}
    print(f"{vname:28s} {cov['P50']:6.1f} {cov['P85']:6.1f} {cov['P95']:6.1f}")
    res["variants"][vname] = {"coverage": {k: round(v, 1) for k, v in cov.items()},
                              "mae_p50": round(float(np.abs(y - d[0.5]).mean()), 2)}

print("\n=== 2) P50 ile nokta tahmini (MAE, gun) ===")
for vname, d in variants:
    print(f"{vname:28s} MAE={np.abs(y - d[0.5]).mean():.2f}")

print("\n=== 3) AYLARA GORE P85 coverage (hedef 85) - dalgalanma azaldi mi? ===")
months = test["order_ts"].dt.to_period("M").astype(str).to_numpy()
tab = pd.DataFrame({"siparis": pd.Series(months).value_counts().sort_index()})
for vname, d in variants:
    tab[vname] = pd.Series((y <= d[0.85]).astype(float), index=range(len(y))).groupby(months).mean().mul(100)
tab["Pazaryeri sozu tuttu"] = pd.Series((y <= 0).astype(float)).groupby(months).mean().mul(100)
print(tab.round(1).to_string())
res["monthly_p85"] = {v: [round(float(x), 1) for x in tab[v].values] for v in tab.columns[1:]}
res["months"] = [str(m) for m in tab.index]
for vname, _ in variants:
    dev = np.abs(tab[vname].values - 85)
    res["variants"][vname]["monthly_p85_mean_abs_dev"] = round(float(dev.mean()), 1)
    res["variants"][vname]["monthly_p85_min"] = round(float(tab[vname].min()), 1)
    res["variants"][vname]["monthly_p85_max"] = round(float(tab[vname].max()), 1)
    print(f"{vname:28s} aylik sapma (ort. |cov-85|): {dev.mean():.1f} puan | en dusuk %{tab[vname].min():.0f} | en yuksek %{tab[vname].max():.0f}")

print("\n=== 4) SATICI BAZINDA P85 coverage (testte >=50 siparisi olan saticilar) ===")
sid = test["supplier_id"].astype(int).to_numpy()
for vname, d in variants:
    s = pd.Series((y <= d[0.85]).astype(float)).groupby(sid).agg(["mean", "size"])
    s = s[s["size"] >= 50]["mean"] * 100
    print(f"{vname:28s} {len(s)} satici | %10: %{s.quantile(.1):.0f} | medyan %{s.median():.0f} | %90: %{s.quantile(.9):.0f}")

print("\n=== 5) SOZ UZUNLUGU vs GUVENILIRLIK (ortalama soz, gun) ===")
plat_cov, plat_days = float((y <= 0).mean() * 100), float(erp.mean())
print(f"{'Pazaryeri sozu':28s} coverage %{plat_cov:.1f} | ortalama soz {plat_days:.1f} gun")
res["platform"] = {"coverage": round(plat_cov, 1), "avg_days": round(plat_days, 1)}
res["promise"] = {}
for vname, d in variants:
    for q, lab in [(0.85, "P85"), (0.95, "P95"), (q_match, f"P{q_match * 100:.0f} (pazaryeri tutma oranina esit hedef)")]:
        cov = float((y <= d[q]).mean() * 100)
        days = float((erp + d[q]).mean())
        res["promise"][f"{vname} | {lab}"] = {"coverage": round(cov, 1), "avg_days": round(days, 1)}
        print(f"{vname + ' ' + lab:70s} coverage %{cov:5.1f} | ortalama soz {days:5.1f} gun")

print("\n=== 6) PINBALL LOSS (dusuk iyi) ===")
for q in [0.85, 0.95]:
    print(f"P{int(q * 100)}: " + " | ".join(f"{v}={pinball(y, d[q], q):.2f}" for v, d in variants))

update_results("calibration", res)
out = test[["po_id", "order_ts", "erp_lead_time_days", "actual_lead_time_days", "delay_days"]].copy()
out["order_ts"] = out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
for tag, d in [("raw", raw_test), ("static", static_test), ("roll", roll_test)]:
    for q in QS:
        out[f"{tag}_{name[q]}"] = d[q]
con = sqlite3.connect(DB)
out.to_sql("predictions_calibrated", con, if_exists="replace", index=False)
con.commit(); con.close()
print("\n'predictions_calibrated' tablosu ve results.json guncellendi.")
