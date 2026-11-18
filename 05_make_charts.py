"""
LeadTime Radar - Grafikler (README icin)

Calistirma:  python 05_make_charts.py
Cikti:       docs/ klasorune 3 PNG
Gereken:     01-04 numarali scriptler calismis olmali.
Not: Grafikler KENDI calistirmanin sonucundan uretilir, README'deki rakamlari
     bu grafiklerle ve terminal ciktisiyla birebir uyumlu tut.
"""
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "docs"
OUT.mkdir(exist_ok=True)

con = sqlite3.connect(HERE / "leadtime.db")
pred = pd.read_sql("SELECT * FROM predictions_test", con)
feat = pd.read_sql("SELECT po_id, supplier_avg_delay_90d FROM features", con)
risk = pd.read_sql("SELECT * FROM open_orders_risk", con)
con.close()
pred = pred.merge(feat, on="po_id")
y = pred["delay_days"]

BLUE, ORANGE, GRAY, GREEN, RED = "#2F6DB5", "#E8833A", "#9AA3AD", "#3E9B5F", "#C94C4C"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------- 1) MAE karsilastirma
mae = {
    "ERP sabit termin": np.abs(y - 0).mean(),
    "ERP + tedarikçi\n90g ortalama": np.abs(y - pred["supplier_avg_delay_90d"]).mean(),
    "LeadTime Radar\n(P50)": np.abs(y - pred["p50"]).mean(),
}
fig, ax = plt.subplots(figsize=(7, 4.2))
bars = ax.bar(mae.keys(), mae.values(), color=[GRAY, ORANGE, BLUE], width=0.55)
for b, v in zip(bars, mae.values()):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center", fontweight="bold")
ax.set_ylabel("Ortalama mutlak hata (gün)")
ax.set_title("Teslimat süresi tahmin hatası (test dönemi, düşük = iyi)")
ax.set_ylim(0, max(mae.values()) * 1.2)
fig.tight_layout()
fig.savefig(OUT / "1_mae_karsilastirma.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- 2) Kalibrasyon (coverage)
targets = [50, 85, 95]
actual_cov = [(y <= pred[f"p{t}"]).mean() * 100 for t in targets]
erp_cov = (y <= 0).mean() * 100
x = np.arange(len(targets))
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(x - 0.18, targets, 0.36, color=GRAY, label="Hedef")
b2 = ax.bar(x + 0.18, actual_cov, 0.36, color=BLUE, label="Gerçekleşen")
for b, v in zip(b2, actual_cov):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"%{v:.1f}", ha="center", fontweight="bold")
ax.axhline(erp_cov, color=RED, linestyle="--", linewidth=1.5,
           label=f"ERP termini ile gelme oranı: %{erp_cov:.0f}")
ax.set_xticks(x)
ax.set_xticklabels([f"P{t}" for t in targets])
ax.set_ylabel("Siparişlerin tahmin içinde gelme oranı (%)")
ax.set_ylim(0, 112)
ax.set_title("Kalibrasyon: \"%85 ihtimalle\" dediğinde gerçekten %85 mi?")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "2_kalibrasyon.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- 3) Risk backtest
order = ["Dusuk", "Orta", "Yuksek"]
labels = {"Dusuk": "Düşük", "Orta": "Orta", "Yuksek": "Yüksek"}
g = risk.groupby("risk").agg(n=("po_id", "count"), pct=("actual_stockout", lambda s: s.mean() * 100)).reindex(order)
fig, ax = plt.subplots(figsize=(7, 4.2))
bars = ax.bar([f"{labels[k]}\n({int(g.loc[k, 'n'])} sipariş)" for k in order], g["pct"],
              color=[GREEN, ORANGE, RED], width=0.5)
for b, v in zip(bars, g["pct"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"%{v:.0f}", ha="center", fontweight="bold")
ax.set_ylabel("Gerçekte stoksuz kalan sipariş (%)")
ax.set_ylim(0, 100)
ax.set_title("Risk sınıfı vs gerçekleşen stoksuzluk (açık siparişler)")
fig.tight_layout()
fig.savefig(OUT / "3_risk_dogrulama.png", dpi=160)
plt.close(fig)

# Terminal ozeti: README'ye yazacagin rakamlar
print("MAE:", {k.replace("\n", " "): round(v, 2) for k, v in mae.items()})
print("Coverage:", dict(zip([f"P{t}" for t in targets], [round(v, 1) for v in actual_cov])), "| ERP:", round(erp_cov, 1))
print(g.round(1).to_string())
print(f"Grafikler: {OUT}")
