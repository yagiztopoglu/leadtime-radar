"""
LeadTime Radar - GERCEK VERI: README grafikleri (Olist)

Calistirma:  python 04_olist_charts.py
Gereken:     02 ve 03 numarali scriptler calismis olmali (results.json)
Cikti:       ../docs/real_*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OUT = HERE.parent / "docs"
OUT.mkdir(exist_ok=True)
R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
base, cal = R["baseline_model"], R["calibration"]

BLUE, ORANGE, GRAY, INK, MUTED = "#2F6DB5", "#C8581A", "#6B7684", "#1F2933", "#8A94A0"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#C5CBD3", "axes.labelcolor": "#3E4A56", "xtick.color": "#3E4A56",
                     "ytick.color": "#3E4A56", "axes.grid": True, "grid.color": "#E6E9ED", "grid.linewidth": 0.8,
                     "axes.axisbelow": True})
V_RAW, V_STAT, V_ROLL = "Ham model", "Statik conformal", "Kayan conformal (8 hafta)"
COL = {V_RAW: GRAY, V_STAT: ORANGE, V_ROLL: BLUE}
LAB = {V_RAW: "Ham model (kalibrasyonsuz)", V_STAT: "Statik kalibrasyon", V_ROLL: "Kayan kalibrasyon (8 hafta)"}

# ---------------------------------------------------------------- 1) coverage: hedef vs gerceklesen
fig, ax = plt.subplots(figsize=(7.5, 4.4))
targets = {"P50": 50, "P85": 85, "P95": 95}
x = np.arange(3)
w = 0.24
for i, v in enumerate([V_RAW, V_STAT, V_ROLL]):
    vals = [cal["variants"][v]["coverage"][k] for k in targets]
    bars = ax.bar(x + (i - 1) * (w + 0.03), vals, w, color=COL[v], label=LAB[v])
    if v == V_ROLL:
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + 1.2, f"%{val:.0f}", ha="center", fontsize=10,
                    color=INK, fontweight="bold")
for xi, t in zip(x, targets.values()):
    ax.plot([xi - 0.42, xi + 0.42], [t, t], color=INK, linewidth=1.6, linestyle=(0, (4, 2)),
            label="Hedef" if xi == 0 else None)
ax.set_xticks(x)
ax.set_xticklabels(list(targets))
ax.set_ylim(0, 112)
ax.set_ylabel("Siparişlerin tahmin içinde gelme oranı (%)")
ax.set_title("Kalibrasyon düzeltmesi (Olist, test: Nisan-Ağustos 2018)", loc="left", fontsize=12)
ax.legend(frameon=False, loc="upper left", fontsize=9, ncol=2)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(OUT / "real_1_kalibrasyon.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- 2) aylik P85 coverage
fig, ax = plt.subplots(figsize=(7.5, 4.4))
months = R["calibration"]["months"]
xs = np.arange(len(months))
ax.axhline(85, color=INK, linewidth=1.4, linestyle=(0, (4, 2)))
ax.text(len(months) - 0.55, 85.8, "Hedef %85", ha="right", va="bottom", fontsize=9, color=INK)
for v in [V_RAW, V_STAT, V_ROLL]:
    ys = cal["monthly_p85"][v]
    ax.plot(xs, ys, color=COL[v], linewidth=2, marker="o", markersize=7, markerfacecolor=COL[v],
            markeredgecolor="white", markeredgewidth=1.5, label=LAB[v], zorder=3)
ax.set_xticks(xs)
ax.set_xticklabels([m[5:] + "/" + m[2:4] for m in months])
ax.set_ylim(55, 102)
ax.set_ylabel("P85 coverage (%)")
ax.set_xlabel("Sipariş ayı")
ax.set_title("Aylık P85 coverage: ortalama düzeliyor, ay şokları düzelmiyor", loc="left", fontsize=12)
ax.legend(frameon=False, loc="lower left", fontsize=9)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(OUT / "real_2_aylik_p85.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- 3) MAE
fig, ax = plt.subplots(figsize=(7.5, 4.0))
items = [("Pazaryeri sözü\n(şişirilmiş)", base["mae"]["A"], MUTED),
         ("Söz + satıcı\n90g ort. sapma", base["mae"]["B"], MUTED),
         ("Satıcı bazlı\nstatik termin", base["mae"]["C"], MUTED),
         ("LeadTime Radar\n(P50)", base["mae"]["D"], BLUE)]
bars = ax.bar([i[0] for i in items], [i[1] for i in items], color=[i[2] for i in items], width=0.55)
for b, it in zip(bars, items):
    ax.text(b.get_x() + b.get_width() / 2, it[1] + 0.25, f"{it[1]:.1f}", ha="center", fontweight="bold", color=INK)
ax.set_ylabel("Ortalama mutlak hata (gün)")
ax.set_ylim(0, max(i[1] for i in items) * 1.18)
ax.set_title("Tahmin hatası, gerçek veri (düşük = iyi)", loc="left", fontsize=12)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(OUT / "real_3_mae.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- 4) soz uzunlugu vs guvenilirlik
fig, ax = plt.subplots(figsize=(7.5, 4.4))
prom = cal["promise"]
for v in [V_RAW, V_ROLL]:
    pts = sorted([(d["avg_days"], d["coverage"]) for k, d in prom.items() if k.startswith(v + " |")])
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COL[v], linewidth=2, marker="o", markersize=8,
            markeredgecolor="white", markeredgewidth=1.5, label=LAB[v], zorder=3)
p = cal["platform"]
ax.scatter([p["avg_days"]], [p["coverage"]], marker="D", s=90, color=INK, zorder=4, label="Pazaryeri sözü")
ax.annotate("Pazaryeri sözü", (p["avg_days"], p["coverage"]), xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=9, color=INK)
ax.set_xlabel("Ortalama verilen söz (gün)")
ax.set_ylabel("Sözün tutma oranı (%)")
ax.set_title("Aynı güvenilirlikte daha kısa söz yok; ama seçenek var", loc="left", fontsize=12)
ax.legend(frameon=False, loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "real_4_soz_vs_guvenilirlik.png", dpi=160)
plt.close(fig)
print("Grafikler:", OUT)
for f in sorted(OUT.glob("real_*.png")):
    print(" -", f.name)
