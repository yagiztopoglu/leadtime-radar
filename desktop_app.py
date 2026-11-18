"""
LeadTime Radar - MASAUSTU uygulamasi (CustomTkinter)

Calistirma:  python desktop_app.py
Gereken:     pip install customtkinter   (matplotlib, pandas, pillow zaten kurulu olmali)
             01, 02, 03, 04 numarali scriptler calismis olmali (leadtime.db ve models/ uretilmis olmali).

Arayuz: CustomTkinter masaustu uygulamasi. Model, veritabani ve risk motoru scriptlerle aynidir.
"""
import io
import json
import sqlite3
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image

from report_generator import generate_report
from scoring import make_template, score_open_orders, validate_input

HERE = Path(__file__).parent
DB_PATH = HERE / "leadtime.db"
REAL_DIR = HERE / "real_data"
DOCS_DIR = HERE / "docs"

RISK_LABEL = {"Yuksek": "Yüksek", "Orta": "Orta", "Dusuk": "Düşük"}
RISK_ORDER = ["Yüksek", "Orta", "Düşük"]

# ----- renkler (koyu tema)
BG, PANEL, CARD, INK, MUTED = "#0F172A", "#111C33", "#1B2A47", "#E6EDF7", "#8EA0BA"
ACCENT = "#3B82F6"
ROW_BG = {"Yüksek": "#5B2230", "Orta": "#5A4A1B", "Düşük": "#1E4733"}
BAR_COL = ["#8EA0BA", "#3B82F6", "#F59E0B", "#EF4444"]


# =============================================================== VERI (arayuzden bagimsiz)
def load_data():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM open_orders_risk", con)
    sup = pd.read_sql("SELECT supplier_id, name AS supplier_name, country FROM suppliers", con)
    as_of = pd.read_sql("SELECT as_of FROM inventory LIMIT 1", con)["as_of"].iloc[0]
    con.close()
    df = df.merge(sup, on="supplier_id", how="left")
    df["risk_tr"] = df["risk"].map(RISK_LABEL)
    return df, as_of


def orders_table(df: pd.DataFrame) -> pd.DataFrame:
    """Siparisleri riske gore sirala ve gosterilecek sutunlara cevir."""
    t = df.assign(_o=df["risk_tr"].map({r: i for i, r in enumerate(RISK_ORDER)}))
    t = t.sort_values(["_o", "shortage_days_p85"], ascending=[True, False])
    return pd.DataFrame({
        "PO": t["po_id"], "Tedarikçi": t["supplier_name"], "Parça": t["name"],
        "ERP'ye göre (gün)": t["erp_arrival_days"].round(0).astype(int),
        "Modele göre (gün)": t["p85_arrival_days"].round(0).astype(int),
        "Stok yeter (gün)": t["days_to_zero"].round(0).astype(int),
        "Stoksuz kalma (gün)": t["shortage_days_p85"].round(0).astype(int),
        "Risk": t["risk_tr"],
    })


def supplier_table(data: pd.DataFrame) -> pd.DataFrame:
    g = data.groupby("supplier_name").agg(
        acik=("po_id", "count"),
        p85=("p85_delay", "mean"),
        yuksek=("risk", lambda s: int((s == "Yuksek").sum())),
        ek=("extra_units_needed", "sum"),
    ).round(1).sort_values("p85", ascending=False).reset_index()
    g.columns = ["Tedarikçi", "Açık sipariş", "Ort. P85 gecikme (gün)", "Yüksek riskli", "Toplam ek ihtiyaç (adet)"]
    return g


def backtest(data: pd.DataFrame):
    bt = data.groupby("risk_tr").agg(
        siparis=("po_id", "count"),
        stoksuz=("actual_stockout", lambda s: round(s.mean() * 100, 1)),
    ).reindex(RISK_ORDER).reset_index()
    bt.columns = ["Risk sınıfı", "Sipariş", "Gerçekte stoksuz (%)"]
    n_real = int(data["actual_stockout"].sum())
    erp_flag = data["shortage_days_erp"] > 0
    model_flag = data["risk"].isin(["Yuksek", "Orta"])
    caught_erp = int((erp_flag & (data["actual_stockout"] == 1)).sum())
    caught_model = int((model_flag & (data["actual_stockout"] == 1)).sum())
    text = (f"Gerçekte stoksuz kalan: {n_real} sipariş.\n"
            f"ERP termini bunların {caught_erp} tanesini ({int(erp_flag.sum())} siparişi işaretleyerek),\n"
            f"LeadTime Radar {caught_model} tanesini ({int(model_flag.sum())} siparişi işaretleyerek) yakaladı.\n\n"
            "Not: Radar daha fazla vakayı yakalar ama daha fazla yanlış alarm da üretir; "
            "önceliklendirme için önce 'Yüksek' sınıfına bakılır.")
    return bt, text


def detail_summary(r) -> str:
    if r.shortage_days_p85 > 0:
        return (f"{r.supplier_name} tedarikçisinin {r['name']} parçasında ERP {r.erp_arrival_days:.0f} gün diyor, "
                f"ancak model %85 ihtimalle {r.p85_arrival_days:.0f} gün içinde geleceğini öngörüyor. "
                f"Stok {r.days_to_zero:.0f} gün yetiyor; yaklaşık {r.shortage_days_p85:.0f} gün stoksuz kalma "
                f"riski var (~{r.extra_units_needed:.0f} adet).")
    return (f"{r['name']} için model %85 ihtimalle {r.p85_arrival_days:.0f} gün öngörüyor, stok "
            f"{r.days_to_zero:.0f} gün yetiyor. Güvenlik stoğu korunuyor, acil aksiyon gerekmiyor.")


# =============================================================== KUCUK ARAYUZ PARCALARI
class Kpi(ctk.CTkFrame):
    def __init__(self, parent, title, value="-", color=INK):
        super().__init__(parent, fg_color=CARD, corner_radius=12)
        ctk.CTkLabel(self, text=title, text_color=MUTED, font=ctk.CTkFont(size=12)).pack(anchor="w", padx=14, pady=(10, 0))
        self.val = ctk.CTkLabel(self, text=str(value), text_color=color, font=ctk.CTkFont(size=26, weight="bold"))
        self.val.pack(anchor="w", padx=14, pady=(0, 10))

    def set(self, v):
        self.val.configure(text=str(v))


def setup_tree_style():
    s = ttk.Style()
    s.theme_use("clam")
    s.configure("Radar.Treeview", background=PANEL, fieldbackground=PANEL, foreground=INK,
                rowheight=26, borderwidth=0, font=("Segoe UI", 10))
    s.configure("Radar.Treeview.Heading", background=CARD, foreground=INK, relief="flat",
                font=("Segoe UI", 10, "bold"))
    s.map("Radar.Treeview", background=[("selected", ACCENT)], foreground=[("selected", "white")])
    s.map("Radar.Treeview.Heading", background=[("active", CARD)])


def make_tree(parent, columns, widths=None, height=10):
    wrap = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=10)
    tv = ttk.Treeview(wrap, columns=columns, show="headings", style="Radar.Treeview", height=height)
    sb = ttk.Scrollbar(wrap, orient="vertical", command=tv.yview)
    tv.configure(yscrollcommand=sb.set)
    for i, c in enumerate(columns):
        tv.heading(c, text=c)
        tv.column(c, width=(widths[i] if widths else 100), anchor="w" if i < 3 else "center", stretch=True)
    for k, col in ROW_BG.items():
        tv.tag_configure(k, background=col)
    tv.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=4)
    sb.pack(side="right", fill="y", pady=4)
    return wrap, tv


def fill_tree(tv, df, risk_col="Risk"):
    tv.delete(*tv.get_children())
    for _, row in df.iterrows():
        tag = row[risk_col] if risk_col in df.columns else ""
        tv.insert("", "end", values=[row[c] for c in df.columns], tags=(tag,))


def bar_figure(parent, labels, values, colors=None, title="", horizontal=False, height=2.6):
    fig = Figure(figsize=(5, height), dpi=100, facecolor=PANEL)
    ax = fig.add_subplot(111, facecolor=PANEL)
    cols = colors or [ACCENT] * len(values)
    bars = (ax.barh if horizontal else ax.bar)(labels, values, color=cols)
    for b, v in zip(bars, values):
        if horizontal:
            ax.text(b.get_width() + 0.1, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", color=INK, fontsize=8)
        else:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{v:.0f}", ha="center", color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#334155")
    ax.grid(axis="x" if horizontal else "y", color="#22304D", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    return canvas.get_tk_widget()


def textbox(parent, height=180):
    tb = ctk.CTkTextbox(parent, fg_color=PANEL, text_color=INK, corner_radius=10, height=height,
                        font=ctk.CTkFont(size=13), wrap="word")
    return tb


def set_text(tb, text):
    tb.configure(state="normal")
    tb.delete("1.0", "end")
    tb.insert("1.0", text)
    tb.configure(state="disabled")


# =============================================================== ANA UYGULAMA
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.title("LeadTime Radar")
        self.geometry("1360x860")
        self.minsize(1180, 740)
        self.configure(fg_color=BG)
        setup_tree_style()

        try:
            self.data, self.as_of = load_data()
        except Exception as e:
            messagebox.showerror("Veritabanı okunamadı",
                                 f"Önce 01-04 numaralı scriptleri sırayla çalıştır.\n\nAyrıntı: {e}")
            self.destroy()
            raise SystemExit(1)

        self.view = self.data
        self.shown_orders = None        # tabloda gosterilen siparisler (PO sirasi)
        self.up_scored = None
        self.detail_widgets = []

        self._build_sidebar()
        self._build_main()
        self.apply_filters()

    # ---------------------------------------------------------- yan panel
    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=250, fg_color=PANEL, corner_radius=0)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)
        ctk.CTkLabel(sb, text="📡 LeadTime Radar", font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=INK).pack(anchor="w", padx=18, pady=(20, 2))
        ctk.CTkLabel(sb, text=f"Rapor tarihi: {self.as_of}\nVeri sentetiktir (portföy projesi)",
                     text_color=MUTED, justify="left", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=18, pady=(0, 16))

        ctk.CTkLabel(sb, text="Risk seviyesi", text_color=INK, font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=18)
        self.risk_vars = {}
        for r in RISK_ORDER:
            v = ctk.BooleanVar(value=r in ("Yüksek", "Orta"))
            self.risk_vars[r] = v
            ctk.CTkCheckBox(sb, text=r, variable=v, command=self.apply_filters,
                            text_color=INK).pack(anchor="w", padx=22, pady=3)

        ctk.CTkLabel(sb, text="Tedarikçi", text_color=INK, font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=18, pady=(14, 2))
        self.sup_options = ["Tümü"] + sorted(self.data["supplier_name"].unique())
        self.sup_var = ctk.StringVar(value="Tümü")
        ctk.CTkOptionMenu(sb, values=self.sup_options, variable=self.sup_var,
                          command=lambda _: self.apply_filters()).pack(fill="x", padx=18)

        ctk.CTkLabel(sb, text="API anahtarı (opsiyonel)", text_color=INK,
                     font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=18, pady=(20, 2))
        self.key_entry = ctk.CTkEntry(sb, show="•", placeholder_text="boşsa şablon rapor")
        self.key_entry.pack(fill="x", padx=18)
        ctk.CTkLabel(sb, text="Boş bırakırsan rapor kural tabanlı\nşablonla üretilir.", text_color=MUTED,
                     justify="left", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=18, pady=(4, 0))

    def api_key(self):
        return self.key_entry.get().strip() or None

    # ---------------------------------------------------------- ana alan
    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color=BG)
        main.pack(side="left", fill="both", expand=True, padx=16, pady=14)

        kp = ctk.CTkFrame(main, fg_color=BG)
        kp.pack(fill="x")
        hidden = self.data[(self.data["shortage_days_erp"] == 0) & (self.data["risk"].isin(["Yuksek", "Orta"]))]
        items = [("Açık sipariş", len(self.data), INK),
                 ("Yüksek risk", int((self.data["risk"] == "Yuksek").sum()), "#F87171"),
                 ("Orta risk", int((self.data["risk"] == "Orta").sum()), "#FBBF24"),
                 ("ERP'nin göremediği riskli sipariş", len(hidden), "#60A5FA")]
        for i, (t, v, c) in enumerate(items):
            kp.grid_columnconfigure(i, weight=1, uniform="k")
            Kpi(kp, t, v, c).grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 8, 0))

        self.tabs = ctk.CTkTabview(main, fg_color=PANEL, segmented_button_selected_color=ACCENT)
        self.tabs.pack(fill="both", expand=True, pady=(12, 0))
        names = ["Siparişler", "Doğrulama", "Kendi verin (CSV)"]
        for n in names:
            self.tabs.add(n)
        self._tab_orders(self.tabs.tab(names[0]))
        self._tab_validation(self.tabs.tab(names[1]))
        self._tab_csv(self.tabs.tab(names[2]))

    # ---------------------------------------------------------- TAB 1
    def _tab_orders(self, tab):
        cols = ["PO", "Tedarikçi", "Parça", "ERP'ye göre (gün)", "Modele göre (gün)", "Stok yeter (gün)", "Stoksuz kalma (gün)", "Risk"]
        w = [60, 150, 170, 120, 120, 110, 130, 80]
        wrap, self.tv_orders = make_tree(tab, cols, w, height=9)
        wrap.pack(fill="x", padx=6, pady=(6, 4))
        self.tv_orders.bind("<<TreeviewSelect>>", self.on_select_order)
        bar = ctk.CTkFrame(tab, fg_color=PANEL)
        bar.pack(fill="x", padx=6)
        ctk.CTkButton(bar, text="Tabloyu CSV olarak kaydet", width=190, command=self.save_orders_csv).pack(side="left")
        self.lbl_count = ctk.CTkLabel(bar, text="", text_color=MUTED)
        self.lbl_count.pack(side="left", padx=12)

        self.detail = ctk.CTkFrame(tab, fg_color=PANEL)
        self.detail.pack(fill="both", expand=True, padx=6, pady=(8, 4))
        ctk.CTkLabel(self.detail, text="Detay için tablodan bir sipariş seç.", text_color=MUTED).pack(pady=30)

    def apply_filters(self):
        risks = [r for r, v in self.risk_vars.items() if v.get()]
        d = self.data[self.data["risk_tr"].isin(risks)]
        if self.sup_var.get() != "Tümü":
            d = d[d["supplier_name"] == self.sup_var.get()]
        self.view = d
        self.shown_orders = orders_table(d)
        fill_tree(self.tv_orders, self.shown_orders)
        self.lbl_count.configure(text=f"{len(d)} sipariş gösteriliyor")

    def save_orders_csv(self):
        p = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="leadtime_radar_acik_siparisler.csv")
        if p:
            self.shown_orders.to_csv(p, index=False, encoding="utf-8-sig", sep=";")

    def on_select_order(self, _=None):
        sel = self.tv_orders.selection()
        if not sel:
            return
        po = self.tv_orders.item(sel[0])["values"][0]
        r = self.data[self.data["po_id"] == po].iloc[0]
        for w in self.detail.winfo_children():
            w.destroy()
        top = ctk.CTkFrame(self.detail, fg_color=PANEL)
        top.pack(fill="x", padx=8, pady=8)
        for i, (t, v) in enumerate([("ERP'ye göre gelir", f"{r.erp_arrival_days:.0f} gün"),
                                    ("Modele göre gelir (%85)", f"{r.p85_arrival_days:.0f} gün"),
                                    ("Stok yeter", f"{r.days_to_zero:.0f} gün")]):
            top.grid_columnconfigure(i, weight=1, uniform="d")
            Kpi(top, t, v).grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 6, 0))
        body = ctk.CTkFrame(self.detail, fg_color=PANEL)
        body.pack(fill="both", expand=True, padx=8)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=2)
        bar_figure(body, ["ERP'ye göre", "Modele göre", "Stok biter"],
                   [r.erp_arrival_days, r.p85_arrival_days, r.days_to_zero],
                   ["#8EA0BA", "#F59E0B", "#EF4444"], f"PO {po} (gün)").grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right = ctk.CTkFrame(body, fg_color=PANEL)
        right.grid(row=0, column=1, sticky="nsew")
        ctk.CTkLabel(right, text=detail_summary(r), wraplength=560, justify="left", text_color=INK,
                     font=ctk.CTkFont(size=13)).pack(anchor="w", pady=(2, 8))
        self.rep_box = textbox(right, 140)
        ctk.CTkButton(right, text="📝 Aksiyon raporu üret", width=190,
                      command=lambda: self.run_async(lambda: generate_report(r, api_key=self.api_key()),
                                                     self.rep_box)).pack(anchor="w")
        self.rep_box.pack(fill="both", expand=True, pady=(8, 0))
        set_text(self.rep_box, "")

    def run_async(self, fn, box):
        """Rapor uretimini arka planda calistir (pencere donmasin)."""
        set_text(box, "Rapor hazırlanıyor...")

        def work():
            try:
                text, src = fn()
                out = f"{text}\n\n— Kaynak: {src}. Tüm sayılar Python tarafından hesaplanır; LLM yalnızca metne döker."
            except Exception as e:
                out = f"Rapor üretilemedi: {e}"
            self.after(0, lambda: set_text(box, out))
        threading.Thread(target=work, daemon=True).start()

    # ---------------------------------------------------------- DOGRULAMA
    def _tab_validation(self, tab):
        sc = ctk.CTkScrollableFrame(tab, fg_color=PANEL)
        sc.pack(fill="both", expand=True)
        H = ctk.CTkFont(size=16, weight="bold")

        ctk.CTkLabel(sc, text="1) Sahte (simülasyon) veride test", font=H, text_color=INK).pack(anchor="w", padx=8, pady=(8, 2))
        ctk.CTkLabel(sc, wraplength=900, justify="left", text_color=MUTED,
                     text="Gerçek varış tarihlerini bildiğimiz için riskli dediğimiz siparişlerin gerçekten stoksuz "
                          "kalıp kalmadığına bakabiliriz.").pack(anchor="w", padx=8)
        bt, text = backtest(self.data)
        wrap, tv = make_tree(sc, list(bt.columns), [160, 100, 180], height=3)
        wrap.pack(fill="x", padx=8, pady=6)
        for _, r in bt.iterrows():
            tv.insert("", "end", values=list(r), tags=(r["Risk sınıfı"],))
        n_real = int(self.data["actual_stockout"].sum())
        erp_flag = self.data["shortage_days_erp"] > 0
        model_flag = self.data["risk"].isin(["Yuksek", "Orta"])
        c_erp = int((erp_flag & (self.data["actual_stockout"] == 1)).sum())
        c_mod = int((model_flag & (self.data["actual_stockout"] == 1)).sum())
        ctk.CTkLabel(sc, wraplength=900, justify="left", text_color=INK, font=ctk.CTkFont(size=14),
                     text=f"Gerçekte stoksuz kalan {n_real} siparişten ERP'nin sabit süresi {c_erp} tanesini, "
                          f"LeadTime Radar {c_mod} tanesini önceden yakaladı."
                     ).pack(anchor="w", padx=8, pady=(2, 14))

        ctk.CTkLabel(sc, text="2) Gerçek veride test (Olist e-ticaret)", font=H, text_color=INK).pack(anchor="w", padx=8, pady=(6, 2))
        res = REAL_DIR / "results.json"
        if not res.exists():
            ctk.CTkLabel(sc, text="Henüz üretilmedi: real_data klasöründeki scriptleri çalıştır.", text_color=MUTED
                         ).pack(anchor="w", padx=8)
            return
        R = json.loads(res.read_text(encoding="utf-8"))
        b, c = R["baseline_model"], R["calibration"]
        roll = c["variants"]["Kayan conformal (8 hafta)"]["coverage"]
        ctk.CTkLabel(sc, wraplength=900, justify="left", text_color=MUTED,
                     text="Brezilya'daki bir pazaryerinin yaklaşık 56 bin gerçek siparişi (201 satıcı). Tahmin ortalama "
                          "kaç gün şaşıyor? Küçük olan iyi.").pack(anchor="w", padx=8)
        row = ctk.CTkFrame(sc, fg_color=PANEL)
        row.pack(fill="x", padx=8, pady=8)
        for i, (t, v) in enumerate([("Pazaryerinin sözü", f"{b['mae']['A']:.1f} gün hata"),
                                    ("Basit satıcı ortalaması", f"{b['mae']['C']:.1f} gün hata"),
                                    ("LeadTime Radar", f"{b['mae']['D']:.1f} gün hata"),
                                    ("%85 aralığı gerçekte tuttu", f"%{roll['P85']:.0f}")]):
            row.grid_columnconfigure(i, weight=1, uniform="r")
            Kpi(row, t, v).grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 6, 0))
        ctk.CTkLabel(sc, wraplength=900, justify="left", text_color="#FBBF24",
                     text="Not: Bu veride pazaryerinin sözü zaten fazla temkinli (siparişler ortalama 12 gün erken geliyor). "
                          "Asıl rakip basit satıcı ortalaması; ona karşı kazancımız küçük ama aralıklar doğru çıktı."
                     ).pack(anchor="w", padx=8, pady=(0, 8))

    # ---------------------------------------------------------- TAB 5
    def _tab_csv(self, tab):
        ctk.CTkLabel(tab, wraplength=950, justify="left", text_color=MUTED,
                     text="ERP'den aldığın açık sipariş listesini CSV olarak yükle; araç her sipariş için varış "
                          "tahmini ve stok riskini hesaplasın. Model bu projedeki (sentetik) tedarikçi ve parça "
                          "kodlarıyla eğitildi; kendi şirket verin için önce yeniden eğitim gerekir (README)."
                     ).pack(anchor="w", padx=8, pady=(6, 8))
        bar = ctk.CTkFrame(tab, fg_color=PANEL)
        bar.pack(fill="x", padx=6)
        ctk.CTkButton(bar, text="1) Şablon CSV'yi kaydet", width=190, command=self.save_template).pack(side="left")
        ctk.CTkLabel(bar, text="Rapor tarihi:", text_color=INK).pack(side="left", padx=(18, 4))
        self.asof_entry = ctk.CTkEntry(bar, width=110)
        self.asof_entry.insert(0, str(self.as_of)[:10])
        self.asof_entry.pack(side="left")
        ctk.CTkButton(bar, text="2) CSV seç ve puanla", width=190, command=self.load_csv).pack(side="left", padx=12)
        self.csv_msg = ctk.CTkLabel(tab, text="", text_color="#FBBF24", wraplength=950, justify="left")
        self.csv_msg.pack(anchor="w", padx=8, pady=4)

        cols = ["PO", "Tedarikçi", "Parça", "ERP'ye göre (gün)", "Modele göre (gün)", "Stok yeter (gün)", "Stoksuz kalma (gün)", "Risk"]
        wrap, self.tv_up = make_tree(tab, cols, [60, 150, 170, 120, 120, 110, 130, 80], height=8)
        wrap.pack(fill="x", padx=6)
        self.tv_up.bind("<<TreeviewSelect>>", self.on_select_upload)
        self.up_rep = textbox(tab, 120)
        self.up_btn = ctk.CTkButton(tab, text="📝 Seçili sipariş için rapor üret", width=230, command=self.up_report)
        self.up_btn.pack(anchor="w", padx=8, pady=(8, 0))
        self.up_rep.pack(fill="both", expand=True, padx=6, pady=8)
        self.up_sel = None

    def save_template(self):
        p = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="acik_siparis_sablonu.csv")
        if p:
            make_template().to_csv(p, index=False, encoding="utf-8-sig", sep=";")
            self.csv_msg.configure(text=f"Şablon kaydedildi: {p}")

    def load_csv(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not p:
            return
        try:
            raw = pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig")
            ts = pd.Timestamp(self.asof_entry.get().strip())
        except Exception as e:
            self.csv_msg.configure(text=f"Dosya okunamadı veya tarih geçersiz: {e}")
            return
        clean, errs, warns = validate_input(raw, ts)
        msgs = list(errs) + list(warns)
        if clean is None:
            self.csv_msg.configure(text="\n".join(msgs))
            return
        scored = score_open_orders(clean, ts)
        scored = scored.assign(risk_tr=scored["risk"].map(RISK_LABEL))
        self.up_scored = scored
        n = scored["risk"].value_counts()
        head = (f"Puanlanan sipariş: {len(scored)} · Yüksek: {int(n.get('Yuksek', 0))} · "
                f"Orta: {int(n.get('Orta', 0))} · Düşük: {int(n.get('Dusuk', 0))}")
        self.csv_msg.configure(text=head + ("\n" + "\n".join(msgs) if msgs else ""))
        fill_tree(self.tv_up, orders_table(scored))
        set_text(self.up_rep, "")

    def on_select_upload(self, _=None):
        sel = self.tv_up.selection()
        self.up_sel = self.tv_up.item(sel[0])["values"][0] if sel else None

    def up_report(self):
        if self.up_scored is None or self.up_sel is None:
            self.csv_msg.configure(text="Önce CSV yükle ve tablodan bir sipariş seç.")
            return
        rr = self.up_scored[self.up_scored["po_id"] == self.up_sel].iloc[0]
        self.run_async(lambda: generate_report(rr, api_key=self.api_key()), self.up_rep)


if __name__ == "__main__":
    App().mainloop()
