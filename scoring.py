"""
LeadTime Radar - Puanlama modulu (arayuzdeki CSV yukleme bunu kullanir)

Ne yapar?
  Bir "acik siparis" tablosu (ERP'den disari aktarilmis gibi: siparis, tedarikci, parca, tarih, miktar,
  ERP termini, eldeki stok, gunluk tuketim) alir ve her siparis icin:
    1) ozellikleri (tedarikci gecmisi vb.) leadtime.db'deki GECMIS verilerden uretir,
    2) egitilmis P50/P85/P95 modellerini calistirir,
    3) 04_risk_engine.py ile AYNI kurallarla Dusuk/Orta/Yuksek riski hesaplar.

Onemli: Ozellikler 02_features_baseline.py'deki SQL ile ayni tanimlarla uretilir
(sadece siparis tarihinden ONCE bilinebilen bilgi). Dogrulama: ayni siparisleri bu modulden
gecirince 04'un tablosuyla ayni sonucu verir (README'de test edildi).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sqlite3

HERE = Path(__file__).parent
DB_PATH = HERE / "leadtime.db"
MODEL_DIR = HERE / "models"
SAFETY_DAYS = 4

REQUIRED = ["po_id", "supplier_id", "material_id", "order_date", "quantity",
            "stock_on_hand", "daily_consumption"]
OPTIONAL = ["erp_lead_time_days", "safety_stock"]      # yoksa malzeme tablosundan / 4 gunluk tuketimden
FEATURES = ["erp_lead_time_days", "quantity", "qty_ratio", "order_month", "order_weekday",
            "supplier_orders_14d", "supplier_avg_delay_90d", "supplier_std_delay_90d",
            "supplier_n_90d", "supplier_id", "category", "country"]


def _num(series: pd.Series) -> pd.Series:
    """Sayiya cevir. Turkce Excel 'ondalik virgul' ile kaydeder (12,5): once noktaya cevir."""
    if not pd.api.types.is_numeric_dtype(series):          # metin (object veya string) ise
        series = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(series, errors="coerce")


def validate_input(raw: pd.DataFrame, as_of: pd.Timestamp, db_path=DB_PATH):
    """Girdiyi kontrol et. Donus: (temiz_df, hatalar[liste], uyarilar[liste])."""
    errors, warnings = [], []
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return None, [f"Eksik sütun(lar): {', '.join(missing)}"], warnings

    con = sqlite3.connect(db_path)
    sups = pd.read_sql("SELECT supplier_id, name AS supplier_name, country FROM suppliers", con)
    mats = pd.read_sql("SELECT material_id, name, category, typical_qty, erp_lead_time_days AS mat_erp "
                       "FROM materials", con)
    con.close()

    # Bizim hesapladigimiz/eslestirdigimiz sutunlar dosyada varsa karismasin
    df = df.drop(columns=[c for c in ["name", "supplier_name", "category", "country", "typical_qty"] if c in df.columns])
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce", dayfirst=False)
    for c in ["supplier_id", "material_id", "quantity", "stock_on_hand", "daily_consumption"]:
        df[c] = _num(df[c])
    bad = df[REQUIRED].isna().any(axis=1)
    if bad.any():
        warnings.append(f"{int(bad.sum())} satır eksik/okunamayan değer içerdiği için atlandı "
                        f"(PO: {', '.join(map(str, df.loc[bad, 'po_id'].head(5)))}...)")
        df = df[~bad]
    unknown_s = ~df["supplier_id"].isin(sups["supplier_id"])
    unknown_m = ~df["material_id"].isin(mats["material_id"])
    if (unknown_s | unknown_m).any():
        warnings.append(f"{int((unknown_s | unknown_m).sum())} satırda tanımsız tedarikçi/parça kodu var, "
                        "atlandı (model yalnızca eğitimde gördüğü tedarikçi ve parçalar için çalışır).")
        df = df[~(unknown_s | unknown_m)]
    future = df["order_date"] > as_of
    if future.any():
        warnings.append(f"{int(future.sum())} satırın sipariş tarihi rapor tarihinden sonra, atlandı.")
        df = df[~future]
    nonpos = (df["quantity"] <= 0) | (df["daily_consumption"] <= 0) | (df["stock_on_hand"] < 0)
    if nonpos.any():
        warnings.append(f"{int(nonpos.sum())} satırda miktar/tüketim/stok değeri geçersiz, atlandı.")
        df = df[~nonpos]
    if df["po_id"].duplicated().any():
        warnings.append("Tekrarlayan PO numaraları var, ilk kayıtlar tutuldu.")
        df = df.drop_duplicates("po_id")
    if len(df) == 0:
        return None, ["Geçerli satır kalmadı."] + warnings, []

    df = df.merge(sups, on="supplier_id", how="left").merge(
        mats.drop(columns=["mat_erp"]), on="material_id", how="left")
    if "erp_lead_time_days" not in df.columns or df["erp_lead_time_days"].isna().all():
        df["erp_lead_time_days"] = df["material_id"].map(mats.set_index("material_id")["mat_erp"])
    else:
        df["erp_lead_time_days"] = _num(df["erp_lead_time_days"]).fillna(
            df["material_id"].map(mats.set_index("material_id")["mat_erp"]))
    if "safety_stock" not in df.columns:
        df["safety_stock"] = df["daily_consumption"] * SAFETY_DAYS
    else:
        df["safety_stock"] = _num(df["safety_stock"]).fillna(
            df["daily_consumption"] * SAFETY_DAYS)
    return df.reset_index(drop=True), errors, warnings


def build_features(df: pd.DataFrame, db_path=DB_PATH) -> pd.DataFrame:
    """02_features_baseline.py'deki SQL ozelliklerinin pandas karsiligi."""
    con = sqlite3.connect(db_path)
    hist = pd.read_sql("SELECT po_id, supplier_id, order_date, actual_delivery_date, delay_days "
                       "FROM purchase_orders", con, parse_dates=["order_date", "actual_delivery_date"])
    cats = pd.read_sql("SELECT DISTINCT supplier_id, category, country FROM features", con)
    con.close()

    hist = hist[~hist["po_id"].isin(df["po_id"])]            # yuklenen siparisler "gecmis" sayilmaz
    g_mean, g_std = float(hist["delay_days"].mean()), float(hist["delay_days"].std())
    by_sup = {s: g for s, g in hist.groupby("supplier_id")}
    pool_dates = {s: np.sort(np.concatenate([g["order_date"].to_numpy(),
                                             df.loc[df["supplier_id"] == s, "order_date"].to_numpy()]))
                  for s, g in by_sup.items()}

    load, avg, std, n = [], [], [], []
    for _, r in df.iterrows():
        s, od = int(r["supplier_id"]), r["order_date"]
        pd_dates = pool_dates.get(s, np.array([], dtype="datetime64[ns]"))
        # son 14 gunde, siparisten STRICT once verilen siparisler (kendisi dahil degil)
        load.append(int(((pd_dates < np.datetime64(od)) & (pd_dates >= np.datetime64(od - pd.Timedelta(days=14)))).sum()))
        g = by_sup.get(s)
        if g is None:
            avg.append(g_mean); std.append(g_std); n.append(0); continue
        w = g[(g["actual_delivery_date"] < od) & (g["actual_delivery_date"] >= od - pd.Timedelta(days=90))]["delay_days"]
        if len(w) == 0:
            avg.append(g_mean); std.append(g_std); n.append(0)
        else:
            m = float(w.mean())
            avg.append(m); std.append(float(np.sqrt(max((w ** 2).mean() - m ** 2, 0)))); n.append(len(w))

    out = df.copy()
    out["qty_ratio"] = out["quantity"] / out["typical_qty"]
    out["order_month"] = out["order_date"].dt.month
    out["order_weekday"] = (out["order_date"].dt.dayofweek + 1) % 7      # SQLite %w: Pazar=0 ... Cumartesi=6
    out["supplier_orders_14d"], out["supplier_avg_delay_90d"] = load, avg
    out["supplier_std_delay_90d"], out["supplier_n_90d"] = std, n
    # Kategorik tipler: egitimdeki ile ayni kategori listesi
    out["supplier_id"] = pd.Categorical(out["supplier_id"].astype(int), categories=sorted(cats["supplier_id"].unique()))
    out["category"] = pd.Categorical(out["category"], categories=sorted(cats["category"].unique()))
    out["country"] = pd.Categorical(out["country"], categories=sorted(cats["country"].unique()))
    return out


def score_open_orders(df: pd.DataFrame, as_of: pd.Timestamp, db_path=DB_PATH, model_dir=MODEL_DIR) -> pd.DataFrame:
    """Temiz (validate_input'tan gecmis) acik siparisleri puanla ve risk sinifini hesapla."""
    f = build_features(df, db_path)
    preds = [joblib.load(Path(model_dir) / f"quantile_{q}.joblib").predict(f[FEATURES]) for q in (50, 85, 95)]
    P = np.maximum.accumulate(np.column_stack(preds), axis=1)
    o = df.copy()
    o["supplier_id"] = o["supplier_id"].astype(int)
    o["p50_delay"], o["p85_delay"], o["p95_delay"] = P[:, 0], P[:, 1], P[:, 2]
    f_cols = ["supplier_orders_14d", "supplier_avg_delay_90d", "supplier_n_90d"]
    for c in f_cols:
        o[c] = f[c].to_numpy()

    elapsed = (as_of - o["order_date"]).dt.days

    def arrival(col):                                    # "bugunden kac gun sonra" (en erken yarin)
        return np.maximum(np.ceil(o["erp_lead_time_days"] + o[col] - elapsed), 1)

    o["erp_arrival_days"] = np.maximum(o["erp_lead_time_days"] - elapsed, 0)
    o["p50_arrival_days"], o["p85_arrival_days"] = arrival("p50_delay"), arrival("p85_delay")
    o["days_to_zero"] = o["stock_on_hand"] / o["daily_consumption"]
    o["days_to_safety"] = (o["stock_on_hand"] - o["safety_stock"]) / o["daily_consumption"]
    o["shortage_days_erp"] = np.maximum(o["erp_arrival_days"] - o["days_to_zero"], 0)
    o["shortage_days_p85"] = np.maximum(o["p85_arrival_days"] - o["days_to_zero"], 0)
    o["extra_units_needed"] = np.ceil(o["shortage_days_p85"] * o["daily_consumption"])
    o["risk"] = np.where(o["p50_arrival_days"] > o["days_to_zero"], "Yuksek",
                         np.where(o["p85_arrival_days"] > o["days_to_safety"], "Orta", "Dusuk"))
    return o


def make_template(db_path=DB_PATH, n=12) -> pd.DataFrame:
    """Kullanicinin indirip doldurabilecegi ornek CSV (veritabanindaki gercek acik siparislerden)."""
    con = sqlite3.connect(db_path)
    t = pd.read_sql("SELECT po_id, supplier_id, material_id, order_date, quantity, erp_lead_time_days, "
                    "stock_on_hand, safety_stock, daily_consumption FROM open_orders_risk ORDER BY po_id DESC "
                    f"LIMIT {int(n)}", con)
    con.close()
    return t
