"""
Olist gercek-veri deneyleri icin ORTAK kod (02 ve 03 numarali scriptler bunu kullanir).

Icerik:
  load_features()  -> leadtime_olist.db'den siparisleri okur, satici gecmisi ozelliklerini uretir
  make_model(q)    -> quantile modeli (LightGBM varsa o, yoksa sklearn)
  pinball(), update_results()
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    BACKEND = "LightGBM"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    BACKEND = "sklearn HistGradientBoosting (lightgbm kurulu degil)"

HERE = Path(__file__).parent
DB = HERE / "leadtime_olist.db"
RESULTS = HERE / "results.json"
SPLIT = pd.Timestamp("2018-04-01")          # oncesi egitim/kalibrasyon, sonrasi test

NUM = ["erp_lead_time_days", "quantity", "price", "freight_value", "weight_g", "distance_km", "same_state",
       "order_month", "order_weekday", "seller_orders_14d", "seller_avg_delay_90d",
       "seller_std_delay_90d", "seller_avg_lead_90d", "seller_n_90d"]
CAT = ["supplier_id", "category", "customer_state", "seller_state"]
FEATURES = NUM + CAT


def load_features():
    """Siparisleri oku + satici gecmisi ozellikleri (sadece siparis ANINDA bilinebilenler)."""
    con = sqlite3.connect(DB)
    po = pd.read_sql("SELECT * FROM purchase_orders ORDER BY order_ts", con,
                     parse_dates=["order_ts", "delivery_ts", "promised_date"]).reset_index(drop=True)
    con.close()

    # Her siparis icin: o saticinin, bu siparisten ONCE TESLIM EDILMIS son 90 gundeki siparisleri.
    # Henuz teslim edilmemis siparisin sonucu bilinemez -> kullanmiyoruz (data leakage onlemi).
    n = len(po)
    avg_delay = np.full(n, np.nan); std_delay = np.full(n, np.nan)
    avg_lead = np.full(n, np.nan); n90 = np.zeros(n); load14 = np.zeros(n)
    for _, g in po.groupby("supplier_id"):
        idx = g.index.to_numpy()
        ot = g["order_ts"].to_numpy()                       # zaten artan sirada
        srt = np.argsort(g["delivery_ts"].to_numpy())
        dt = g["delivery_ts"].to_numpy()[srt]
        dly = g["delay_days"].to_numpy(float)[srt]
        lead = g["actual_lead_time_days"].to_numpy(float)[srt]
        c1 = np.concatenate([[0], np.cumsum(dly)])
        c2 = np.concatenate([[0], np.cumsum(dly ** 2)])
        c3 = np.concatenate([[0], np.cumsum(lead)])
        hi = np.searchsorted(dt, ot, side="left")                            # teslimi siparisten once olanlar
        lo = np.searchsorted(dt, ot - np.timedelta64(90, "D"), side="left")  # ... ve son 90 gunde olanlar
        k = (hi - lo).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = (c1[hi] - c1[lo]) / k
            v = (c2[hi] - c2[lo]) / k - m ** 2
            avg_delay[idx] = np.where(k > 0, m, np.nan)
            std_delay[idx] = np.where(k > 0, np.sqrt(np.clip(v, 0, None)), np.nan)
            avg_lead[idx] = np.where(k > 0, (c3[hi] - c3[lo]) / k, np.nan)
        n90[idx] = k
        h2 = np.searchsorted(ot, ot, side="left")                            # yuk: son 14 gunde verilen siparis
        l2 = np.searchsorted(ot, ot - np.timedelta64(14, "D"), side="left")
        load14[idx] = h2 - l2

    po["seller_avg_delay_90d"] = avg_delay
    po["seller_std_delay_90d"] = std_delay
    po["seller_avg_lead_90d"] = avg_lead
    po["seller_n_90d"] = n90
    po["seller_orders_14d"] = load14
    po["order_month"] = po["order_ts"].dt.month
    po["order_weekday"] = po["order_ts"].dt.dayofweek
    for c in CAT:
        po[c] = po[c].astype("category")
    return po


def make_model(q):
    if BACKEND == "LightGBM":
        return lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=400, learning_rate=0.05,
                                 num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, random_state=42, verbose=-1)
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=400, learning_rate=0.05,
                                         max_leaf_nodes=31, min_samples_leaf=50,
                                         categorical_features="from_dtype", random_state=42)


def pinball(y, p, q):
    d = np.asarray(y, float) - np.asarray(p, float)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def update_results(section, payload):
    """results.json icine bir bolum yaz (arayuz ve grafikler buradan okur)."""
    data = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    data[section] = payload
    data["_backend"] = BACKEND
    RESULTS.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
