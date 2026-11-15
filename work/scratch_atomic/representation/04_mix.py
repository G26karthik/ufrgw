"""Agent A: model-independent generator view (ADDENDUM). cwd=work, PYTHONPATH=.
python scratch_atomic/representation/04_mix.py > scratch_atomic/representation/04_out.txt
Part 1. Fraud-path share of each class's fraud-LABELLED rows on home devices, from three row-level traces of the
generator's fraud-path "finalize" step (notes/fresh_columns.md s2), no model involved:
  t_stamp = txn_type != QR and qr_vpa_mismatch == 0 ; t_app = app_version < 4 ; t_ip = ip_risk_score >= July all-row q95.
  share_k(t) = (rate_k - rate_legit) / (rate_ref - rate_legit); rate_ref for t_app / t_ip = rate among t_stamp rows (non-QR),
  for t_stamp = rate among non-QR rows with t_app & t_ip.  Then (needs held_N.parquet from 01) the same share inside the
  rows the final decoder calls legit = estimated fraud-path rows that the pipeline loses, per class.
Part 2. Trace-free fraud labels (home device, none of t_stamp / t_app / t_ip, non-integer amount) against trace-free legit
  rows: small diagnostic LightGBM on ORIGINAL fields only, fit July, AUC on August and September (frozen before Sep)."""
import numpy as np
import polars as pl
import lightgbm as lgb
from scipy.stats import rankdata
from common import load_all, LABELS

c = pl.col; OUT = "scratch_atomic/representation/"
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
tx = load_all().filter(~c("is_test") & c("month").is_in([7, 8, 9]))
q95 = float(tx.filter(c("month") == 7)["ip_risk_score"].quantile(.95))
num = lambda x: c(x).str.extract(r"(\d+)$").cast(pl.Int64, strict=False)
tx = tx.with_columns(
    (num("device_id") == num("sender_id")).fill_null(False).alias("home_dev"),
    ((c("txn_type") != "QR") & (c("qr_vpa_mismatch") == 0)).fill_null(False).alias("t_stamp"),
    (c("app_version") < 4).alias("t_app"), (c("ip_risk_score") >= q95).alias("t_ip"),
    (c("amount") == c("amount").round(0)).alias("int_amt"), (c("txn_type") != "QR").alias("nonqr"))
print("ids:", tx.select("sender_id", "device_id", "receiver_id").head(2).rows(), "home_dev share legit %.3f fraud-labelled %.3f" % (
    tx.filter(c("y") == 0)["home_dev"].mean(), tx.filter(c("y") > 0)["home_dev"].mean()))
held = pl.read_parquet(f"{OUT}held_N.parquet").select("txn_id", "q")
tx = tx.join(held, on="txn_id", how="left")
T3 = ("t_stamp", "t_app", "t_ip")
for m in (7, 8, 9):
    M = tx.filter((c("month") == m) & c("home_dev")); Lg = M.filter(c("y") == 0)
    ref = {"t_app": M.filter(c("t_stamp"))["t_app"].mean(), "t_ip": M.filter(c("t_stamp"))["t_ip"].mean(),
           "t_stamp": M.filter(c("nonqr") & c("t_app") & c("t_ip"))["t_stamp"].mean()}
    base = {t: (Lg.filter(c("nonqr")) if t == "t_stamp" else Lg)[t].mean() for t in T3}
    print(f"\n== month {m}, home-device rows: legit {Lg.height}; trace rate among legit-labelled {({k: round(v, 4) for k, v in base.items()})}; fraud-path reference rate {({k: round(v, 3) for k, v in ref.items()})}")
    rows = []
    for k in range(1, 12):
        Fk = M.filter(c("y") == k); r = dict(cls=LABELS[k], n=Fk.height)
        for t in T3:
            d = Fk.filter(c("nonqr")) if t == "t_stamp" else Fk
            r[f"share_{t[2:]}"] = round((d[t].mean() - base[t]) / (ref[t] - base[t]), 3) if d.height else None
        if m > 7:
            Mi = Fk.filter(c("q") == 0); r["n_dec_legit"] = Mi.height; r["dec_fraud_share"] = round(1 - Mi.height / max(Fk.height, 1), 3)
            for t in T3:
                d = Mi.filter(c("nonqr")) if t == "t_stamp" else Mi; n = max(d.height, 1); p = d[t].mean() if d.height else 0.0
                s = (p - base[t]) / (ref[t] - base[t]); se = (p * (1 - p) / n) ** .5 / (ref[t] - base[t])
                r[f"lost_{t[2:]}"] = f"{s * d.height:.0f} +-{2 * se * d.height:.0f}"
            r["stamp_rows_dec_legit"] = int(Mi["t_stamp"].sum())
        rows.append(r)
    print(pl.DataFrame(rows))
    if m > 7:
        print("legit-labelled home-device rows decoded legit that carry the stamp:", int(M.filter((c("y") == 0) & (c("q") == 0))["t_stamp"].sum()),
              "| decoded fraud:", int(M.filter((c("y") == 0) & (c("q") > 0))["t_stamp"].sum()), "| legit-labelled non-QR rows:", Lg.filter(c("nonqr")).height)

# ---------------- Part 2: trace-free fraud labels vs trace-free legit, original fields only (small diagnostic fit)
day = pl.col("timestamp")
Z = tx.filter(c("home_dev") & ~c("t_stamp") & ~c("t_app") & ~c("t_ip") & ~c("int_amt")).with_columns(
    day.dt.hour().alias("hour"), day.dt.minute().alias("minute"), day.dt.second().alias("second"), day.dt.weekday().alias("wd"), day.dt.day().alias("dom"),
    (day.dt.date() - c("snd_opened_on")).dt.total_days().alias("snd_age"), (day.dt.date() - c("rcv_opened_on")).dt.total_days().alias("rcv_age"),
    (day.dt.date() - c("mer_onboarded_on")).dt.total_days().alias("mer_age"),
    ((c("amount") * 100).round() % 100).alias("cents"), (c("amount") / c("balance_before")).alias("share"),
    (c("device_city") == c("snd_home_city")).cast(pl.Int8).alias("at_home"), (c("device_city") == c("rcv_home_city")).cast(pl.Int8).alias("at_rcv_home"),
    (c("device_city") == c("mer_city")).cast(pl.Int8).alias("at_mer_city"), (c("snd_bank_code") == c("rcv_bank_code")).cast(pl.Int8).alias("same_bank"),
    (c("merchant_category") == c("mer_category")).cast(pl.Int8).alias("cat_match"),
    num("sender_id").alias("snd_num"), num("receiver_id").alias("rcv_num"),
    pl.len().over("sender_id", "month").alias("snd_n_month"), pl.len().over("receiver_id", "month").alias("rcv_n_month"))
CATS = ["receiver_type", "txn_type", "merchant_category", "device_city", "app", "snd_home_city", "snd_bank_code", "snd_kyc_level", "snd_os",
        "rcv_home_city", "rcv_bank_code", "rcv_kyc_level", "rcv_os", "mer_category", "mer_city"]
Z = Z.with_columns([c(x).fill_null("NA").cast(pl.Categorical).to_physical().cast(pl.Int32).alias("c_" + x) for x in CATS])
NUMS = ["amount", "sim_age_hours", "device_age_days", "sender_account_age_days", "new_payee", "pin_retries", "login_to_txn_sec", "balance_before",
        "app_version", "ip_risk_score", "qr_vpa_mismatch", "hour", "minute", "second", "wd", "dom", "snd_age", "rcv_age", "mer_age", "cents", "share",
        "at_home", "at_rcv_home", "at_mer_city", "same_bank", "cat_match", "snd_num", "rcv_num", "snd_n_month", "rcv_n_month"]
FE = NUMS + ["c_" + x for x in CATS]
X = lambda d: d.select(FE).to_numpy().astype(np.float32)
fit = Z.filter(c("month") == 7); yf = (fit["y"] > 0).to_numpy().astype(int)
print(f"\n== Part 2: trace-free rows (home device, no stamp, app_version >= 4, ip < July q95, non-integer amount): July fit {fit.height} rows, fraud-labelled {int(yf.sum())}")
mdl = lgb.train(dict(objective="binary", learning_rate=.05, num_leaves=15, min_data_in_leaf=200, feature_fraction=.7, bagging_fraction=.8, bagging_freq=1,
                     lambda_l2=10, num_threads=4, verbose=-1, seed=0), lgb.Dataset(X(fit), yf, feature_name=FE), 150)
for m in (8, 9):
    V = Z.filter(c("month") == m); yv = (V["y"] > 0).to_numpy(); p = mdl.predict(X(V)); r = rankdata(p)
    a = (r[yv].sum() - yv.sum() * (yv.sum() + 1) / 2) / (yv.sum() * (~yv).sum())
    top = np.argsort(-p)[:len(p) // 100]
    print(f"month {m}: rows {V.height}, trace-free fraud labels {int(yv.sum())} (base rate {yv.mean():.4f}); AUC {a:.4f}; fraud-label rate in top 1% of scores {yv[top].mean():.4f}")
    for k in range(1, 12):
        yk = (V["y"] == k).to_numpy(); keep = yk | (V["y"] == 0).to_numpy(); rk = rankdata(p[keep]); b = yk[keep]
        print(f"   {LABELS[k]:20s} n {int(yk.sum()):5d} AUC vs legit {(rk[b].sum() - b.sum() * (b.sum() + 1) / 2) / (b.sum() * (~b).sum()):.4f}")
imp = sorted(zip(mdl.feature_importance("gain"), FE), reverse=True)[:8]
print("top gain:", [(n, round(float(g))) for g, n in imp])
