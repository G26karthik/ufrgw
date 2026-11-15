"""Agent A: what the 190-column vector shows for missed events against what raw-history aggregates show.
cwd=work, PYTHONPATH=.  python scratch_atomic/representation/02_trace.py > scratch_atomic/representation/02_out.txt
Needs 01_missed.py outputs (missed_events_N_m*.parquet, detected_events_N_m*.parquet, controls_N_m*.parquet).
For each fraud class: AUC(missed event rep rows vs their matched legit controls) for (a) every column of the 190-col
vector (benchmark N table) and (b) label-free raw aggregates that have no column in the vector: sender's own profile,
receiver's own profile, +-48 h link counts on sender / receiver / device, new_payee against the observed pair history.
AUC near .5 on every vector column and away from .5 on a raw aggregate = information the representation loses.
Month 8 (fold 0 table) is the freeze month, month 9 (fold 1 table) is reported."""
import sys
import numpy as np
import polars as pl
from scipy.stats import rankdata
from common import load_all, LABELS
from feats.ring import _sym_roll

c = pl.col; OUT = "scratch_atomic/representation/"; B = sys.argv[1] if len(sys.argv) > 1 else "N"
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(280); pl.Config.set_fmt_str_lengths(40)
tx = load_all()
la = c("amount").log1p()
pr = pl.concat_str("sender_id", "receiver_id")
R = tx.select(
    "txn_id", "timestamp", "ts", "sender_id", "receiver_id", "device_id", "month",
    (la - la.median().over("sender_id")).alias("amt_vs_snd_med"),
    (la - la.max().over("sender_id")).alias("amt_vs_snd_max"),
    (la - la.median().over("receiver_id")).alias("amt_vs_rcv_med"),
    ((c("txn_type") == c("txn_type")).sum().over("sender_id", "txn_type") / pl.len().over("sender_id")).alias("snd_type_share"),
    (pl.len().over("sender_id", "receiver_type") / pl.len().over("sender_id")).alias("snd_rtype_share"),
    (pl.len().over("sender_id", c("merchant_category").fill_null("NA")) / pl.len().over("sender_id")).alias("snd_cat_share"),
    (pl.len().over("sender_id", "device_city") / pl.len().over("sender_id")).alias("snd_city_share"),
    (pl.len().over("sender_id", "app") / pl.len().over("sender_id")).alias("snd_app_share"),
    (c("timestamp").dt.hour() - c("timestamp").dt.hour().median().over("sender_id")).abs().alias("hour_dev_snd"),
    (c("login_to_txn_sec").log1p() - c("login_to_txn_sec").log1p().median().over("sender_id")).alias("login_vs_snd_med"),
    (c("balance_before").log1p() - c("balance_before").log1p().median().over("sender_id")).alias("bal_vs_snd_med"),
    pl.len().over("sender_id").alias("snd_n_all"),
    c("receiver_id").n_unique().over("sender_id").alias("snd_nrcv_all"),
    c("new_payee").mean().over("sender_id").alias("snd_newpayee_share"),
    c("new_payee").mean().over("receiver_id").alias("rcv_newpayee_share"),
    c("ts").rank("ordinal").over(pr).alias("pair_rank"),
    pl.len().over(pr).alias("pair_n_all"),
    (c("ts") - c("ts").min().over("sender_id")).alias("snd_secs_since_first"),
    (c("ts") - c("ts").min().over("receiver_id")).alias("rcv_secs_since_first"),
    (c("ts").max().over("receiver_id") - c("ts").min().over("receiver_id")).alias("rcv_life_secs"),
).with_columns(
    ((c("pair_rank") > 1) & (tx["new_payee"] == 1)).cast(pl.Int8).alias("np1_but_pair_seen_before"),
    ((c("pair_rank") == 1) & (tx["new_payee"] == 0)).cast(pl.Int8).alias("np0_but_pair_first"),
)
# receiver's activity as a sender (P2P receivers that also send), label-free
out_n = tx.group_by("sender_id").agg(pl.len().alias("rcv_out_n_all")).rename({"sender_id": "receiver_id"})
R = R.join(out_n, on="receiver_id", how="left").with_columns(c("rcv_out_n_all").fill_null(0))
df = tx.select("txn_id", "timestamp", "sender_id", "receiver_id", "device_id", "new_payee", (c("receiver_type") == "P2P").cast(pl.Int8).alias("p2p"))
for key, other, nm in (("sender_id", "receiver_id", "snd"), ("receiver_id", "sender_id", "rcv"), ("device_id", "sender_id", "dev")):
    r = _sym_roll(df, key, "48h", [pl.len().alias(f"{nm}_n48"), c(other).n_unique().alias(f"{nm}_nother48"), c("new_payee").sum().alias(f"{nm}_newpayee48")])
    R = R.join(r, on="txn_id", how="left")
RAW = [x for x in R.columns if x not in ("txn_id", "timestamp", "ts", "sender_id", "receiver_id", "device_id", "month", "pair_rank")]
print("raw aggregates:", RAW)
print("rate over ALL rows per month of the two new_payee consistency flags (Oct = test):")
print(R.group_by("month").agg(pl.len().alias("n"), c("np1_but_pair_seen_before").mean().round(4), c("np0_but_pair_first").mean().round(4),
                              c("snd_n48").mean().round(2), c("rcv_n48").mean().round(2), c("dev_n48").mean().round(2)).sort("month"))


def auc(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 8 or len(b) < 8: return float("nan")
    r = rankdata(np.concatenate([a, b]))
    return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


res = []
for m, fold in ((8, 0), (9, 1)):
    miss, det, ctl = (pl.read_parquet(f"{OUT}{k}_{B}_m{m}.parquet") for k in ("missed_events", "detected_events", "controls"))
    ids = pl.concat([miss["txn_id"], det["txn_id"], ctl["txn_id"]]).unique().to_list()
    X = pl.scan_parquet(f"scratch_gap/{B}_f{fold}.parquet").filter(c("txn_id").is_in(ids)).collect().drop("month", "is_test", "y_fit")
    VEC = [x for x in X.columns if x != "txn_id"]
    T_ = X.join(R.select("txn_id", *RAW), on="txn_id", how="left")
    Mx = miss.select("txn_id", "y_true", "marg").join(T_, on="txn_id", how="left")
    Dx = det.select("txn_id", "y_true").join(T_, on="txn_id", how="left")
    Cx = ctl.join(T_, on="txn_id", how="left")
    for k in [0] + sorted(Mx["y_true"].unique().to_list()):
        mk = Mx if k == 0 else Mx.filter(c("y_true") == k); dk = Dx if k == 0 else Dx.filter(c("y_true") == k)
        ck = Cx.filter(c("case_txn_id").is_in(mk["txn_id"].to_list()))
        cf = mk.filter(c("marg") <= -3); ccf = Cx.filter(c("case_txn_id").is_in(cf["txn_id"].to_list()))
        for kind, cols in (("vector", VEC), ("raw", RAW)):
            for x in cols:
                f = lambda d: d[x].cast(pl.Float64).fill_nan(None).to_numpy()
                res.append(dict(month=m, cls="ALL" if k == 0 else LABELS[k], kind=kind, col=x, n_missed_ev=mk.height, n_conf=cf.height,
                                auc_missed=auc(f(mk), f(ck)), auc_conf=auc(f(cf), f(ccf)), auc_detected=auc(f(dk), f(ck)),
                                med_missed=mk[x].median(), med_ctl=ck[x].median(), med_det=dk[x].median()))
A = pl.DataFrame(res).with_columns((c("auc_missed") - .5).abs().alias("sep_missed"), (c("auc_conf") - .5).abs().alias("sep_conf"))
A.write_csv(f"{OUT}trace_auc_{B}.csv")
W = A.pivot(on="month", index=["cls", "kind", "col"], values=["n_missed_ev", "n_conf", "auc_missed", "auc_conf", "auc_detected", "med_missed", "med_ctl", "med_det"])
for cls in ["ALL"] + LABELS[1:]:
    w = W.filter(c("cls") == cls)
    if w.height == 0: continue
    print(f"\n==== {cls}: missed events Aug {w['n_missed_ev_8'][0]} (confident-legit {w['n_conf_8'][0]}), Sep {w['n_missed_ev_9'][0]} (confident-legit {w['n_conf_9'][0]})")
    for kind in ("vector", "raw"):
        t = w.filter(c("kind") == kind).with_columns((c("auc_missed_8") - .5).abs().alias("s8")).sort("s8", descending=True, nulls_last=True).head(8 if kind == "vector" else 12)
        print(f"-- {kind} columns ranked on August |AUC - .5| (missed vs matched controls); September reported next to it")
        print(t.select("col", c("auc_missed_8").round(3), c("auc_missed_9").round(3), c("auc_conf_8").round(3), c("auc_conf_9").round(3),
                       c("auc_detected_8").round(3), c("auc_detected_9").round(3), c("med_missed_9").round(3), c("med_ctl_9").round(3), c("med_det_9").round(3)))
