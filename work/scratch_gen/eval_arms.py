"""Evaluate the arms of scratch_gen/arms.py. Bias is tuned on Aug (development) only, then applied to Sep. cwd=work, PYTHONPATH=.
PASS RULE (fixed before the Sep comparison was read): an arm C passes if, on Sep with supplied labels,
  macro-F1(C) - macro-F1(A) >= .002 and macro-F1(C) - best B arm >= .002,
  the event-bootstrap 90% interval of (C - A) excludes 0, and the private side gains in >= 80% of the 30/70 splits."""
import sys, numpy as np, polars as pl
import train as T
from common import load_all
from clean_labels import clean, fraud_events
c = pl.col; K = T.K
ARMS = sys.argv[1:] or ["A", "B05", "B1", "BR", "C05", "C1"]

def f1w(y, p, w=None):
    w = np.ones(len(y)) if w is None else w; ok = p == y
    tp = np.bincount(y[ok], w[ok], K); den = np.bincount(y, w, K) + np.bincount(p, w, K)
    return 2 * tp / np.maximum(den, 1e-9)

tx = load_all().filter(~c("is_test")); tx = tx.with_columns(clean(tx).alias("yc"))
_, FE = fraud_events(tx, g=48 * 3600)
sep = pl.read_parquet("scratch_gen/ids_sep.parquet").join(tx.select("txn_id", "yc"), on="txn_id", how="left", maintain_order="left") \
        .join(FE.select("txn_id", "ev"), on="txn_id", how="left", maintain_order="left") \
        .join(pl.read_parquet("scratch_gen/src_rows.parquet").select("txn_id", c("fam").alias("bait_fam")), on="txn_id", how="left", maintain_order="left")
aug = pl.read_parquet("scratch_gen/ids_aug.parquet")
y, yc, ya = sep["y"].to_numpy().astype(np.int64), sep["yc"].to_numpy().astype(np.int64), aug["y"].to_numpy().astype(np.int64)
hard = sep["bait_fam"].is_not_null().to_numpy()
pred, rows = {}, []
for a in ARMS:
    b = T.tune_bias(np.load(f"scratch_gen/P_{a}_aug.npy"), ya)
    P = np.load(f"scratch_gen/P_{a}_sep.npy"); p = T.decode(P, b); pred[a] = p
    f, fc = f1w(y, p), f1w(yc, p)
    pr = lambda k: (((p == k) & (y == k)).sum() / max((p == k).sum(), 1), ((p == k) & (y == k)).sum() / (y == k).sum())
    rows.append(dict(arm=a, aug_macro=round(f1w(ya, T.decode(np.load(f"scratch_gen/P_{a}_aug.npy"), b)).mean(), 4), sep_macro=round(f.mean(), 4),
                     sep_macro_argmax=round(f1w(y, P.argmax(1)).mean(), 4), sep_macro_clean=round(fc.mean(), 4),
                     fc_f1=round(f[9], 3), fc_p=round(pr(9)[0], 3), fc_r=round(pr(9)[1], 3), ph_f1=round(f[6], 3), ph_p=round(pr(6)[0], 3), ph_r=round(pr(6)[1], 3),
                     legit_fp=int(((y == 0) & (p > 0)).sum()), fc_as_ph=int(((y == 9) & (p == 6)).sum()), ph_as_fc=int(((y == 6) & (p == 9)).sum()),
                     hard_acc=round((p[hard] == y[hard]).mean(), 3), n_fc_pred=int((p == 9).sum())))
pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(30)
print(pl.DataFrame(rows)); print("Sep rows", len(y), "bait-family rows (hard subset)", int(hard.sum()))
print("per-class Sep F1 (supplied labels):"); print(pl.DataFrame({"class": T.LABELS, **{a: f1w(y, pred[a]).round(3) for a in ARMS}}))
# fixed 30/70 row splits (simulation: assumes a random row split) and event bootstrap (fraud events whole, legit rows single)
rng = np.random.default_rng(2026); n = len(y)
pub = [rng.random(n) < 0.3 for _ in range(200)]
ev = sep["ev"].to_numpy().astype(float); grp = np.where(np.isnan(ev), -1, ev).astype(np.int64); ne = grp.max() + 1
boot = [np.where(grp >= 0, rng.poisson(1, ne)[np.maximum(grp, 0)], rng.poisson(1, n)).astype(float) for _ in range(200)]
out = []
for a in ARMS:
    if a == "A": continue
    dp = np.array([f1w(y[m], pred[a][m]).mean() - f1w(y[m], pred["A"][m]).mean() for m in pub])
    dq = np.array([f1w(y[~m], pred[a][~m]).mean() - f1w(y[~m], pred["A"][~m]).mean() for m in pub])
    db = np.array([f1w(y, pred[a], w).mean() - f1w(y, pred["A"], w).mean() for w in boot])
    out.append(dict(arm=a, d_full=round(f1w(y, pred[a]).mean() - f1w(y, pred["A"]).mean(), 4), d_pub_mean=round(dp.mean(), 4), d_pub_sd=round(dp.std(), 4),
                    d_priv_mean=round(dq.mean(), 4), d_priv_sd=round(dq.std(), 4), corr_pub_priv=round(np.corrcoef(dp, dq)[0, 1], 2),
                    priv_gain_share=round((dq > 0).mean(), 2), priv_gain_given_pub_gain=round((dq[dp > 0] > 0).mean(), 2) if (dp > 0).any() else None,
                    boot_p05=round(np.quantile(db, .05), 4), boot_p95=round(np.quantile(db, .95), 4)))
print("delta vs A on Sep: 200 fixed 30/70 row splits, and 200 event-bootstrap draws"); print(pl.DataFrame(out))
