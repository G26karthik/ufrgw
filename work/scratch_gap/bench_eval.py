"""Decode and score the four fit-target variants on one benchmark (N = natural label-hidden, D = density stress test).
Decoder = run_all steps 5-6: bias tuned on Aug -> takeover archetype forcing (postproc.apply) -> bias re-tuned on Aug
-> device-zone override with edges found by ts_event_post.dp_edges on the Sep decode (label-free) + bait rule.
Steps 12-17 of run_all (test-only rule layers, about 300 test rows, +.006 public) are not applied.
Scores: supplied Sep labels. `clean_final` = event-majority cleaned labels (not valid for AC: its target uses the same cleaning).
`zone_ok` = share of Sep rows on a class-zone device (zone class from the Jul+Aug edges in feats/ts_event.py) whose
prediction equals the zone class: a truth proxy that does not come from event cleaning.
usage (cwd=work, PYTHONPATH=.): python scratch_gap/bench_eval.py N|D"""
import sys
import numpy as np
import polars as pl
import train as T
from common import load_all, LABEL2ID as L
from clean_labels import clean, fraud_events
from feats.ts_event import TR_EDGES, zone_ids
from ts_judge_post import zones, override

c = pl.col; K = T.K; B = sys.argv[1]; VARS = ["C", "RAW", "AC", "CE"]


def f1w(y, p, w=None):
    w = np.ones(len(y)) if w is None else w; ok = p == y
    return 2 * np.bincount(y[ok], w[ok], K) / np.maximum(np.bincount(y, w, K) + np.bincount(p, w, K), 1e-9)


def force(P, D):
    Q = P.copy()
    for col, k in (("takeover__arch_sim", L["sim_swap"]), ("takeover__arch_ato", L["account_takeover"])):
        h = D.select((c(col).fill_nan(None) == 1).fill_null(False)).to_series().to_numpy(); Q[h] = 0; Q[h, k] = 1
    return Q


Da, Ds = (pl.read_parquet(f"scratch_gap/{B}_{k}_rows.parquet") for k in ("aug", "sep"))
ya, y = Da["y_true"].to_numpy().astype(np.int64), Ds["y_true"].to_numpy().astype(np.int64)
tx = load_all().filter(~c("is_test")); _, FE = fraud_events(tx, g=48 * 3600)
ext = Ds.select("txn_id").join(tx.select("txn_id", clean(tx).alias("yc")), on="txn_id", how="left", maintain_order="left") \
        .join(FE.select("txn_id", "ev"), on="txn_id", how="left", maintain_order="left")
yc = ext["yc"].to_numpy().astype(np.int64); ev = ext["ev"].to_numpy().astype(float); grp = np.where(np.isnan(ev), -1, ev).astype(np.int64)
rng = np.random.default_rng(2026); n = len(y)
boot = [np.where(grp >= 0, rng.poisson(1, grp.max() + 1)[np.maximum(grp, 0)], rng.poisson(1, n)).astype(float) for _ in range(200)]
dn = Ds["dn"].to_numpy(); zt = zone_ids(dn, np.zeros(n, bool), [2 ** 62] * 12); zrow = (zt > 0) & (dn >= TR_EDGES[0])
rule = Ds["ts_bait__rule"].fill_null(0).to_numpy()
out = {}
for v in VARS:
    Pa, Ps = np.load(f"scratch_gap/P{B}_{v}_aug.npy"), np.load(f"scratch_gap/P{B}_{v}_sep.npy")
    pre = T.decode(Ps, T.tune_bias(Pa, ya))
    p1 = T.decode(force(Ps, Ds), T.tune_bias(force(Pa, Da), ya))
    z, e = zones(Ds, p1, TR_EDGES[0], TR_EDGES[-1]); q = override(p1, z, rule)
    out[v] = dict(pre=pre, p1=p1, q=q, edges=e)
rows = []
for v in VARS:
    o, b = out[v], out["C"]
    d = lambda k: np.quantile([f1w(y, o[k], w).mean() - f1w(y, b[k], w).mean() for w in boot], [.05, .95]).round(4).tolist()
    rows.append(dict(variant=v, pre=round(f1w(y, o["pre"]).mean(), 4), forced=round(f1w(y, o["p1"]).mean(), 4), final=round(f1w(y, o["q"]).mean(), 4),
                     d_pre=round(f1w(y, o["pre"]).mean() - f1w(y, b["pre"]).mean(), 4), d_final=round(f1w(y, o["q"]).mean() - f1w(y, b["q"]).mean(), 4),
                     boot_pre=str(d("pre")) if v != "C" else "", boot_final=str(d("q")) if v != "C" else "",
                     chg_pre=int((o["pre"] != b["pre"]).sum()), chg_final=int((o["q"] != b["q"]).sum()), overridden=int((o["q"] != o["pre"]).sum()),
                     clean_final=round(f1w(yc, o["q"]).mean(), 4), zone_ok_pre=round((o["pre"][zrow] == zt[zrow]).mean(), 3),
                     zone_ok_final=round((o["q"][zrow] == zt[zrow]).mean(), 3), legit_fp=int(((y == 0) & (o["q"] > 0)).sum())))
pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(40)
print(f"benchmark {B}: Sep rows {n}, fraud-labelled {int((y > 0).sum())}, rows on class-zone devices {int(zrow.sum())}")
print(pl.DataFrame(rows))
print("zone edges found on the Sep decode (C):", out["C"]["edges"], "| label-fit edges:", TR_EDGES)
for v in VARS:
    q = out[v]["q"]; tp = np.bincount(y[q == y], minlength=K); P_ = np.bincount(q, minlength=K); N_ = np.bincount(y, minlength=K)
    print(f"{v:4s} final P/R/F1: " + " ".join(f"{T.LABELS[k][:5]} {tp[k]/max(P_[k],1):.2f}/{tp[k]/N_[k]:.2f}/{2*tp[k]/(P_[k]+N_[k]):.3f}" for k in range(1, K)))
np.savez(f"scratch_gap/dec_{B}.npz", y=y, yc=yc, **{f"{v}_{k}": out[v][k] for v in VARS for k in ("pre", "p1", "q")})
