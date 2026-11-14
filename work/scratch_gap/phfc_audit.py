"""Audit of the cited look-alike result "phishing label, fake_cashback look: event AUC .80, precision .56 at t=.5"
(scratch_x/classwise/e24.py: Aug+Sep pooled, GroupKFold by event, look = model argmax before any rule).
Here, on benchmark N and after the existing decoder (bias, archetype forcing, zones, bait rule):
  look rows = rows the final decode calls fake_cashback; events = connected components over accounts inside a month;
  events with a zone device are left out (already typed), as in e24; same 47 features and the same classifier settings.
  Discovery on August only: classifier fit on August events, threshold picked on August (grouped 5-fold OOF, best August
  macro-F1 after flipping whole events to phishing). Check on September: flip the flagged events, score the full decode.
usage (cwd=work, PYTHONPATH=.): python scratch_gap/phfc_audit.py"""
import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import train as T
from common import load_all, LABEL2ID as L
from clean_labels import fraud_events
from feats.ts_event import TR_EDGES
from ts_judge_post import zones, override

c = pl.col; K = T.K; PH, FC = L["phishing_collect"], L["fake_cashback"]


def f1c(y, p):
    return T.macro_f1(y, p)


def force(P, D):
    Q = P.copy()
    for col, k in (("takeover__arch_sim", L["sim_swap"]), ("takeover__arch_ato", L["account_takeover"])):
        h = D.select((c(col).fill_nan(None) == 1).fill_null(False)).to_series().to_numpy(); Q[h] = 0; Q[h, k] = 1
    return Q


Dm = {k: pl.read_parquet(f"scratch_gap/N_{k}_rows.parquet") for k in ("aug", "sep")}
Pm = {k: force(np.load(f"scratch_gap/PN_C_{k}.npy"), Dm[k]) for k in Dm}
b = T.tune_bias(Pm["aug"], Dm["aug"]["y_true"].to_numpy().astype(np.int64))
tx = load_all().filter(~c("is_test"))
dt = pl.from_epoch(c("ts"), time_unit="s")
FE = ["amount", "hrf", "dow", "dom", "sim_age_hours", "device_age_days", "sender_account_age_days", "new_payee", "pin_retries", "login_to_txn_sec",
      "balance_before", "ab", "aintf", "a100", "homef", "cmf", "rcm", "rcv_age", "ttc", "p2p",
      "e_n", "e_ns", "e_nr", "e_dur", "e_amed", "e_astd", "e_asum", "e_amax", "e_amin", "e_hr", "e_hrsd", "e_np", "e_home", "e_cm", "e_pin", "e_ab",
      "e_simmin", "e_devmin", "e_ntt", "e_p2p", "e_login", "e_sage", "e_rage", "e_pos", "e_arel", "e_rate", "e_snr"]
TT = {"PAY": 1, "QR": 2, "COLLECT": 3, "AUTOPAY": 4, "REFUND": 5}
Q, Fm = {}, {}
for k, D in Dm.items():
    p1 = T.decode(Pm[k], b); z, _ = zones(D, p1, TR_EDGES[0], TR_EDGES[-1]); q = override(p1, z, D["ts_bait__rule"].fill_null(0).to_numpy()); Q[k] = q
    f = D.select("txn_id", c("y_true").alias("y"), pl.Series("q", q)).with_row_index("pos").filter(c("q") == FC).join(tx, on="txn_id", how="left")
    f = f.with_columns(hrf=dt.dt.hour(), dow=dt.dt.weekday(), dom=dt.dt.day(), rcv_age=(dt.dt.date() - c("rcv_opened_on")).dt.total_days(),
                       dnum=c("device_id").str.slice(2).cast(pl.Int64), snum=c("sender_id").str.slice(2).cast(pl.Int64)).with_columns(
        homef=(c("dnum") == c("snum")).cast(pl.Int8), cmf=(c("device_city") != c("snd_home_city")).cast(pl.Int8), rcm=(c("device_city") != c("rcv_home_city")).cast(pl.Int8),
        ab=c("amount") / (c("balance_before") + 1), aintf=(c("amount") == c("amount").round(0)).cast(pl.Int8), a100=c("amount") % 100,
        p2p=(c("receiver_type") == "P2P").cast(pl.Int8), ttc=c("txn_type").replace_strict(TT, return_dtype=pl.Int8), zdev=(c("dnum") >= 423000).cast(pl.Int8))
    acc = {a: i for i, a in enumerate(set(f["sender_id"]) | set(f["receiver_id"]))}
    s, r = [acc[a] for a in f["sender_id"]], [acc[a] for a in f["receiver_id"]]
    _, comp = connected_components(coo_matrix((np.ones(len(s)), (s, r)), shape=(len(acc), len(acc))), directed=False)
    f = f.with_columns(ev=pl.Series(comp[s]))
    g = f.group_by("ev").agg(pl.len().alias("e_n"), c("sender_id").n_unique().alias("e_ns"), c("receiver_id").n_unique().alias("e_nr"), (c("ts").max() - c("ts").min()).alias("e_dur"),
                             c("amount").median().alias("e_amed"), c("amount").std().alias("e_astd"), c("amount").sum().alias("e_asum"), c("amount").max().alias("e_amax"), c("amount").min().alias("e_amin"),
                             c("hrf").mean().alias("e_hr"), c("hrf").std().alias("e_hrsd"), c("new_payee").mean().alias("e_np"), c("homef").mean().alias("e_home"), c("cmf").mean().alias("e_cm"),
                             c("pin_retries").mean().alias("e_pin"), c("ab").mean().alias("e_ab"), c("sim_age_hours").min().alias("e_simmin"), c("device_age_days").min().alias("e_devmin"),
                             c("ts").min().alias("e_t0"), c("ttc").n_unique().alias("e_ntt"), c("p2p").mean().alias("e_p2p"), c("login_to_txn_sec").mean().alias("e_login"),
                             c("sender_account_age_days").mean().alias("e_sage"), c("rcv_age").mean().alias("e_rage"), c("zdev").max().alias("e_z"))
    Fm[k] = f.join(g, on="ev", how="left").with_columns(e_pos=(c("ts") - c("e_t0")) / (c("e_dur") + 1), e_arel=c("amount") / (c("e_amed") + 1),
                                                         e_rate=c("e_n") / (c("e_dur") + 60), e_snr=c("e_ns") / c("e_nr"))
fit = lambda X, y: H(max_iter=200, learning_rate=.05, max_leaf_nodes=8, min_samples_leaf=10, l2_regularization=1.).fit(X, y)
sel = lambda f: f.filter(c("y").is_in([PH, FC]) & (c("e_z") == 0))
A, S = sel(Fm["aug"]), sel(Fm["sep"])
for nm, d in (("Aug", A), ("Sep", S)):
    e = d.group_by("ev").agg((c("y") == PH).mean().alias("ph"), pl.len())
    print(f"{nm}: rows called fake_cashback by the final decode {Fm[nm.lower()].height} (labelled fc {int((Fm[nm.lower()]['y'] == FC).sum())}, phishing {int((Fm[nm.lower()]['y'] == PH).sum())}, "
          f"legit {int((Fm[nm.lower()]['y'] == 0).sum())}); without zone device and labelled ph/fc: rows {d.height}, events {e.height}, phishing events {int((e['ph'] > .5).sum())}")
Xa, ya, ga = A.select(FE).to_numpy().astype(float), (A["y"] == PH).to_numpy().astype(int), A["ev"].to_numpy()
oof = np.zeros(len(ya))
for tri, tei in GroupKFold(5).split(Xa, ya, ga):
    oof[tei] = fit(Xa[tri], ya[tri]).predict_proba(Xa[tei])[:, 1]
m = fit(Xa, ya); ps = m.predict_proba(S.select(FE).to_numpy().astype(float))[:, 1]


def ev_table(d, p):
    return d.select("ev", "pos", "y").with_columns(pl.Series("p", p)).group_by("ev").agg(c("p").mean(), (c("y") == PH).mean().alias("ph"), pl.len().alias("n"), c("pos"))


Ea, Es = ev_table(A, oof), ev_table(S, ps)
for nm, E in (("Aug, grouped 5-fold OOF", Ea), ("Sep, model fit on Aug", Es)):
    print(f"event AUC {nm}: {roc_auc_score((E['ph'] > .5).to_numpy(), E['p'].to_numpy()):.3f} over {E.height} events")
ysep, yaug = (Dm[k]["y_true"].to_numpy().astype(np.int64) for k in ("sep", "aug"))


def flip(E, q, t):
    """flagged events: every row of the event that the decode calls fake_cashback -> phishing (labels play no part)."""
    p = q.copy(); pos = np.concatenate(E.filter(c("p") > t)["pos"].to_list()) if E.filter(c("p") > t).height else np.array([], int)
    p[pos] = PH; return p, pos


best = max(((f1c(yaug, flip(Ea, Q["aug"], t)[0]).mean(), t) for t in (.3, .4, .5, .6, .7, .8)))
base_a = f1c(yaug, Q["aug"]).mean()
print(f"August threshold search: base {base_a:.4f}; " + " ".join(f"t={t}: {f1c(yaug, flip(Ea, Q['aug'], t)[0]).mean() - base_a:+.4f}" for t in (.3, .4, .5, .6, .7, .8)))
tx9 = tx.join(Dm["sep"].select("txn_id"), on="txn_id", how="semi"); _, FEv = fraud_events(tx, g=48 * 3600)
ev = Dm["sep"].select("txn_id").join(FEv.select("txn_id", "ev"), on="txn_id", how="left", maintain_order="left")["ev"].to_numpy().astype(float)
grp = np.where(np.isnan(ev), -1, ev).astype(np.int64); rng = np.random.default_rng(2026); n = len(ysep)
boot = [np.where(grp >= 0, rng.poisson(1, grp.max() + 1)[np.maximum(grp, 0)], rng.poisson(1, n)).astype(float) for _ in range(200)]


def f1w(y, p, w):
    ok = p == y
    return (2 * np.bincount(y[ok], w[ok], K) / np.maximum(np.bincount(y, w, K) + np.bincount(p, w, K), 1e-9)).mean()


q0 = Q["sep"]; f0 = f1c(ysep, q0)
pr = lambda p, k: (((p == k) & (ysep == k)).sum() / max((p == k).sum(), 1), ((p == k) & (ysep == k)).sum() / (ysep == k).sum())
print(f"Sep before: macro {f0.mean():.4f} | fake_cashback P/R/F1 {pr(q0, FC)[0]:.3f}/{pr(q0, FC)[1]:.3f}/{f0[FC]:.3f} | phishing P/R/F1 {pr(q0, PH)[0]:.3f}/{pr(q0, PH)[1]:.3f}/{f0[PH]:.3f}"
      f" | phishing rows called fake_cashback {int(((ysep == PH) & (q0 == FC)).sum())}")
for name, t in ((f"threshold picked on August (t={best[1]})", best[1]), ("cited threshold t=0.5", .5)):
    p, pos = flip(Es, q0, t); f = f1c(ysep, p); fl = Es.filter(c("p") > t); yl = ysep[pos]
    db = np.quantile([f1w(ysep, p, w) - f1w(ysep, q0, w) for w in boot], [.05, .95])
    print(f"Sep, {name}: events flagged {fl.height} of {Es.height} ({int((fl['ph'] > .5).sum())} phishing events, precision {(fl['ph'] > .5).mean() if fl.height else float('nan'):.2f}); rows flipped {len(pos)}: "
          f"labelled phishing {int((yl == PH).sum())}, fake_cashback {int((yl == FC).sum())}, other {int(((yl != PH) & (yl != FC)).sum())}")
    print(f"    macro {f.mean():.4f} ({f.mean() - f0.mean():+.4f}, event bootstrap 5%..95% {db[0]:+.4f}..{db[1]:+.4f}) | fake_cashback P/R/F1 {pr(p, FC)[0]:.3f}/{pr(p, FC)[1]:.3f}/{f[FC]:.3f}"
          f" | phishing P/R/F1 {pr(p, PH)[0]:.3f}/{pr(p, PH)[1]:.3f}/{f[PH]:.3f}")
from sklearn.inspection import permutation_importance as PI
r = PI(m, Xa, ya, n_repeats=2, random_state=0, scoring="roc_auc"); o = np.argsort(-r.importances_mean)[:6]
print("top features (fit on Aug):", [(FE[i], round(float(r.importances_mean[i]), 3)) for i in o])
fl = Es.filter(c("p") > .5).select("ev").join(S.unique("ev").select("ev", "e_n", "e_dur", "e_amax", "e_ntt", "e_np"), on="ev")
print("flagged Sep events at 0.5: size / duration h / max amount / txn types:", fl.select(c("e_n").mean(), (c("e_dur") / 3600).median(), c("e_amax").median(), c("e_ntt").mean()).row(0),
      "| all look events:", S.unique("ev").select(c("e_n").mean(), (c("e_dur") / 3600).median(), c("e_amax").median(), c("e_ntt").mean()).row(0))
