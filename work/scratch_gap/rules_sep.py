"""September test of the later production rule layers that are generic (no test-only constants), alone and as a union,
on top of the benchmark decode (bias, archetype forcing, zones, bait rule). Held month = September; labels of Jul+Aug only.
  ZPROP  v3_rules.zprop (R2 takeover, R3 chain hop): off-zone rows take the type of linked zone rows.
  CZ     bc_rules_layering.cz_hits: hop-chain components take the type of their zone rows.
  LFM    bc_rules_mule.hits: fan-in rows take the type of zone rows at the same receiver.
  DD     dd_rules_both.hits (pool-bait P and stamp S): legit-decoded rows -> phishing; actor accounts from Jul+Aug labels
         and from the September decode (September is the pseudo-test month).
Union = production order ZPROP -> CZ -> LFM -> DD, each layer applied to the decode left by the previous one.
Not testable on September (test-only constants or zero train rows): QRN/qrm, RS, FCB, X999, BANDV, EDGEFIX, --qr-null.
usage (cwd=work, PYTHONPATH=.): python scratch_gap/rules_sep.py N|D [DEC_SUFFIX VARIANT]"""
import sys
import numpy as np
import polars as pl
import train as T
import v3_rules, bc_rules_mule, bc_rules_layering, dd_rules_both as DD
from common import load_all, LABEL2ID as L
from clean_labels import fraud_events
from feats.ts_event import TR_EDGES
from ts_judge_post import zones

c = pl.col; K = T.K; B = sys.argv[1]
SUF, VAR = (sys.argv[2] if len(sys.argv) > 2 else ""), (sys.argv[3] if len(sys.argv) > 3 else "C")  # decode file suffix and model variant
z_ = np.load(f"scratch_gap/dec_{B}{SUF}.npz"); y, q0, p1 = z_["y"], z_[f"{VAR}_q"], z_[f"{VAR}_p1"]
rows = pl.read_parquet(f"scratch_gap/{B}_sep_rows.parquet").select("txn_id", "dn")
tx = load_all().filter(~c("is_test"))
zid, _ = zones(rows, p1, TR_EDGES[0], TR_EDGES[-1])
D = rows.join(tx.select("txn_id", "ts", "sender_id", "receiver_id", "receiver_type"), on="txn_id", how="left", maintain_order="left") \
        .join(bc_rules_layering.chains(tx), on="txn_id", how="left", maintain_order="left")
pos = {t: i for i, t in enumerate(rows["txn_id"])}
world = tx.with_columns((c("month") >= 9).alias("is_test")).with_columns(pl.when(c("is_test")).then(None).otherwise(c("y")).alias("y"))
frame = DD.frame(world)


def layer(name, q):
    if name == "ZPROP": h = v3_rules.zprop(D, q, zid).select("txn_id", "zz")
    elif name == "CZ": h = bc_rules_layering.cz_hits(D, q, zid, band=())
    elif name == "LFM": h = bc_rules_mule.hits(D, q, zid, band=()).select("txn_id", "zz")
    else:
        f, _ = DD.hits(frame.join(pl.DataFrame({"txn_id": rows["txn_id"], "q": q.astype(np.int64)}), on="txn_id", how="left"))
        h = f.filter(c("in_P") | c("in_S")).select("txn_id", pl.lit(L["phishing_collect"]).alias("zz"))
    p = q.copy(); i = np.array([pos[t] for t in h["txn_id"]], int); p[i] = h["zz"].to_numpy(); return p


_, FE = fraud_events(tx, g=48 * 3600)
ev = rows.select("txn_id").join(FE.select("txn_id", "ev"), on="txn_id", how="left", maintain_order="left")["ev"].to_numpy().astype(float)
grp = np.where(np.isnan(ev), -1, ev).astype(np.int64); rng = np.random.default_rng(2026); n = len(y)
boot = [np.where(grp >= 0, rng.poisson(1, grp.max() + 1)[np.maximum(grp, 0)], rng.poisson(1, n)).astype(float) for _ in range(200)]


def f1w(p, w=None):
    w = np.ones(n) if w is None else w; ok = p == y
    return 2 * np.bincount(y[ok], w[ok], K) / np.maximum(np.bincount(y, w, K) + np.bincount(p, w, K), 1e-9)


def report(name, p, base):
    ch = p != base; d = [f1w(p, w).mean() - f1w(base, w).mean() for w in boot]; f, fb = f1w(p), f1w(base)
    top = sorted(((f[k] - fb[k], T.LABELS[k][:6]) for k in range(K)), key=lambda t: -abs(t[0]))[:3]
    print(f"{name:26s} rows changed {int(ch.sum()):4d} | corrected {int((ch & (p == y)).sum()):4d} | newly wrong {int((ch & (base == y)).sum()):4d} | wrong->wrong {int((ch & (p != y) & (base != y)).sum()):4d}"
          f" | macro {f.mean():.4f} ({f.mean() - fb.mean():+.4f}, boot {np.quantile(d, .05):+.4f}..{np.quantile(d, .95):+.4f}) | " + " ".join(f"{l} {v:+.3f}" for v, l in top))


print(f"benchmark {B}: decode before the later layers macro {f1w(q0).mean():.4f}")
single = {nm: layer(nm, q0) for nm in ("ZPROP", "CZ", "LFM", "DD")}
for nm, p in single.items(): report(nm + " alone", p, q0)
hit = np.stack([p != q0 for p in single.values()]); multi = hit.sum(0) >= 2
conf = multi & (np.stack(list(single.values())).max(0) != np.where(hit, np.stack(list(single.values())), 99).min(0))
print(f"rows hit by 2 or more layers: {int(multi.sum())}, of them with different outputs: {int(conf.sum())}")
u = q0
for nm in ("ZPROP", "CZ", "LFM", "DD"): u = layer(nm, u)
report("union, production order", u, q0)
np.save("scratch_gap/rules_union_" + B + ("" if VAR == "C" else "_" + VAR) + ".npy", u)
