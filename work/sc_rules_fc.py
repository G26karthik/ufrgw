"""sc_fc: bait-aware test zone edges (notes/sc_fc.md). Label-free: test features + the model's own decode.

Train fact (all months): rows of a bait event (feats/ts_bait rule > 0) that sit on a zone device are in the phishing or
fake_cashback zone 71/72 times (Jul+Aug 60/61, Sep 26/26). No qr-zone row is a bait row (0/230).
The judge DP (ts_event_post.dp_edges) ignores this: in Oct it puts the answered-bait event on devices 474358/474359
in the qr zone (edge ph|qr = 474352), so 5 zone rows are qr_swap. The same DP with a bait emission (bait devices pay
-8 outside ph/fc) moves ph|qr to 474360; contiguity puts 474352-474359 in the phishing zone.
apply(): rerun the DP with and without the bait emission on the model decode (subs/h6_ppq.csv, the judge's evidence),
and move rows whose zone changed and whose label is still the old zone's type to the new zone's type.
usage (cwd=work): python sc_rules_fc.py   -> subs/sc_fc.csv from subs/v3_final.csv
"""
import numpy as np
import polars as pl
from common import ROOT, LABELS, LABEL2ID as L, load_all, run_feature
from feats.ts_event import TR_EDGES, TYPES, zone_ids, test_block
from ts_event_post import dp_edges

c = pl.col
K = len(TYPES)
PHFC = [TYPES.index("phishing_collect"), TYPES.index("fake_cashback")]


def dp_bait(dev_ids, dev_preds, dev_multi, dev_bait, lo, hi, pen=-8.0):
    """ts_event_post.dp_edges (default sim/ATO group) + bait emission."""
    N = hi - lo
    exp = np.diff(TR_EDGES) / (TR_EDGES[-1] - TR_EDGES[0]) * N
    S = np.zeros((N, K)); kk = np.arange(K); tk = np.isin(kk, [3, 4])
    hit, miss = np.log(0.8), np.log(0.2 / (K - 1))
    row = {t: np.where(np.isin(kk, [3, 4]) if t in (3, 4) else kk == t, hit, miss) for t in range(K)}
    for i, ps, m, b in zip(dev_ids, dev_preds, dev_multi, dev_bait):
        for p in ps:
            if p: S[i - lo] += row[TYPES.index(LABELS[p])]
        if m: S[i - lo] += np.where(tk, 0.0, -8.0)
        if b: S[i - lo] += np.where(np.isin(kk, PHFC), 0.0, pen)
    C = np.vstack([np.zeros(K), np.cumsum(S, 0)])
    best = np.full((K + 1, N + 1), -np.inf); arg = np.zeros((K + 1, N + 1), int); best[0, 0] = 0
    for k in range(K):
        for e in range(N + 1):
            s = np.arange(e + 1)
            v = best[k, s] + C[e, k] - C[s, k] - (e - s - exp[k]) ** 2 / (2 * max(exp[k], 1))
            arg[k + 1, e] = int(np.argmax(v)); best[k + 1, e] = v[arg[k + 1, e]]
    out = [N]
    for k in range(K, 0, -1): out.append(arg[k, out[-1]])
    return (np.array(out[::-1]) + lo).tolist()


def apply(sub_df, tx, evidence="subs/h6_ppq.csv"):
    """sub_df: id,label (sample order) -> sub_df with bait-aware zone moves."""
    ev = pl.read_csv(ROOT / evidence)
    assert ev["id"].equals(sub_df["id"])
    D = sub_df.select(c("id").alias("txn_id")).join(
        tx.select("txn_id", c("device_id").str.slice(2).cast(pl.Int64).alias("dn")).join(
            run_feature("ts_bait", tx).select("txn_id", c("ts_bait__rule").alias("rule")), on="txn_id"),
        on="txn_id", how="left", maintain_order="left")
    lo, hi = test_block(tx)
    g = (D.with_columns(pl.Series("p", [L[s] for s in ev["label"].to_list()])).filter(c("dn").is_between(lo, hi - 1))
         .group_by("dn").agg(c("p"), (pl.len() > 1).alias("multi"), (c("rule") > 0).any().alias("bait")).sort("dn"))
    a = [g[k].to_list() for k in ("dn", "p", "multi", "bait")]
    e0, e1 = dp_edges(*a[:3], lo, hi), dp_bait(*a, lo, hi)
    print("edges judge", e0, "\nedges bait ", e1)
    dn, one = D["dn"].to_numpy(), np.ones(D.height, bool)
    z0, z1 = zone_ids(dn, one, e0), zone_ids(dn, one, e1)
    lab = sub_df["label"].to_numpy().copy()
    mv = (z0 != z1) & (z0 > 0) & (z1 > 0) & (lab == np.array(LABELS)[z0])
    lab[mv] = np.array(LABELS)[z1[mv]]
    return sub_df.with_columns(pl.Series("label", lab))


if __name__ == "__main__":
    sub = pl.read_csv(ROOT / "subs" / "v3_final.csv")
    out = ROOT / "subs" / "sc_fc.csv"
    assert not out.exists(), f"{out} exists; never overwrite"
    new = apply(sub, load_all())
    ch = sub.join(new.rename({"label": "to"}), on="id").filter(c("label") != c("to"))
    print(ch)
    # expected on Oct test: exactly the 5 qr_swap rows on devices 474352..474359 -> phishing_collect
    assert ch.height == 5 and (ch["label"] == "qr_swap").all() and (ch["to"] == "phishing_collect").all()
    assert new.height == sub.height and new["label"].is_in(LABELS).all()
    new.write_csv(out)
    print("wrote", out)
