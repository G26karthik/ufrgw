"""Post-rule candidate from the generator device-id zones (feats/ts_event.py, notes/typesplit_event.md).

h6 test probs -> postproc.apply(qr_null=True) -> h6 bias decode (= subs/h6_ppq.csv), then rows on a zone device take the
zone's type. Test zone edges: train layout scaled to the test block, refined by a DP that keeps the zones contiguous and in
train order, with a Poisson length prior and h6's own fraud-type predictions as weak evidence.
Writes subs/ts_event.csv (every zone row overridden) and subs/ts_event_simato.csv (only the sim/ATO choice of rows h6
already decodes as sim or ATO).   usage (cwd=work): python ts_event_post.py
"""
import json
import numpy as np
import polars as pl
from common import ROOT, LABELS, LABEL2ID, load_all
from train import SAMPLE, decode, write_sub
from postproc import apply
from feats.ts_event import TYPES, TR_EDGES, test_block, edges_for, zone_ids

c = pl.col


def dp_edges(dev_ids, dev_preds, dev_multi, lo, hi, groups=((3, 4),)):
    """Contiguous zones in TYPES order over ids [lo,hi). dev_preds: decoded label ids of each device's rows.
    groups: TYPES indices pooled as evidence (a pred of one hits every zone of its group). Default: sim/ATO, which h6 forces
    by archetype. ((3, 4), (0, 1)) also pools mule/velocity (art_zonemax: h6 decodes velocity-zone rows as legit/mule)."""
    K, N = len(TYPES), hi - lo
    exp = np.diff(TR_EDGES) / (TR_EDGES[-1] - TR_EDGES[0]) * N
    S = np.zeros((N, K)); kk = np.arange(K); tk = np.isin(kk, [3, 4])  # sim/ATO zones
    hit, miss = np.log(0.8), np.log(0.2 / (K - 1))
    row = {t: np.where(kk == t, hit, miss) for t in range(K)}
    for g in groups:
        for t in g: row[t] = np.where(np.isin(kk, g), hit, miss)
    for i, ps, m in zip(dev_ids, dev_preds, dev_multi):
        for p in ps:
            if p == 0: continue  # legit prediction: no evidence
            S[i - lo] += row[TYPES.index(LABELS[p])]
        if m: S[i - lo] += np.where(tk, 0.0, -8.0)  # multi-row new devices only occur in the takeover zones
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


def main():
    tx = load_all()
    tid = pl.read_csv(SAMPLE).select(c("id").alias("txn_id"))
    b = np.array(json.loads((ROOT / "oof" / "h6_meta.json").read_text())["bias"])
    T = apply(np.load(ROOT / "preds" / "h6_test.npy"), tid, tx, qr_null=True)
    pr = decode(T, b)
    ref = pl.read_csv(ROOT / "subs" / "h6_ppq.csv")["label"].to_numpy()
    assert (np.array(LABELS)[pr] == ref).all(), "decode does not reproduce subs/h6_ppq.csv"
    D = tid.join(tx.select("txn_id", "is_test", c("device_id").str.slice(2).cast(pl.Int64).alias("dnum")), on="txn_id", how="left", maintain_order="left")
    dnum, ist = D["dnum"].to_numpy(), D["is_test"].to_numpy()
    lo, hi = test_block(tx)
    g = (pl.DataFrame({"dnum": dnum, "p": pr}).filter(c("dnum").is_between(lo, hi - 1)).group_by("dnum")
         .agg(c("p"), (pl.len() > 1).alias("multi")).sort("dnum"))
    te = dp_edges(g["dnum"].to_list(), g["p"].to_list(), g["multi"].to_list(), lo, hi)
    print("test block", lo, hi, "\n  proportional edges", edges_for(lo, hi), "\n  DP edges          ", te)
    z = zone_ids(dnum, ist, te).astype(int)
    hz = z > 0
    full = np.where(hz, z, pr)
    SA = [LABEL2ID["sim_swap"], LABEL2ID["account_takeover"]]
    sa = hz & np.isin(pr, SA) & np.isin(z, SA)
    simato = np.where(sa, z, pr)
    print(f"zone rows {hz.sum()}  changed: full {(full != pr).sum()}  sim/ATO-only {(simato != pr).sum()}")
    print(pl.DataFrame({"from": np.array(LABELS)[pr[full != pr]], "to": np.array(LABELS)[full[full != pr]]})
          .group_by("from", "to").len().sort("len", descending=True))
    for lab, tag in ((full, "ts_event"), (simato, "ts_event_simato")):
        write_sub(np.eye(len(LABELS), dtype=np.float32)[lab], np.zeros(len(LABELS)), tag)


if __name__ == "__main__":
    main()
