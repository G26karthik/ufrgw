"""bc_layering: chain-wide zone propagation (CZ) for layering hop chains (notes/bc_layering.md).

Rule (label-free): hop components of size >= 3 (edges as feats/flow.py: P2P A->B then B->C within 6 h, amount ratio in
[0.9, 1), middle account rin_n3h <= 3). If the chain's zone-device rows (zone type in {mule, layering}, device not in
the disputed v3 BAND) carry exactly one zone type, every off-zone chain row decoded in that group takes it.
Generalises v3 R3 (direct +-1 h account link to a zone row) to the whole chain: the generator labels a chain as one event
(train chain label purity .99), and the zone type matched the chain label in every train chain (Jul-Sep, notes).
Test: 10 rows layering_chain -> mule_account (4 chains with a mule-zone device, all far below BAND).
usage (cwd=work): python bc_rules_layering.py   -> subs/bc_layering.csv from subs/sc_final.csv
"""
import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from common import ROOT, LABELS, LABEL2ID as L, load_all, run_feature
from v3_rules import BAND, G1, test_zones

c = pl.col
H = 3600
SRC = [L["mule_account"], L["layering_chain"]]  # no velocity sources: train has no velocity chains (the Sep TR-edge velocity hit is an edge error)


def chains(tx, min_size=3):
    """txn_id -> comp for rows in hop components of size >= min_size (whole data, as flow.py)."""
    t = tx.select("txn_id", "sender_id", "receiver_id", "receiver_type", "ts", "amount").join(
        run_feature("flow", tx).select("txn_id", c("flow__rin_n3h").alias("rin3")), on="txn_id", how="left").with_row_index("i")
    p = t.filter(c("receiver_type") == "P2P").select("i", "sender_id", "receiver_id", "ts", "amount", "rin3")
    e = p.join(p, left_on="receiver_id", right_on="sender_id", suffix="2").filter(
        (c("ts2") > c("ts")) & (c("ts2") - c("ts") <= 6 * H) & (c("amount2") / c("amount")).is_between(0.9, 1.0, closed="left")
        & (c("rin3") <= 3))
    i1, i2 = e["i"].to_numpy(), e["i2"].to_numpy()
    n = t.height
    _, comp = connected_components(coo_matrix((np.ones(len(i1)), (i1, i2)), shape=(n, n)), directed=False)
    inE = np.zeros(n, bool); inE[i1] = True; inE[i2] = True
    out = t.select("txn_id").with_columns(pl.Series("comp", comp)).filter(pl.Series(inE))
    return out.filter(pl.len().over("comp") >= min_size)


def cz_hits(D, q, z, band=BAND):
    """D: rows (txn_id, dn, comp) aligned with decode q and zone ids z -> DataFrame txn_id, zz (new label id)."""
    D = D.with_columns(pl.Series("q", q), pl.Series("z", z)).filter(c("comp").is_not_null())
    inband = pl.any_horizontal([c("dn").is_between(lo, hi - 1) for lo, hi in band]) if band else pl.lit(False)
    src = D.filter(c("z").is_in(SRC) & ~inband).group_by("comp").agg(c("z").unique().alias("zs"))
    src = src.filter(c("zs").list.len() == 1).select("comp", c("zs").list.first().alias("zz"))
    return (D.filter((c("z") == 0) & c("q").is_in(G1)).join(src, on="comp").filter(c("zz") != c("q")).select("txn_id", "zz"))


def apply(sub_df, tx, edges=None):
    """sub_df: id,label (sample order) -> new sub with CZ applied. Zones: v3_rules.test_zones on the sub's own decode."""
    D = sub_df.select(c("id").alias("txn_id")).join(
        tx.select("txn_id", c("device_id").str.slice(2).cast(pl.Int64).alias("dn")), on="txn_id", how="left", maintain_order="left"
    ).join(chains(tx), on="txn_id", how="left", maintain_order="left")
    assert D["dn"].null_count() == 0
    q = np.array([L[s] for s in sub_df["label"].to_list()])
    z, _ = test_zones(D, q, tx, edges)
    h = cz_hits(D, q, z)
    m = dict(zip(h["txn_id"].to_list(), h["zz"].to_list()))
    lab = [LABELS[m[t]] if t in m else s for t, s in zip(sub_df["id"].to_list(), sub_df["label"].to_list())]
    return sub_df.with_columns(pl.Series("label", lab))


if __name__ == "__main__":
    out = ROOT / "subs" / "bc_layering.csv"
    assert not out.exists(), f"{out} exists; never overwrite"
    base = pl.read_csv(ROOT / "subs" / "sc_final.csv")
    new = apply(base, load_all())
    ch = base.with_columns(new["label"].alias("to")).filter(c("label") != c("to"))
    print("rows changed:", ch.height, ch.group_by("label", "to").len().rows())
    assert ch.height == 10 and ch.group_by("label", "to").len().rows() == [("layering_chain", "mule_account", 10)]
    assert new.height == base.height and new["id"].equals(base["id"]) and new["label"].is_in(LABELS).all()
    new.write_csv(out)
    print("wrote", out)
