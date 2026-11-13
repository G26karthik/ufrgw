"""bc_velocity: velocity_bot class sweep rule (notes/bc_velocity.md). Label-free on test; edge-independent for the mule|vel cut.

LFM: an off-zone row decoded layering_chain whose P2P receiver also gets a row from a mule-zone or velocity-zone fresh device
within +-48 h, and gets no layering-zone row in that window -> mule_account.
  Train (label-fit zones): label-only layering share among linked rows .5% / .8% / .3% (Jul/Aug/Sep). Judge-decoded layering
  hits: Aug 8 (cleaned 6 mule, 1 vel, 1 legit), Sep 4 (4 mule), 0 layering. Window chosen on Aug, checked on Sep.
  Target mule (not velocity): the host's mule|vel side is the disputed band; mule is positive under both resolutions.
usage (cwd=work): python bc_rules_velocity.py   -> subs/bc_velocity.csv (built on subs/sc_final.csv)
"""
import numpy as np
import polars as pl
from common import ROOT, LABELS, LABEL2ID as L, load_all
from feats.ts_event import zone_ids
import v3_rules as V3

c = pl.col
W = 2 * 86400
MU, VE, LA = L["mule_account"], L["velocity_bot"], L["layering_chain"]


def hits(rows, z):
    """rows: txn_id, ts, receiver_id, dn (+ q decode); z zone id per row -> txn_ids of off-zone rows linked to mule/vel zone rows only."""
    D = rows.with_columns(pl.Series("z", z), pl.when(c("receiver_id").str.starts_with("AC")).then(c("receiver_id")).alias("rn")) \
            .with_columns(pl.when(c("dn").is_between(420000, 422999)).then(0).otherwise(c("z")).alias("z"))  # pool never sends P2P
    Z = D.filter(c("z").is_in([MU, VE, LA]) & c("rn").is_not_null()).select("rn", c("ts").alias("zts"), c("z").alias("zz"))
    g = (D.filter((c("z") == 0) & c("rn").is_not_null()).select("txn_id", "rn", "ts").join(Z, on="rn")
         .filter((c("ts") - c("zts")).abs() <= W).group_by("txn_id").agg((c("zz") == LA).any().alias("has_lay")))
    return g.filter(~c("has_lay"))["txn_id"]


def check_train(tx):
    tr = tx.filter(~c("is_test")).select("txn_id", "ts", "receiver_id", "y", "month", c("device_id").str.slice(2).cast(pl.Int64).alias("dn"))
    h = tr.filter(c("txn_id").is_in(hits(tr, zone_ids(tr["dn"].to_numpy(), np.zeros(tr.height, bool), [10**9])).implode()))
    s = h.filter(c("y").is_in([MU, VE, LA]) & c("month").is_in([7, 8, 9])).group_by("month").agg(pl.len().alias("n"), (c("y") == LA).mean().alias("lay")).sort("month")
    print("train label-only linked rows (n, layering share) per month:", s.rows())
    assert s.height == 3 and (s["n"] > 300).all() and (s["lay"] < .02).all(), "LFM train purity broke"


def apply(sub_df, tx):
    check_train(tx)
    D = sub_df.select(c("id").alias("txn_id")).join(
        tx.select("txn_id", "ts", "receiver_id", c("device_id").str.slice(2).cast(pl.Int64).alias("dn")), on="txn_id", how="left", maintain_order="left")
    q = np.array([L[s] for s in sub_df["label"].to_list()])
    z, edges = V3.test_zones(D, q, tx)
    print("test edges", list(edges))
    h = set(hits(D, z).to_list())
    m = np.array([t in h for t in D["txn_id"].to_list()]) & (q == LA)
    lab = sub_df["label"].to_numpy().copy(); lab[m] = "mule_account"
    return sub_df.with_columns(pl.Series("label", lab))


if __name__ == "__main__":
    out = ROOT / "subs" / "bc_velocity.csv"
    assert not out.exists(), f"{out} exists; never overwrite"
    base = pl.read_csv(ROOT / "subs" / "sc_final.csv")
    new = apply(base, load_all())
    ch = base.with_columns(new["label"].alias("to")).filter(c("label") != c("to"))
    print("rows changed:", ch.height, ch.group_by("label", "to").len().rows(), ch["id"].to_list())
    assert new.height == base.height and new["id"].equals(base["id"]) and new["label"].is_in(LABELS).all()
    assert ch.height <= 6 and (ch["label"] == "layering_chain").all()
    new.write_csv(out)
    print("wrote", out)
