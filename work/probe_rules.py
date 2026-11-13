"""Single-event / edge-band LB probe bets on top of a decoded sub (evidence and ranking: notes/probe_bets.md, scratch_probe/).

Rows are picked by explicit device id, NOT by re-running test_zones: relabelled zone rows would move the DP edges.
  MULEBAND  11 rows velocity_bot -> mule_account: every zone row on devices [473749, 473783) (the old disputed band).
            Only for the bandv < nr outcome: if the DV0473782-hosted fan-in (receiver AC369510) is mule, contiguity puts the
            whole band in the mule zone. Needs that fan-in to be mule in the base, so it never applies on top of bc_bandv.
  EDGEFIX    3 rows, each its own zone edge (length-prior edge posterior, scratch_probe/p3_post.py; sub-threshold, not one event):
            474281, 474283 account_takeover -> phishing_collect (ATO|ph edge 474285 -> 474281);
            473917 sim_swap -> layering_chain (lay|sim edge 473916 -> 473920).
usage (cwd=work): python probe_rules.py KEY [BASE=subs/bc_nr.csv]   -> subs/probe_<key>.csv (bc_nr base) or
                  subs/probe_<key>_<base>.csv (other base)
"""
import polars as pl
from common import ROOT, LABELS, load_all

c = pl.col
BETS = {  # key: ({device id: (from, to)} or ((lo, hi), from, to))
    "MULEBAND": ((473749, 473783), "velocity_bot", "mule_account"),
    "EDGEFIX": {474281: ("account_takeover", "phishing_collect"), 474283: ("account_takeover", "phishing_collect"),
                473917: ("sim_swap", "layering_chain")},
}
# Hostless mule-decoded fan-ins (no zone row pays the receiver within +-48 h, no new_payee=0 row): train Aug/Sep
# velocity-majority 9/63 (.14; size 15-19: 5/21). PROBE ONLY, never ship blind (break-even .39). scratch_probe/p12_hostless.py
HOSTLESS = {
    "HL_AC015935": ["TX1be92637", "TX418a73f5", "TX7c43fc8b", "TX57871ef4", "TXcb924df8", "TX071f1a4b", "TXd31be239",
                    "TX769aeae2", "TX9bef437d", "TX37340d51", "TX28f5b09b", "TX0012b626", "TX9c1128bf", "TX4145b368",
                    "TX844b0014", "TX2dc14976", "TXd02a715f", "TX216c2fcd", "TX30eb9c1c"],
    "HL_AC279075": ["TX4af8df28", "TXafa22a45", "TX76a89608", "TX953ceab8", "TX677e44f3", "TX2ff4212c", "TX756031c8",
                    "TXdd934a40", "TXb5a8a8ec", "TX3064cdbe", "TXbace7b07", "TX2d309cdb", "TX2c785c72", "TXbd83f088",
                    "TX4e0dfcbb", "TX79803c16", "TX9ee358f1", "TXf7c5eaef"],
}
NROWS = {"MULEBAND": 11, "EDGEFIX": 3} | {k: len(v) for k, v in HOSTLESS.items()}


def apply(sub_df, tx, key):
    """sub_df: id,label (sample order) -> new sub_df with the KEY bet applied."""
    D = sub_df.join(tx.select(c("txn_id").alias("id"), "receiver_id", c("device_id").str.slice(2).cast(pl.Int64).alias("dn")),
                    on="id", how="left", maintain_order="left")
    assert D["dn"].null_count() == 0 and D["id"].equals(sub_df["id"])
    if key in HOSTLESS:  # whole fan-in segment mule -> velocity
        m = c("id").is_in(HOSTLESS[key])
        r = D.filter(m)
        assert r.height == len(HOSTLESS[key]) and (r["label"] == "mule_account").all() and r["receiver_id"].n_unique() == 1, key
        out = D.select("id", pl.when(m).then(pl.lit("velocity_bot")).otherwise(c("label")).alias("label"))
        assert (out["label"] != sub_df["label"]).sum() == NROWS[key], key
        return out
    b = BETS[key]
    if key == "MULEBAND":
        (lo, hi), f, t = b
        m = c("dn").is_between(lo, hi - 1)
        fan = D.filter((c("receiver_id") == "AC369510") & ~m)["label"]  # off-band rows of the 473782-hosted fan-in
        assert (fan == "mule_account").sum() == 8 and not (fan == "velocity_bot").any(), \
            "MULEBAND needs the AC369510 fan-in mule (bandv < nr base, never bc_bandv)"
        assert (D.filter(m)["label"] == f).all(), "band rows must all be zone-overridden velocity in the base"
        new = pl.when(m).then(pl.lit(t)).otherwise(c("label"))
    else:
        for dn, (f, _) in b.items():
            r = D.filter(c("dn") == dn)
            assert r.height == 1 and r["label"][0] == f, (dn, r["label"].to_list())
        new = c("label")
        for dn, (_, t) in b.items():
            new = pl.when(c("dn") == dn).then(pl.lit(t)).otherwise(new)
    out = D.select("id", new.alias("label"))
    assert (out["label"] != sub_df["label"]).sum() == NROWS[key], key
    return out


if __name__ == "__main__":
    import sys
    key = sys.argv[1]
    inp = sys.argv[2] if len(sys.argv) > 2 else "subs/bc_nr.csv"
    stem = inp.replace("\\", "/").split("/")[-1].removesuffix(".csv")
    out = ROOT / "subs" / (f"probe_{key.lower()}.csv" if stem == "bc_nr" else f"probe_{key.lower()}_{stem}.csv")
    assert not out.exists(), f"{out} exists; never overwrite"
    base = pl.read_csv(ROOT / inp)
    new = apply(base, load_all(), key)
    assert new.height == base.height and new["id"].equals(base["id"]) and new["label"].is_in(LABELS).all()
    d = base.with_columns(new["label"].alias("to")).filter(c("label") != c("to"))
    # disjoint from the 15 no-regret rows (sc_final -> bc_nr) and the 10 BANDV rows (bc_nr -> bc_bandv)
    rd = lambda a, b: set(pl.read_csv(ROOT / "subs" / a).with_columns(pl.read_csv(ROOT / "subs" / b)["label"].alias("x"))
                          .filter(c("label") != c("x"))["id"].to_list())
    nr15, bv10 = rd("sc_final.csv", "bc_nr.csv"), rd("bc_nr.csv", "bc_bandv.csv")
    assert len(nr15) == 15 and len(bv10) == 10 and not (set(d["id"].to_list()) & (nr15 | bv10))
    print(key, "on", inp, "rows changed", d.height, d.group_by("label", "to").len().rows())
    print(d["id"].to_list())
    new.write_csv(out)
    print("wrote", out)
