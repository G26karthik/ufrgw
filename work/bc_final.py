"""bc_final: accepted bc_* class-dive rules on top of subs/sc_final.csv (judge notes: notes/bc_final.md).

Each rule's apply() runs on sc_final itself, never on another rule's output: v3_rules.test_zones re-runs its edge DP on
the sub's own decode, so chaining could move zone edges. The diffs are then unioned; overlaps must agree on the target.
Accepted:
  CZ    bc_rules_layering  10 rows layering -> mule   (chain-wide mule-zone propagation; train label purity, no-regret)
  LFM   bc_rules_velocity   4 rows layering -> mule   (fan-in hosted by mule/vel zone; same 4 rows as bc_mule_b, band-free)
  BANDV bc_rules_mule      10 rows mule -> velocity   (host DV0473782 placed in velocity zone by device-id occupancy; a bet)
  X999  bc_rules_phishing   1 row  legit -> phishing  (x999 P2P amount, Oct-only np=0 cell)
Rejected: phishing MIX, mule variant b (subsumed by LFM), qr (identity).
usage (cwd=work): python bc_final.py [INP NAME RULES]   default: subs/sc_final.csv bc_final CZ,LFM,BANDV,X999
  e.g. python bc_final.py subs/sc_final.csv bc_nr CZ,LFM,X999   (no-regret part; BANDV is probed on its own upload)
"""
import polars as pl
from common import ROOT, LABELS, load_all
import bc_rules_layering, bc_rules_velocity, bc_rules_mule, bc_rules_phishing

c = pl.col
RULES = {  # key: (apply, existing candidate sub it must reproduce, expected (from, to, n))
    "CZ": (bc_rules_layering.apply, "bc_layering.csv", ("layering_chain", "mule_account", 10)),
    "LFM": (bc_rules_velocity.apply, "bc_velocity.csv", ("layering_chain", "mule_account", 4)),
    "BANDV": (bc_rules_mule.apply, "bc_mule.csv", ("mule_account", "velocity_bot", 10)),
    "X999": (bc_rules_phishing.apply, "bc_phishing.csv", ("legit", "phishing_collect", 1)),
}

if __name__ == "__main__":
    import sys
    inp, name, keys = (sys.argv[1:4] + ["subs/sc_final.csv", "bc_final", ",".join(RULES)][len(sys.argv[1:4]):])[:3]
    keys = keys.split(",")
    out = ROOT / "subs" / f"{name}.csv"
    assert not out.exists(), f"{out} exists; never overwrite"
    base = pl.read_csv(ROOT / inp)
    on_sc = inp.replace("\\", "/").endswith("subs/sc_final.csv")  # the class agents' candidate subs were built on sc_final
    tx = load_all()
    x = tx.filter(~c("is_test") & c("amount").is_in(bc_rules_phishing.X999) & (c("receiver_type") == "P2P"))
    assert x.height > 1000 and (x["label"] == "legit").mean() < 0.15, "train x999 P2P cell must stay fraud-dominated"
    diffs = []
    for k in keys:
        fn, ref, exp = RULES[k]
        new = fn(base, tx)
        assert new["id"].equals(base["id"]), k
        if on_sc and (ROOT / "subs" / ref).exists():
            assert new["label"].equals(pl.read_csv(ROOT / "subs" / ref)["label"]), f"{k} != {ref}"
        ch = base.with_columns(new["label"].alias("to")).filter(c("label") != c("to")).with_columns(pl.lit(k).alias("rule"))
        assert ch.group_by("label", "to").len().rows() == [exp], (k, ch.group_by("label", "to").len().rows())
        diffs.append(ch)
    d = pl.concat(diffs)
    clash = d.group_by("id").agg(c("to").n_unique().alias("nt")).filter(c("nt") > 1)
    assert clash.height == 0, f"rules disagree on {clash['id'].to_list()}"
    d = d.unique("id", keep="first", maintain_order=True)
    n = sum(RULES[k][2][2] for k in keys)
    assert d.height == n, (d.height, n)  # the rule sets are disjoint
    m = dict(zip(d["id"].to_list(), d["to"].to_list()))
    fin = base.with_columns(pl.Series("label", [m.get(i, s) for i, s in zip(base["id"].to_list(), base["label"].to_list())]))
    assert fin.height == base.height and fin["id"].equals(base["id"]) and fin["label"].is_in(LABELS).all()
    assert (fin["label"] != base["label"]).sum() == n
    print("rows changed per rule:", d.group_by("rule", "label", "to").len().sort("rule").rows())
    fin.write_csv(out)
    print("wrote", out, "rows changed", d.height)
