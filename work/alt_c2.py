"""Alternative for review: the c2s6 model in place of h6, with every accepted later rule layer of dd_final (run_all steps
12-17) and without step 11 (the discarded art_acct / art_zonemax transplant that made final1).

The step scripts sc_final.py and bc_final.py gate on row counts measured on the h6 base (LFM must change exactly 4 rows), so
they refuse another base. This driver calls the same rule functions in the same composition as those scripts:
  12 v3_rules.apply_rules            (QRN + zone propagation)
  13 FCB.apply(RS.apply(base))       (sc_final.py)
  14 CZ, LFM, X999 each on the step-13 output, diffs unioned, overlaps must agree   (bc_final.py)
  15 BANDV on the step-14 output     (bc_final.py)
  16 probe_rules.apply(.., "EDGEFIX")
  17 dd_rules_both.apply             (its expected-id assertions stay on)
Self-check: the same driver on subs/ts_judge_h6.csv must give the bytes of subs/dd_final.csv, or nothing is written.
usage (cwd=work, PYTHONPATH=.): python alt_c2.py [BASE=subs/ts_judge_c2s6.csv] [NAME=alt_c2s6_dd_final]
  writes subs/NAME.csv (never overwrites), notes/NAME_layers.csv (rows each layer changed), notes/NAME_diff.csv (rows that differ
  from dd_final)."""
import hashlib, io, sys
import polars as pl
import v3_rules, sc_rules_collusion as RS, sc_rules_fc as FCB
import bc_rules_layering, bc_rules_velocity, bc_rules_mule, bc_rules_phishing, probe_rules, dd_rules_both as DD
from common import ROOT, LABELS, load_all

c = pl.col
NR = (("CZ", bc_rules_layering.apply), ("LFM", bc_rules_velocity.apply), ("X999", bc_rules_phishing.apply))


def changed(a, b, layer):
    return a.with_columns(b["label"].alias("to"), pl.lit(layer).alias("layer")).filter(c("label") != c("to")).rename({"label": "from"})


def chain(base, tx):
    """-> (final sub, frame of every row a layer changed: id, from, to, layer)"""
    log = []

    def step(name, old, new):
        assert new["id"].equals(old["id"]), name
        log.append(changed(old, new, name)); return new.select("id", "label")

    s = step("v3", base, v3_rules.apply_rules(base, tx)[0])
    rs, fcb = RS.apply(s, tx), FCB.apply(s, tx)
    assert not set(changed(s, rs, "")["id"]) & set(changed(s, fcb, "")["id"]), "RS and FCB overlap"
    s = step("FCB", step("RS", s, rs), FCB.apply(rs, tx))
    d = pl.concat([changed(s, fn(s, tx), k) for k, fn in NR])  # each on the same base, as bc_final.py does
    clash = d.group_by("id").agg(c("to").n_unique().alias("n")).filter(c("n") > 1)
    assert clash.height == 0, f"rules disagree on {clash['id'].to_list()}"
    d = d.unique("id", keep="first", maintain_order=True); log.append(d); m = dict(zip(d["id"], d["to"]))
    s = s.with_columns(pl.Series("label", [m.get(i, l) for i, l in zip(s["id"], s["label"])]))
    s = step("BANDV", s, bc_rules_mule.apply(s, tx))
    s = step("EDGEFIX", s, probe_rules.apply(s, tx, "EDGEFIX"))
    s = step("DD", s, DD.apply(s, tx))
    assert s["label"].is_in(LABELS).all()
    return s, pl.concat(log)


def csv_bytes(df):
    b = io.BytesIO(); df.write_csv(b); return b.getvalue()


if __name__ == "__main__":
    inp, name = (sys.argv[1:3] + ["subs/ts_judge_c2s6.csv", "alt_c2s6_dd_final"][len(sys.argv[1:3]):])[:2]
    out = ROOT / "subs" / f"{name}.csv"
    assert not out.exists(), f"{out} exists; never overwrite"
    tx = load_all(); ref_bytes = (ROOT / "subs" / "dd_final.csv").read_bytes()
    h6, log_h6 = chain(pl.read_csv(ROOT / "subs" / "ts_judge_h6.csv"), tx)
    assert csv_bytes(h6) == ref_bytes, "self-check failed: the driver does not reproduce dd_final from ts_judge_h6"
    print("self-check: driver on ts_judge_h6 == subs/dd_final.csv (byte-identical)")
    base = pl.read_csv(ROOT / inp); new, log = chain(base, tx)
    assert new.height == h6.height and new["id"].equals(h6["id"])
    for k in log_h6["layer"].unique(maintain_order=True):
        a, b = log_h6.filter(c("layer") == k), log.filter(c("layer") == k)
        print(f"{k:8s} h6 base {a.height:3d} rows | this base {b.height:3d} rows | same rows and targets: {a.sort('id').equals(b.sort('id'))}"
              f" | only here: {sorted(set(b['id']) - set(a['id']))} | only h6: {sorted(set(a['id']) - set(b['id']))}")
    diff = h6.rename({"label": "dd_final"}).with_columns(new["label"].alias(name), base["label"].alias("base_before_rules"),
                                                         pl.read_csv(ROOT / "subs" / "ts_judge_h6.csv")["label"].alias("h6_before_rules")) \
             .filter(c("dd_final") != c(name))
    print("rows that differ from dd_final:", diff.height, diff.group_by("dd_final", name).len().sort("len", descending=True).rows())
    new.write_csv(out); log.write_csv(ROOT / "notes" / f"{name}_layers.csv"); diff.write_csv(ROOT / "notes" / f"{name}_diff.csv")
    print("wrote", out, "sha256", hashlib.sha256(out.read_bytes()).hexdigest())
