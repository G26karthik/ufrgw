"""sc_final = v3_final + accepted class-sweep rules (13 Nov judge pass, see notes in sc_rules_*.py headers).

Accepted:
  RS  (sc_rules_collusion.py): merchant_collusion decode inside a 4-8 row round-set merchant segment -> refund_abuse. 3 rows.
      Same 3 rows as SEGR (sc_rules_refund.py); applied once.
  FCB (sc_rules_fc.py): bait-aware ph|qr zone edge (474352 -> 474360); qr_swap zone rows -> phishing_collect. 5 rows.
Rejected: SC1 (sc_rules_takeover.py, 1 row; test-spotted, EV ~0 at conservative loan odds), loan (identity).
usage (cwd=work): python sc_final.py
"""
import polars as pl
from common import ROOT, LABELS, load_all
import sc_rules_collusion as RS
import sc_rules_fc as FCB

c = pl.col


def diff_ids(a, b):
    return set(a.filter(a["label"] != b["label"])["id"])


if __name__ == "__main__":
    import sys
    inp, name = (sys.argv[1:3] + ["subs/v3_final.csv", "sc_final"][len(sys.argv[1:3]):])[:2]
    out = ROOT / "subs" / f"{name}.csv"
    assert not out.exists(), f"{out} exists; never overwrite"
    base = pl.read_csv(ROOT / inp)
    tx = load_all()
    cell = RS.segments(tx).filter(~c("is_test") & c("n").is_between(4, 8))
    assert cell.height > 300 and (cell["y"] == LABELS.index("merchant_collusion")).sum() == 0, "RS train cell must hold 0 collusion"

    rs, fcb = RS.apply(base, tx), FCB.apply(base, tx)
    d_rs, d_fc = diff_ids(base, rs), diff_ids(base, fcb)
    assert len(d_rs) == 3 and len(d_fc) == 5 and not (d_rs & d_fc), "rule sets must be 3 / 5 rows and disjoint"
    for ref, d in (("sc_collusion", d_rs), ("sc_fc", d_fc)):  # reproduce the class agents' candidate subs (when present)
        if (ROOT / "subs" / f"{ref}.csv").exists():
            assert d == diff_ids(base, pl.read_csv(ROOT / "subs" / f"{ref}.csv")), ref

    new = FCB.apply(RS.apply(base, tx), tx)
    assert diff_ids(base, new) == d_rs | d_fc
    assert new.height == base.height and new["id"].equals(base["id"]) and new["label"].is_in(LABELS).all()
    for name, d in (("RS", d_rs), ("FCB", d_fc)):
        ch = base.with_columns(new["label"].alias("to")).filter(c("id").is_in(list(d)))
        print(name, ch.height, ch.group_by("label", "to").len().rows(), sorted(d))
    new.write_csv(out)
    print("wrote", out)
