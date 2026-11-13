"""bc_phishing rules (notes/bc_phishing.md). Label-free: test features + the decoded sub itself.

X999 (SHIPPED): a P2P row with amount 1999/4999/9999/24999 decoded legit, off the zone block -> phishing_collect.
  Train: 1,236 such P2P rows: ph 918, legit 160 (13% = fraud->legit noise rate), fc 98, loan 38. No genuine legit x999 rows
  (legit P2P amounts are multiples of 10 or non-integer). Aug+Sep judge OOF decodes 0 of them legit.
  Oct-only cell: PAY P2P new_payee=0 x999 has 0 of 1.56M train rows and 1 Oct row, which the model decodes legit
  (p_legit .96: new_payee=0 is out of distribution). The sibling cell PAY P2P np=1 x999 is 30/31 ph in train and depleted
  in Oct (1 vs 5.2 expected), so it looks like a flipped new_payee flag (cf. QRN). Test: 1 row legit -> ph, EV ~+.00006.
MIX (NOT SHIPPED, mix=False): a rule-3 bait event (nf=1, nb=1, dt<=180 min) whose decode splits ph/fc -> both ph.
  Found on Sep (13 guarded rows: clean ph 10 / fc 2 / legit 1). With the zone guard, Aug has 0 such events.
  Aug off-zone rule-3 events with any row decoded ph are ph 5 / fc 4 rows. Break-even P(ph | ph or fc) = .645.
  Test: 7 rows fc -> ph, EV about -.0002..+.0004.
Checks: scratch_bc_phishing/chk_mix.py, chk_mix2.py, s15.py, s16.log.
usage (cwd=work): python bc_rules_phishing.py   -> subs/bc_phishing.csv from subs/sc_final.csv
"""
import polars as pl
from common import ROOT, LABELS, load_all, run_feature
from feats.ts_event import test_block

c = pl.col
PH, FC = "phishing_collect", "fake_cashback"
X999 = [1999.0, 4999.0, 9999.0, 24999.0]


def mix_ids(d, lo, hi):
    """d: txn_id, sender_id, receiver_id, receiver_type, amount, dn, rule, dec -> txn_ids to set to phishing.
    [lo, hi): zone-block device range; events with any row there are skipped."""
    tiny = (c("receiver_type") == "P2P") & (c("amount") <= 10) & (c("amount") == c("amount").round(0))
    b = d.filter(c("rule") == 3).with_columns(
        pl.when(tiny).then(pl.concat_str("sender_id", "receiver_id", separator="|"))
        .otherwise(pl.concat_str("receiver_id", "sender_id", separator="|")).alias("ev"))
    e = b.group_by("ev").agg((c("dec") == PH).sum().alias("nph"), (c("dec") == FC).sum().alias("nfc"), pl.len().alias("n"),
                             c("dn").is_between(lo, hi - 1).any().alias("zb"))
    ev = e.filter((c("nph") >= 1) & (c("nfc") >= 1) & (c("nph") + c("nfc") == c("n")) & ~c("zb"))["ev"]
    return b.filter(c("ev").is_in(ev.implode()) & (c("dec") == FC))["txn_id"]


def x999_ids(d, lo, hi):
    return d.filter(c("amount").is_in(X999), c("receiver_type") == "P2P", c("dec") == "legit",
                    ~c("dn").is_between(lo, hi - 1))["txn_id"]


def frame(ids, tx):
    """ids (txn_id Series, sample order) -> rule inputs aligned to ids."""
    return ids.to_frame("txn_id").join(
        tx.select("txn_id", "sender_id", "receiver_id", "receiver_type", "amount",
                  c("device_id").str.slice(2).cast(pl.Int64).alias("dn")).join(
            run_feature("ts_bait", tx).select("txn_id", c("ts_bait__rule").alias("rule")), on="txn_id"),
        on="txn_id", how="left", maintain_order="left")


def apply(sub_df, tx, mix=False):
    """sub_df: id,label (sample order) -> sub_df with X999 (and MIX if mix=True) applied."""
    d = frame(sub_df["id"].alias("txn_id"), tx).with_columns(sub_df["label"].alias("dec"))
    lo, hi = test_block(tx)
    hit = pl.concat([x999_ids(d, lo, hi)] + ([mix_ids(d, lo, hi)] if mix else []))
    return sub_df.with_columns(pl.when(c("id").is_in(hit.implode())).then(pl.lit(PH)).otherwise(c("label")).alias("label"))


if __name__ == "__main__":
    out = ROOT / "subs" / "bc_phishing.csv"
    assert not out.exists(), f"{out} exists; never overwrite"
    base = pl.read_csv(ROOT / "subs" / "sc_final.csv")
    tx = load_all()
    tr = tx.filter(~c("is_test") & c("amount").is_in(X999) & (c("receiver_type") == "P2P"))
    assert tr.height > 1000 and (tr["label"] == "legit").mean() < 0.15, "train x999 P2P cell must stay fraud-dominated"
    new = apply(base, tx)
    ch = base.with_columns(new["label"].alias("to")).filter(c("label") != c("to"))
    assert new["id"].equals(base["id"]) and new["label"].is_in(LABELS).all()
    assert (ch["label"] == "legit").all() and (ch["to"] == PH).all() and ch.height == 1, ch
    assert apply(base, tx, mix=True).filter(c("label") != base["label"]).height == 8  # MIX adds 7 fc -> ph (not shipped)
    print("X999", ch.height, "rows legit -> ph:", sorted(ch["id"]))
    new.write_csv(out)
    print("wrote", out)
