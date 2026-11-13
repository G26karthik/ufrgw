"""dd_stamp: STAMPREV rule (notes/dd_stamp.md). Label-free on test; no fitted parameter.

Stamp = non-QR row with qr_vpa_mismatch == 0. The fraud path keeps the mismatch flag of a QR base draw; the legit path never
sets it on a non-QR row (train: 0 flagged among 584,642 non-QR non-integer non-REFUND rows; legit-label share of flagged rows
= the label-noise rate). Two cells of stamped rows decoded legit -> phishing_collect:
  A: P2P PAY, integer amount <= 10, unanswered bait (ts_bait__rule == 1)        train ph share of fraud-labelled .96/.97/.97
  B: P2P COLLECT, integer amount, new_payee == 1, not a bait event (rule == 0)   train ph share of fraud-labelled .83/.85/.83
Gate (scratch_dd/stamp/gate.py): decode-conditioned OOF hits A 9 (5 ph), B 1 (1 ph); macro delta >= 0 on Aug/Sep noisy, cleaned, OOS.
usage (cwd=work): python dd_rules_stamp.py   -> subs/dd_stamp.csv (built on subs/probe_edgefix_bc_bandv.csv)
"""
import polars as pl
from common import ROOT, LABELS, load_all, run_feature

c = pl.col
PH = LABELS.index("phishing_collect")
RESERVED = ["AC015935", "AC279075"]  # planned fan-in probes: never touch
INT = c("amount") % 1 == 0
ST = ((c("txn_type") != "QR") & (c("qr_vpa_mismatch") == 0)).fill_null(False)
CA = ((c("receiver_type") == "P2P") & (c("txn_type") == "PAY") & INT & (c("amount") <= 10) & (c("rule") == 1)).fill_null(False)
CB = ((c("receiver_type") == "P2P") & (c("txn_type") == "COLLECT") & INT & (c("new_payee") == 1) & (c("rule") == 0)).fill_null(False)


def frame(tx):
    return tx.select("txn_id", "is_test", "month", "y", "sender_id", "receiver_id", "txn_type", "receiver_type", "amount", "new_payee", "qr_vpa_mismatch") \
        .join(run_feature("ts_bait", tx).select("txn_id", c("ts_bait__rule").fill_null(0).alias("rule")), on="txn_id", how="left", maintain_order="left")


def check_train(tx):
    tr = frame(tx).filter(~c("is_test") & c("month").is_in([7, 8, 9]))
    n_bad = tr.filter((c("txn_type") != "QR") & ~INT & (c("txn_type") != "REFUND") & ST).height
    print("train flagged non-QR non-integer non-REFUND rows:", n_bad)
    assert n_bad == 0, "stamp is no longer a pure fraud-path marker"
    for nm, cell, floor in (("A", CA, .9), ("B", CB, .8)):
        s = tr.filter(ST & cell & (c("y") > 0)).group_by("month").agg(pl.len().alias("n"), (c("y") == PH).mean().alias("ph")).sort("month")
        print(f"train cell {nm} fraud-labelled (n, phishing share) per month:", s.rows())
        assert s["month"].to_list() == [7, 8, 9] and (s["ph"] >= floor).all(), f"cell {nm} purity broke"


def apply(sub_df, tx):
    check_train(tx)
    D = sub_df.select(c("id").alias("txn_id"), "label").join(frame(tx), on="txn_id", how="left", maintain_order="left")
    assert D.height == sub_df.height and D["txn_id"].equals(sub_df["id"]) and D["is_test"].fill_null(False).all(), "test join broke"
    m = ST & (CA | CB) & (c("label") == "legit") & ~c("sender_id").is_in(RESERVED) & ~c("receiver_id").is_in(RESERVED)
    return D.select(c("txn_id").alias("id"), pl.when(m).then(pl.lit(LABELS[PH])).otherwise(c("label")).alias("label"))


if __name__ == "__main__":
    out = ROOT / "subs" / "dd_stamp.csv"
    assert not out.exists(), f"{out} exists; never overwrite"
    tx = load_all()
    base = pl.read_csv(ROOT / "subs" / "probe_edgefix_bc_bandv.csv")
    new = apply(base, tx)
    ch = base.with_columns(new["label"].alias("to")).filter(c("label") != c("to"))
    print("rows changed:", ch.height, ch.group_by("label", "to").len().rows(), ch["id"].to_list())
    assert new.height == base.height and new["id"].equals(base["id"]) and new["label"].is_in(LABELS).all()
    assert set(ch["id"].to_list()) == {"TX4a2a8f0c", "TXbeb95064", "TX4bde9ebc", "TXfe33f874"}, "changed set drifted"
    assert (ch["label"] == "legit").all() and (ch["to"] == LABELS[PH]).all()
    acc = tx.filter(c("txn_id").is_in(ch["id"].implode())).select("sender_id", "receiver_id")
    assert not (acc["sender_id"].is_in(RESERVED) | acc["receiver_id"].is_in(RESERVED)).any()
    new.write_csv(out)
    print("wrote", out)
