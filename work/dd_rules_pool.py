"""dd_pool: pool-bait rule (notes/dd_pool.md). Label-free on test except role accounts decoded by the pinned base sub.

An unanswered (ts_bait__rule == 1) Rs 1-10 integer P2P PAY that the sub decodes legit, sent by a known fraudster-pool account,
becomes phishing_collect. Pool F = velocity-pool senders B (device 420000-422999, all months) + fraud role accounts
(collector / bait sender / refunder / round-set buyer / qr receiver) from Jul-Sep labels and from the base sub's Oct decode.
  Train facts: tiny-int P2P PAY with B sender is 11-14% legit-labelled (= label-noise rate); outside B and unstamped ~70%.
  Gate (scratch_dd/pool/gate.py, judge decode, out of time): Sep fwd 13 rows 9 ph (Wilson90 .465), Aug strict 21 rows 10 ph,
  Aug leave-out 23 rows 12 ph; macro delta +.00025..+.00033 on noisy, cleaned and OOS. No fitted parameter.
usage (cwd=work): python dd_rules_pool.py   -> subs/dd_pool.csv (built on subs/probe_edgefix_bc_bandv.csv)
"""
import polars as pl
from common import ROOT, LABELS, LABEL2ID as L, load_all, run_feature

c = pl.col
BASE = ROOT / "subs" / "probe_edgefix_bc_bandv.csv"  # role decode pinned to this file whatever sub apply() gets
RESERVED = ["AC015935", "AC279075"]
PH = L["phishing_collect"]
p2p = c("receiver_type") == "P2P"; intg = c("amount") % 1 == 0
tiny = p2p & (c("txn_type") == "PAY") & intg & (c("amount") <= 10)
rs = c("amount").is_in([1000.0, 2000.0, 2500.0, 5000.0, 10000.0])
EXPECT = {"TXb8377124", "TXe4900616", "TXa9d6af1c", "TX0cec348b", "TX81feab4f", "TXe1644fe3", "TX621a3e6c", "TX4a2a8f0c",
          "TXbeb95064", "TX21726530", "TX212a458c"}


def pool_b(tx):
    return set(tx.filter(c("device_id").str.slice(2).cast(pl.Int64).is_between(420000, 422999))["sender_id"].to_list())


def roles(df, k):
    """df rows with class column k -> union of fraud role accounts."""
    return set().union(
        df.filter(p2p & (c("txn_type") == "COLLECT") & intg & c(k).is_in([L["phishing_collect"], L["fake_cashback"], L["loan_app_extortion"]]))["receiver_id"].to_list(),
        df.filter(tiny & c(k).is_in([L["phishing_collect"], L["fake_cashback"]]))["sender_id"].to_list(),
        df.filter((c("txn_type") == "REFUND") & (c(k) == L["refund_abuse"]))["sender_id"].to_list(),
        df.filter(~p2p & rs & c(k).is_in([L["qr_swap"], L["refund_abuse"], L["merchant_collusion"]]))["sender_id"].to_list(),
        df.filter(p2p & (c("txn_type") == "QR") & (c(k) == L["qr_swap"]))["receiver_id"].to_list())


def check_train(tx):
    B = pool_b(tx)
    s = (tx.filter(~c("is_test") & c("month").is_in([7, 8, 9]) & tiny)
         .with_columns(c("sender_id").is_in(list(B)).alias("inB"), ((c("txn_type") != "QR") & (c("qr_vpa_mismatch") == 0)).fill_null(False).alias("st"))
         .group_by("month").agg(pl.len().alias("n"), (c("y") == 0).filter(c("inB")).mean().alias("legit_B"),
                                (c("y") == 0).filter(~c("inB") & ~c("st")).mean().alias("legit_rest")).sort("month"))
    print("train tiny-int P2P PAY legit share (month, n, sender in B, outside B unstamped):", s.rows())
    assert s.height == 3 and (s["legit_B"] < .20).all() and (s["legit_rest"] > .50).all(), "pool-bait train purity broke"


def apply(sub_df, tx):
    check_train(tx)
    base = pl.read_csv(BASE).select(c("id").alias("txn_id"), c("label").replace_strict(L, return_dtype=pl.Int64).alias("qb"))
    te = tx.filter(c("is_test")).join(base, on="txn_id", how="left")
    F = pool_b(tx) | roles(tx.filter(~c("is_test") & c("month").is_in([7, 8, 9])), "y") | roles(te, "qb")
    print("pool F accounts:", len(F))
    bait = run_feature("ts_bait", tx).select("txn_id", c("ts_bait__rule").fill_null(0))
    hit = set(te.join(bait, on="txn_id", how="left")
              .filter(tiny & (c("ts_bait__rule") == 1) & c("sender_id").is_in(list(F))
                      & ~c("sender_id").is_in(RESERVED) & ~c("receiver_id").is_in(RESERVED))["txn_id"].to_list())
    m = sub_df["id"].is_in(list(hit)) & (sub_df["label"] == "legit")
    return sub_df.with_columns(pl.when(m).then(pl.lit(LABELS[PH])).otherwise(c("label")).alias("label"))


if __name__ == "__main__":
    out = ROOT / "subs" / "dd_pool.csv"
    assert not out.exists(), f"{out} exists; never overwrite"
    tx = load_all()
    base = pl.read_csv(BASE)
    new = apply(base, tx)
    ch = base.with_columns(new["label"].alias("to")).filter(c("label") != c("to"))
    print("rows changed:", ch.height, ch.group_by("label", "to").len().rows())
    print(ch.join(tx.select(c("txn_id").alias("id"), "sender_id", "receiver_id", "amount"), on="id", how="left"))
    assert new.height == base.height and new["id"].equals(base["id"]) and new["label"].is_in(LABELS).all()
    assert set(ch["id"].to_list()) == EXPECT and (ch["label"] == "legit").all() and (ch["to"] == LABELS[PH]).all()
    touched = tx.filter(c("txn_id").is_in(ch["id"].implode()))
    assert not (touched["sender_id"].is_in(RESERVED) | touched["receiver_id"].is_in(RESERVED)).any()
    new.write_csv(out)
    print("wrote", out)
