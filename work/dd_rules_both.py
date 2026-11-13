"""dd_both: union of the pool-bait rule (P) and the stamp rule (S); every hit -> phishing_collect. Label-free on test.

P (pool-bait): legit-decoded tiny-int P2P PAY in an unanswered bait event (ts_bait rule == 1) whose sender is a known
  fraud-actor account: a velocity-pool device sender (DV420000-422999, any month) or a role account (collector / bait sender /
  refunder / round-set buyer / qr receiver) from Jul-Sep train labels or the test base decode.
S (stamp): legit-decoded stamped row (non-QR, qr_vpa_mismatch == 0) in cell A (tiny-int P2P PAY, rule == 1) or
  cell B (int P2P COLLECT, new_payee == 1, rule == 0).
Both rules read the ORIGINAL base labels (single pass), so order does not matter. Gate: scratch_dd/both/gate.py, notes/dd_both.md.
usage (cwd=work): python dd_rules_both.py   -> subs/dd_both.csv (built on subs/probe_edgefix_bc_bandv.csv)
"""
import polars as pl
from common import ROOT, LABELS, LABEL2ID as L, load_all, run_feature

c = pl.col
PH = "phishing_collect"
RESERVED = ["AC015935", "AC279075"]
P2P, INT = c("receiver_type") == "P2P", c("amount") % 1 == 0
TINY = P2P & (c("txn_type") == "PAY") & INT & (c("amount") <= 10)
ST = ((c("txn_type") != "QR") & (c("qr_vpa_mismatch") == 0)).fill_null(False)
POOL = c("dnum").is_between(420000, 422999)
CELL_A = ST & TINY & (c("rule") == 1)
CELL_B = ST & P2P & (c("txn_type") == "COLLECT") & INT & (c("new_payee") == 1) & (c("rule") == 0)
# expected test hits per disjoint group; dropping a group from KEEP is the fallback filter
EXPECT = {
    "P_only": {"TXb8377124", "TXe4900616", "TXa9d6af1c", "TX0cec348b", "TX81feab4f", "TXe1644fe3", "TX621a3e6c", "TX21726530", "TX212a458c"},
    "P_and_S": {"TX4a2a8f0c", "TXbeb95064"},
    "S_only": {"TX4bde9ebc", "TXfe33f874"},
}
KEEP = ("P_only", "P_and_S", "S_only")


def frame(tx):
    return tx.select("txn_id", "sender_id", "receiver_id", "receiver_type", "txn_type", "amount", "new_payee", "qr_vpa_mismatch",
                     "month", "is_test", "y", c("device_id").str.slice(2).cast(pl.Int64).alias("dnum")).join(
        run_feature("ts_bait", tx).select("txn_id", c("ts_bait__rule").fill_null(0).alias("rule")), on="txn_id", how="left")


def roles(df, k):
    """role accounts from rows df with class column k (label or decode)."""
    s = lambda m, col: set(df.filter(m)[col].to_list())
    return (s(P2P & (c("txn_type") == "COLLECT") & INT & c(k).is_in([6, 9, 10]), "receiver_id")   # collector
            | s(TINY & c(k).is_in([6, 9]), "sender_id")                                            # bait sender
            | s((c("txn_type") == "REFUND") & (c(k) == 8), "sender_id")                            # refunder
            | s(~P2P & c("amount").is_in([1000.0, 2000.0, 2500.0, 5000.0, 10000.0]) & c(k).is_in([7, 8, 11]), "sender_id")  # round-set buyer
            | s(P2P & (c("txn_type") == "QR") & (c(k) == 7), "receiver_id"))                       # qr receiver


def actors(f, lab_rows, dec_rows, pool_rows):
    """B = velocity-pool senders, F = B | role accounts (label part from lab_rows.y, decode part from dec_rows.q)."""
    B = set(pool_rows.filter(POOL)["sender_id"].to_list())
    return B, B | roles(lab_rows, "y") | roles(dec_rows, "q")


def check_train(tx):
    f = frame(tx)
    tr = f.filter(~c("is_test") & c("month").is_in([7, 8, 9]))
    B, F = actors(f, tr, tr.head(0).with_columns(c("y").alias("q")), f)
    flag = tr.filter(CELL_A | CELL_B | (TINY & (c("rule") == 1) & c("sender_id").is_in(list(F))))
    odd = flag.filter((c("txn_type") != "QR") & (c("amount") % 1 != 0) & (c("txn_type") != "REFUND")).height
    s = tr.filter(TINY & c("sender_id").is_in(list(B))).group_by("month").agg(pl.len().alias("n"), (c("y") == 0).mean().alias("legit")).sort("month")
    pa = tr.filter(CELL_A & (c("y") > 0))["y"].eq(6).mean()
    pb = tr.filter(CELL_B & (c("y") > 0))["y"].eq(6).mean()
    print(f"train: odd flagged {odd} | pool-sender tiny rows (month, n, legit share) {s.rows()} | phishing share of fraud rows cell A {pa:.3f} cell B {pb:.3f}")
    assert odd == 0, "flag reached non-integer non-QR non-REFUND rows"
    assert s.height == 3 and (s["legit"] < .20).all(), "pool-sender tiny rows no longer fraud-dominated"
    assert pa >= .9 and pb >= .8, "stamp cells lost phishing purity"


def hits(f):
    """f: frame() rows with base decode q (test) -> (P mask, S mask) over the q==0 rows of f where is_test."""
    tr = f.filter(~c("is_test") & c("month").is_in([7, 8, 9]))
    B, F = actors(f, tr, f.filter(c("is_test")), f)
    cand = c("is_test") & (c("q") == 0)
    return f.with_columns((cand & TINY & (c("rule") == 1) & c("sender_id").is_in(list(F))).alias("in_P"),
                          (cand & (CELL_A | CELL_B)).alias("in_S"), c("sender_id").is_in(list(B)).alias("in_B")), len(F)


def rows(sub_df, tx):
    """test hit rows (kept groups) with in_P, in_S, in_B, grp; asserts the expected ids per group."""
    q = sub_df.select(c("id").alias("txn_id"), c("label").replace_strict(L, return_dtype=pl.Int64).alias("q"))
    f, nF = hits(frame(tx).join(q, on="txn_id", how="left"))
    print("F size", nF)
    h = f.filter(c("in_P") | c("in_S")).with_columns(
        pl.when(c("in_P") & c("in_S")).then(pl.lit("P_and_S")).when(c("in_P")).then(pl.lit("P_only")).otherwise(pl.lit("S_only")).alias("grp"))
    for g in EXPECT:
        got = set(h.filter(c("grp") == g)["txn_id"].to_list())
        assert got == EXPECT[g], (g, sorted(got ^ EXPECT[g]))
    h = h.filter(c("grp").is_in(list(KEEP)))
    assert h.filter(c("sender_id").is_in(RESERVED) | c("receiver_id").is_in(RESERVED)).height == 0
    return h


def apply(sub_df, tx):
    check_train(tx)
    ids = rows(sub_df, tx)["txn_id"].implode()
    return sub_df.with_columns(pl.when(c("id").is_in(ids)).then(pl.lit(PH)).otherwise(c("label")).alias("label"))


if __name__ == "__main__":
    out = ROOT / "subs" / "dd_both.csv"
    assert not out.exists(), f"{out} exists; never overwrite"
    base = pl.read_csv(ROOT / "subs" / "probe_edgefix_bc_bandv.csv")
    tx = load_all()
    new, h = apply(base, tx), rows(base, tx)
    (ROOT / "scratch_dd" / "both").mkdir(parents=True, exist_ok=True)
    h.sort("grp", "txn_id").select("txn_id", "in_P", "in_S", "in_B", "amount", "grp").write_csv(ROOT / "scratch_dd" / "both" / "rows.csv")
    ch = base.with_columns(new["label"].alias("to")).filter(c("label") != c("to"))
    want = set().union(*(EXPECT[g] for g in KEEP))
    print("rows changed:", ch.height, ch.group_by("label", "to").len().rows(), sorted(ch["id"].to_list()))
    assert new.height == base.height and new["id"].equals(base["id"]) and new["label"].is_in(LABELS).all()
    assert set(ch["id"].to_list()) == want and (ch["label"] == "legit").all() and (ch["to"] == PH).all()
    new.write_csv(out)
    print("wrote", out)
