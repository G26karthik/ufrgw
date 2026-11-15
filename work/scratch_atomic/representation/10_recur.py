"""Agent A follow-up to 09: (a) which rows make the repeated same-class sender-receiver pairs inside the trace-free cohort;
(b) the cross-month P2P-receiver recurrence seen in cohort B (all home-device fraud labels) as a LABEL-INFORMED predicate:
    receiver_type == P2P and the receiver account held a fraud label (supplied train label) in an earlier month.
cwd=work, PYTHONPATH=.  python scratch_atomic/representation/10_recur.py > scratch_atomic/representation/10_out.txt
Freeze July -> August, check August -> September and July+August -> September. Rows = ALL train rows (not only home devices) and
home-device rows. The last block reads the benchmark-N final decode (held_N.parquet) ONLY as a reading aid: it shows how many
of the predicate's fraud rows the pipeline already finds; it is not used to define anything."""
import numpy as np
import polars as pl
from common import load_all, LABELS

c = pl.col; OUT = "scratch_atomic/representation/"
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(240)
tx = load_all().filter(~c("is_test") & c("month").is_in([7, 8, 9]))
q95 = float(tx.filter(c("month") == 7)["ip_risk_score"].quantile(.95))
num = lambda x: c(x).str.extract(r"(\d+)$").cast(pl.Int64, strict=False)
tx = tx.with_columns(
    (num("device_id") == num("sender_id")).fill_null(False).alias("home_dev"), (c("y") > 0).alias("fraud"),
    (~((c("txn_type") != "QR") & (c("qr_vpa_mismatch") == 0)).fill_null(False) & (c("app_version") >= 4) & (c("ip_risk_score") < q95)
     & (c("amount") != c("amount").round(0))).alias("tracefree"),
    pl.when(c("qr_vpa_mismatch").is_null()).then(pl.lit("null")).otherwise(c("qr_vpa_mismatch").cast(pl.Int8).cast(pl.String)).alias("mm"))

# (a) repeated pairs inside the trace-free cohort
A = tx.filter(c("home_dev") & c("tracefree") & c("fraud"))
rep = A.with_columns(pl.len().over("month", "sender_id", "receiver_id").alias("k")).filter(c("k") >= 2)
print("== (a) trace-free fraud rows on a sender-receiver pair with >= 2 such rows in the month:", rep.height, "rows")
print(rep.group_by("month", "txn_type", "receiver_type", "mm").agg(pl.len().alias("rows"), c("y").n_unique().alias("classes"), c("y").mode().first().alias("top_class"),
                                                                  (c("ts").max() - c("ts").min()).over("sender_id", "receiver_id").max().alias("x")).drop("x").sort("month", "rows", descending=[False, True]))

# (b) P2P receiver held a fraud label in an earlier month
held = pl.read_parquet(f"{OUT}held_N.parquet").select("txn_id", "q")
tx = tx.join(held, on="txn_id", how="left")
for scope, D in (("ALL train rows", tx), ("home-device rows", tx.filter(c("home_dev")))):
    print(f"\n######## (b) scope: {scope}")
    for src, tgt in (([7], 8), ([8], 9), ([7, 8], 9)):
        S = D.filter(c("month").is_in(src) & c("fraud") & (c("receiver_type") == "P2P"))["receiver_id"].unique()
        T = D.filter(c("month") == tgt); P = T.filter((c("receiver_type") == "P2P") & c("receiver_id").is_in(S.implode() if hasattr(S, "implode") else S))
        p2p = T.filter(c("receiver_type") == "P2P"); base, basep = T["fraud"].mean(), p2p["fraud"].mean()
        print(f"\n== source months {src} -> month {tgt}: flagged receivers {S.len()}; predicate rows {P.height}, fraud-labelled {int(P['fraud'].sum())} "
              f"(rate {P['fraud'].mean():.4f}); month base rate {base:.4f} (lift {P['fraud'].mean() / base:.2f}), P2P base rate {basep:.4f} (lift {P['fraud'].mean() / basep:.2f}); "
              f"coverage of the month's fraud labels {P['fraud'].sum() / T['fraud'].sum():.4f}, of P2P fraud labels {P['fraud'].sum() / p2p['fraud'].sum():.4f}")
        print("   predicate rows by txn_type:", P.group_by("txn_type").agg(pl.len().alias("n"), c("fraud").sum().alias("fraud"), c("fraud").mean().round(3).alias("rate")).sort("n", descending=True).rows())
        print("   fraud-labelled predicate rows by class:", {LABELS[k]: n for k, n in P.filter(c("fraud")).group_by("y").len().sort("y").rows()})
        # receiver in-degree control: fraud rate of non-flagged P2P rows whose receiver was ALSO seen as a P2P receiver in the source months
        seen = D.filter(c("month").is_in(src) & (c("receiver_type") == "P2P"))["receiver_id"].unique()
        Cn = p2p.filter(c("receiver_id").is_in(seen.implode() if hasattr(seen, "implode") else seen) & ~c("receiver_id").is_in(S.implode() if hasattr(S, "implode") else S))
        print(f"   control: P2P rows to receivers seen in the source months without a fraud label there: rows {Cn.height}, fraud rate {Cn['fraud'].mean():.4f}")
        # reading aid (uses the final decode of benchmark N)
        g = P.group_by("fraud", (c("q") > 0).alias("decoded_fraud")).len().sort("fraud", "decoded_fraud")
        print("   reading aid, benchmark-N final decode on predicate rows (fraud label, decoded fraud, rows):", g.rows())
        dl = P.filter(c("q") == 0); cl = Cn.filter(c("q") == 0); al = p2p.filter(c("q") == 0)
        print(f"   reading aid: among decoded-LEGIT rows, fraud-label rate: predicate rows {dl['fraud'].mean():.4f} (n {dl.height}, fraud {int(dl['fraud'].sum())}); "
              f"control rows {cl['fraud'].mean():.4f} (n {cl.height}); all P2P decoded-legit rows {al['fraud'].mean():.4f} (n {al.height})")
        # trace-free part of the predicate rows (no model output)
        tf = P.filter(c("tracefree") & c("home_dev")); tfa = p2p.filter(c("tracefree") & c("home_dev"))
        print(f"   trace-free home-device part: predicate rows {tf.height}, fraud-labelled {int(tf['fraud'].sum())} (rate {tf['fraud'].mean() if tf.height else float('nan'):.4f}); all trace-free P2P rows rate {tfa['fraud'].mean():.4f}")
