"""Which fields does the bait-family recipe set? Compare bait / payback rows with legit P2P PAY rows. cwd=work."""
import polars as pl, numpy as np
from common import load_all
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260)
c = pl.col
tx = load_all().filter(~c("is_test"))
print(tx.columns)
p2p = tx.filter(c("receiver_type") == "P2P").with_columns(
    pl.min_horizontal("sender_id", "receiver_id").alias("a"), pl.max_horizontal("sender_id", "receiver_id").alias("b"),
    ((c("txn_type") == "PAY") & (c("amount") <= 10) & (c("amount") % 1 == 0)).alias("bait"))
ev = p2p.filter(c("y").is_in([6, 9])).group_by("a", "b").agg(c("bait").sum().alias("nb"), c("y").mode().first().alias("fam")).filter(c("nb") >= 1)
r = p2p.join(ev, on=["a", "b"]).with_columns(pl.when(c("bait")).then(pl.lit("bait")).otherwise(pl.lit("payback")).alias("role"))
ref = tx.filter((c("receiver_type") == "P2P") & (c("txn_type") == "PAY") & (c("y") == 0)).with_columns(pl.lit(0).cast(pl.Int8).alias("fam"), pl.lit("legitP2P").alias("role"))
cols = ["fam", "role"]
both = pl.concat([r.select(cols + tx.columns), ref.select(cols + tx.columns)])
dn = c("device_id").str.slice(2).cast(pl.Int64); sn = c("sender_id").str.slice(2).cast(pl.Int64)
print(both.group_by("fam", "role").agg(
    pl.len(), (c("y") == 0).mean().alias("lab_legit"), c("new_payee").mean(), c("pin_retries").mean(), c("login_to_txn_sec").median().alias("login_med"),
    c("sim_age_hours").median().alias("sim_med"), c("device_age_days").median().alias("dev_med"), (c("device_city") != c("snd_home_city")).mean().alias("away"),
    (dn == sn).mean().alias("home_dev"), (dn >= 420000).mean().alias("fresh"), c("timestamp").dt.hour().is_between(0, 5).mean().alias("night"),
    (c("amount") / c("balance_before")).median().alias("amt_bal_med"), (c("amount") > c("balance_before")).mean().alias("amt_gt_bal"),
    c("ip_risk_score").mean().alias("ip"), c("amount").median().alias("amt_med"), (c("amount") % 1 == 0).mean().alias("int"),
    (c("txn_type") == "COLLECT").mean().alias("collect")).sort("fam", "role"))
# event-level: gaps
e = r.sort("ts").group_by("a", "b").agg(c("fam").first(), c("nb").first(), pl.len().alias("n"), c("ts").filter(c("bait")).min().alias("tb0"), c("ts").filter(c("bait")).max().alias("tb1"),
                                           c("ts").filter(~c("bait")).min().alias("tp"), c("amount").filter(~c("bait")).first().alias("pamt"), c("amount").filter(c("bait")).alias("bamts"),
                                           (c("sender_id").filter(c("bait")).first() == c("receiver_id").filter(~c("bait")).first()).alias("pay_to_baiter"),
                                           (c("ts").max() - c("ts").min()).alias("span"), c("month").first())
e = e.with_columns(((c("tp") - c("tb1")) / 3600).alias("delay_h"), (c("tb1") - c("tb0")).alias("bgap_s"))
print(e.group_by("fam", "nb", c("n").clip(upper_bound=4)).agg(pl.len(), c("delay_h").quantile(.05).alias("d05"), c("delay_h").median().alias("d50"), c("delay_h").quantile(.95).alias("d95"), c("delay_h").max().alias("dmax"),
      c("bgap_s").median().alias("bgap50"), c("bgap_s").max().alias("bgapmax"), c("pamt").quantile(.05).alias("a05"), c("pamt").median().alias("a50"), c("pamt").quantile(.95).alias("a95"), c("pay_to_baiter").mean()).sort("fam", "nb", "n"))
x = e.filter(c("delay_h").is_not_null() & (c("fam") == 9) & (c("nb") == 2))
print("fc 2-bait: corr(delay, payback amount)", np.corrcoef(x["delay_h"].to_numpy(), x["pamt"].to_numpy())[0, 1], "| bait amounts", r.filter(c("bait")).group_by("fam", "amount").len().sort("fam", "amount").rows())
print("events per month", e.group_by("fam", "month").len().sort("fam", "month").rows())
# app constant per account?
print("apps per sender (legit rows, senders with >=5 rows):", tx.filter(c("y") == 0).group_by("sender_id").agg(pl.len(), c("app").n_unique().alias("u")).filter(c("len") >= 5)["u"].mean())
