"""fake_cashback / phishing event structure in train (labels), for the event generator. cwd=work."""
import polars as pl
from common import load_all
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
c = pl.col
tx = load_all().filter(~c("is_test"))
f = tx.filter(c("y").is_in([6, 9]))
print(f.group_by("y", "txn_type", "receiver_type", (c("amount") <= 10).alias("tiny"), (c("amount") % 1 == 0).alias("int")).len().sort("y", "len", descending=[False, True]))
# pair key (unordered accounts) for P2P rows
p = f.filter(c("receiver_type") == "P2P").with_columns(
    pl.min_horizontal("sender_id", "receiver_id").alias("a"), pl.max_horizontal("sender_id", "receiver_id").alias("b"),
    ((c("txn_type") == "PAY") & (c("amount") <= 10) & (c("amount") % 1 == 0)).alias("bait"))
g = p.group_by("a", "b").agg(pl.len().alias("n"), c("bait").sum().alias("nb"), c("y").mode().first().alias("maj"), (c("y") == 9).mean().alias("fc"),
                             (c("ts").max() - c("ts").min()).alias("span"), c("sender_id").n_unique().alias("nsend"))
print("pairs by majority label, n rows, n baits")
print(g.group_by("maj", c("n").clip(upper_bound=6), c("nb").clip(upper_bound=3)).agg(pl.len(), c("span").median() / 3600, c("nsend").mean()).sort("maj", "n", "nb"))
# fraudster reuse: bait senders in fc pairs
b = p.filter(c("bait") & (c("y") == 9)).group_by("sender_id").agg(c("receiver_id").n_unique().alias("victims"), pl.len())
print("fc bait senders:", b.height, b.group_by(c("victims").clip(upper_bound=8)).len().sort("victims"))
v = p.filter(c("bait") & (c("y") == 9)).group_by("receiver_id").agg(c("sender_id").n_unique().alias("fraudsters"))
print("fc victims by number of bait senders:", v.group_by("fraudsters").len().sort("fraudsters"))
# example events
ex = g.filter((c("maj") == 9) & (c("n") >= 3)).head(4)
for a_, b_ in ex.select("a", "b").rows():
    print(tx.filter(((c("sender_id") == a_) & (c("receiver_id") == b_)) | ((c("sender_id") == b_) & (c("receiver_id") == a_))).select(
        "timestamp", "sender_id", "receiver_id", "device_id", "txn_type", "amount", "new_payee", "pin_retries", "login_to_txn_sec", "sim_age_hours", "device_age_days", "balance_before", "device_city", "snd_home_city", "label"))
