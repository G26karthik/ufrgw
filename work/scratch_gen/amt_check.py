"""Is the Oct-like emulation faithful for integer amounts? Digit structure of integer amounts: real Sep, real Oct, emulated. cwd=work, PYTHONPATH=."""
import numpy as np, polars as pl
from common import load_all, ROOT
c = pl.col; a = c("amount")
tx = load_all()
sub = pl.read_csv(ROOT / "subs/dd_final.csv").rename({"id": "txn_id", "label": "pred"})
t = tx.join(sub, on="txn_id", how="left").filter((c("month") == 9) & ~c("is_test") & (c("y") == 0) | c("is_test") & (c("pred") == "legit"))
t = t.filter(a % 1 == 0)
def prof(df, col, name):
    x = df.select(c(col).alias("a"))
    print(f"{name:26s} n {x.height:7d} mult10 {x.select((c('a') % 10 == 0).mean()).item():.3f} mult50 {x.select((c('a') % 50 == 0).mean()).item():.3f} mult100 {x.select((c('a') % 100 == 0).mean()).item():.3f}"
          f" | within 50 below a round limit (2000,5000,10000,25000,50000,100000): {x.select(pl.any_horizontal([(c('a') < R) & (c('a') >= R - 50) for R in (2000, 5000, 10000, 25000, 50000, 100000)]).mean()).item():.5f}")
for rt in ("P2P", "MERCHANT"):
    s = t.filter(c("receiver_type") == rt)
    prof(s.filter(~c("is_test")), "amount", f"{rt} Sep legit int")
    prof(s.filter(c("is_test")), "amount", f"{rt} Oct decoded-legit int")
    prof(s.filter(~c("is_test")).with_columns((a * 1.17).round(0).alias("e")), "e", f"{rt} Sep emulated x1.17")
col = (c("txn_type") == "COLLECT") & (c("receiver_type") == "P2P")
print("COLLECT P2P rows per day: Sep", tx.filter(col & (c("month") == 9) & ~c("is_test")).height / 30, "Oct", tx.filter(col & c("is_test") & (c("month") == 10)).height / 31)
print("pred phishing: test dd_final", sub.filter(c("pred") == "phishing_collect").height, "| Sep arm A decode (x0.5 for density)", "see eval_arms n: ph_p/ph_r")
