"""Agent B: (1) univariate AUC of every rel__ column, population vs legit of the same month (no model fit);
(2) variation of the new column inside fixed values of the closest existing column (what the new one adds), with population rates per cell."""
import numpy as np, polars as pl
from scipy.stats import rankdata
from common import load_all
from clean_labels import fraud_events
D = "scratch_atomic/relationships/"
tx = load_all(); n = tx.height; c = pl.col
t, F = fraud_events(tx, 48 * 3600)
ev = np.full(n, -1); ev[F["i"].to_numpy()] = F["ev"].to_numpy(); evn = np.zeros(n); u, k = np.unique(ev[ev >= 0], return_counts=True); evn[ev >= 0] = k[np.searchsorted(u, ev[ev >= 0])]
rows = pl.read_parquet("scratch_gap/N_sep_rows.parquet", columns=["txn_id"]).with_columns(pl.Series("q", np.load("scratch_gap/dec_N_wt.npz")["C_q"]))
q = tx.select("txn_id").join(rows, on="txn_id", how="left", maintain_order="left")["q"].fill_null(-1).to_numpy()
y = tx["y"].to_numpy().astype(float); mo = tx["month"].to_numpy()
R = tx.select("txn_id").join(pl.read_parquet(D + "rel_feats.parquet"), on="txn_id", how="left", maintain_order="left")
EX = tx.select("txn_id").join(pl.read_parquet("scratch_gap/N_f1.parquet", columns=["txn_id", "flow__sin_n24h", "scam__rcv_in_15d", "takeover__n_other_dev", "takeover__dev_frac", "scam__rev_pair", "flow__rout_n6h"]), on="txn_id", how="left", maintain_order="left")
def uniq(m):
    i = np.where(m)[0]; _, j = np.unique(ev[i], return_index=True); return i[j]
POP = {"single_m7": np.where((y > 0) & (evn == 1) & (mo == 7))[0], "single_m8": np.where((y > 0) & (evn == 1) & (mo == 8))[0],
       "missed_m9": uniq((y > 0) & (q == 0) & (mo == 9)), "caught_m9": uniq((y > 0) & (q > 0) & (mo == 9))}
def auc(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 30: return np.nan
    r = rankdata(np.concatenate([a, b])); return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
out = []
for col in R.columns[1:]:
    v = R[col].to_numpy().astype(float); row = {"col": col}
    for pn, idx in POP.items():
        m = int(pn[-1]); L = np.where((y == 0) & (mo == m))[0]
        row["auc_" + pn] = auc(v[idx], v[L]); row["n_" + pn] = int((~np.isnan(v[idx])).sum())
    out.append(row)
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(40)
A = pl.DataFrame(out).with_columns(pl.selectors.float().round(4)); A.write_csv(D + "auc.csv"); print(A)
# (2) new information inside a fixed value of the closest existing feature
g = lambda cn: np.nan_to_num(EX[cn].fill_null(float("nan")).to_numpy().astype(float), nan=-1)
rv = lambda cn: np.nan_to_num(R["rel__" + cn].to_numpy().astype(float), nan=-1)
PAIRS = [("a_snd_prev_role", "flow__sin_n24h"), ("a_snd_in_run", "flow__sin_n24h"), ("a_rcv_prev_role", "scam__rcv_in_15d"), ("b_own_to_rcv", "takeover__n_other_dev"),
         ("b_snd_to_own", "takeover__n_other_dev"), ("b_dev_nco_before", "takeover__n_other_dev"), ("c_rcv_first_in", "scam__rcv_in_15d"), ("c_snd_first_out", "flow__sin_n24h")]
info = []
for new, old in PAIRS:
    a, b = np.minimum(rv(new), 3), np.minimum(g(old), 3)
    for m, pn in ((8, "single_m8"), (9, "missed_m9")):
        L = (y == 0) & (mo == m); P = np.zeros(n, bool); P[POP[pn]] = True
        d = pl.DataFrame({"old": b[L | P], "new": a[L | P], "pop": P[L | P]}).group_by("old", "new").agg(pl.len().alias("n"), c("pop").sum().alias("n_pop")).sort("old", "new")
        d = d.with_columns((c("n_pop") / c("n") * 100).round(2).alias("pop_pct"), pl.lit(new).alias("new_col"), pl.lit(old).alias("old_col"), pl.lit(pn).alias("popn"))
        info.append(d)
I = pl.concat(info).select("new_col", "old_col", "popn", "old", "new", "n", "n_pop", "pop_pct"); I.write_csv(D + "info_cells.csv")
print(I.filter(c("n") >= 200))
