"""Experiment A: numeric representation and binning (benchmark N, same LightGBM recipe as arm C, 2 seeds).
audit : raw CSV text -> Float64 (prep.py) -> Float32 matrix. For each raw numeric column: distinct values as integer keys of
        the printed precision (exact), distinct values after the Float32 cast, how many raw values collapse, and how many
        repeated values have support (>= 50 and >= 500 rows in Jul+Aug).
arms  : A_BSEL  max_bin 1023 on the row-level numeric columns only (max_bin_by_feature), 255 elsewhere
        A_B1023 max_bin 1023 on every feature
        A_EXACT current bins + exact-value features from the raw columns amount, balance_before, sim_age_hours,
                device_age_days, login_to_txn_sec: log count of the exact value in the world (label-free, fit + held rows)
                and, for amount / sim age / device age, a categorical id of every value with >= 500 rows (0 otherwise).
                Keys are integers built from the Float64 values (cents, tenths, seconds), never from Float32.
A new lgb.Dataset is built for every fit, so the bins follow the arm's settings. No target encoding is used.
usage (cwd=work, PYTHONPATH=.): python scratch_gap/numrep.py audit | python scratch_gap/numrep.py ARM [ARM ...]"""
import sys, time
import numpy as np
import polars as pl
import lightgbm as lgb
import train as T
from common import load_all

c = pl.col; SEEDS, ROUNDS, LF = 2, 150, 0.2
META = ("txn_id", "month", "is_test", "y_fit")
RAW = {"amount": 100, "balance_before": 100, "sim_age_hours": 10, "device_age_days": 10, "login_to_txn_sec": 1}  # column -> units per integer key
ROWNUM = ["amount", "log_amount", "cents", "sim_age_hours", "device_age_days", "login_to_txn_sec", "balance_before", "amt_share_bal", "minute_of_day"]


def keys(tx):
    return tx.select("txn_id", *[(c(k) * u).round(0).cast(pl.Int64).alias(f"k_{k}") for k, u in RAW.items()])


def audit():
    tx = load_all(); fit = tx.filter(~c("is_test") & c("month").is_in([7, 8]))
    print(f"{'column':22s} {'distinct exact':>14s} {'distinct f32':>12s} {'collapsed':>9s} {'max abs err':>11s} {'values n>=50':>12s} {'rows in them':>12s} {'values n>=500':>13s} {'rows in them':>12s}")
    for k, u in RAW.items():
        x = tx[k].to_numpy(); kk = np.round(x * u).astype(np.int64); f32 = x.astype(np.float32)
        assert np.abs(kk / u - x).max() < 1e-6, k  # the integer key reproduces the parsed value: no finer raw precision exists
        vc = fit.select((c(k) * u).round(0).cast(pl.Int64).alias("k")).group_by("k").len()
        a, b = vc.filter(c("len") >= 50), vc.filter(c("len") >= 500)
        print(f"{k:22s} {len(np.unique(kk)):14d} {len(np.unique(f32)):12d} {len(np.unique(kk)) - len(np.unique(f32)):9d} {np.abs(f32.astype(np.float64) - x).max():11.2e}"
              f" {a.height:12d} {int(a['len'].sum()):12d} {b.height:13d} {int(b['len'].sum()):12d}  (fit rows {fit.height})")
    a = tx["amount"].to_numpy()
    for name, v in (("log_amount", np.log1p(a)), ("amt_share_bal", a / tx["balance_before"].to_numpy())):
        print(f"{name:22s} distinct in Float64 {len(np.unique(v))}, after Float32 {len(np.unique(v.astype(np.float32)))}")
    t = fit.with_columns((c("amount") * 100).round(0).cast(pl.Int64).alias("k")).group_by("k").agg(pl.len().alias("n"), (c("y") > 0).mean().alias("fraud"))
    print("amount values with >= 500 fit rows and fraud-label share >= 0.5:", t.filter((c("n") >= 500) & (c("fraud") >= .5)).sort("n", descending=True).select((c("k") / 100).alias("amount"), "n", c("fraud").round(3)).rows())
    print("amount values with 50..499 fit rows and fraud-label share >= 0.5:", t.filter((c("n") >= 50) & (c("n") < 500) & (c("fraud") >= .5)).height, "values,", int(t.filter((c("n") >= 50) & (c("n") < 500) & (c("fraud") >= .5))["n"].sum()), "rows")


def run(arm, K=None):
    K = keys(load_all()) if K is None else K
    for fi, (fm, vm) in enumerate(T.FOLDS):
        D = pl.read_parquet(f"scratch_gap/N_f{fi}.parquet"); feats = [k for k in D.columns if k not in META]; cat = "auto"
        params = dict(objective="multiclass", num_class=T.K, learning_rate=0.05, **T.LGB, num_threads=8, verbose=-1)
        if arm == "A_B1023": params["max_bin"] = 1023
        if arm == "A_BSEL": params["max_bin_by_feature"] = [1023 if k in ROWNUM else 255 for k in feats]
        if arm == "A_EXACT":
            W = D.select("txn_id").join(K, on="txn_id", how="left", maintain_order="left"); new = []
            for k in RAW:
                W = W.with_columns(pl.len().over(f"k_{k}").alias("n"))
                new.append(W["n"].cast(pl.Float32).log1p().alias(f"xc_{k}"))
                if k in ("amount", "sim_age_hours", "device_age_days"):
                    ids = W.filter(c("n") >= 500).select(f"k_{k}").unique().sort(f"k_{k}").with_row_index("id", offset=1)
                    new.append(W.select(f"k_{k}").join(ids, on=f"k_{k}", how="left", maintain_order="left")["id"].fill_null(0).cast(pl.Float32).alias(f"xid_{k}"))
            D = D.with_columns(new); feats = feats + [s.name for s in new]; cat = [s.name for s in new if s.name.startswith("xid_")]
            if fi == 0: print(arm, "added", [s.name for s in new], "| categories:", {s.name: int(s.max()) for s in new if s.name.startswith("xid_")}, flush=True)
        fit = D.filter(~c("is_test")); V = D.filter(c("is_test") & (c("month") == vm)); y = fit["y_fit"].to_numpy().astype(np.int32)
        P = np.zeros((V.height, T.K), np.float32); t = time.time()
        for s in range(SEEDS):
            keep = ((y != 0) | (np.random.default_rng(s).random(len(y)) < LF)) & (y >= 0)
            ds = lgb.Dataset(fit.select(feats).filter(pl.Series(keep)).to_numpy(), y[keep], weight=np.where(y[keep] == 0, 1 / LF, 1.0).astype(np.float32),
                             feature_name=feats, categorical_feature=cat)
            m = lgb.train({**params, "seed": s}, ds, ROUNDS)
            P += m.predict(V.select(feats).to_numpy()).astype(np.float32) / SEEDS
        np.save(f"scratch_gap/PN_{arm}_{'aug' if vm == 8 else 'sep'}.npy", P)
        print(f"{arm} fold{fi} done ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "audit": audit()
    else:
        K = keys(load_all())
        for a in sys.argv[1:]: run(a, K)
