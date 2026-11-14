"""Event generator v1 for the bait family (fake_cashback and phishing bait events): constrained event resampling.

Recipe read from train (scratch_gen/x1.py, x2.py): a bait event is 1-2 tiny P2P PAYs (1/2/5/10) from a fraudster F to a
victim V, then an integer payback V -> F (PAY or COLLECT, balance = amount / 0.909) minutes to 47 h later. Bait rows are
legit-recipe rows apart from amount and receiver; every row draws sim age, device age, pin retries and login time per row.
One generated event = one source event with:
  - both accounts remapped to two accounts drawn like legit senders (weight = rows sent; home device, home city, app
    of the new account),
  - all rows shifted by the same whole number of days inside the source month (time of day and gaps kept),
  - per-row nuisance fields (sim_age_hours, device_age_days, pin_retries, login_to_txn_sec, ip_risk_score) redrawn
    from a random legit P2P PAY row of the same month,
  - shape, amounts, balances, txn types, new_payee and the observed (noisy) row labels kept.
Not modelled (left as in the source or dropped): fresh / secondary devices (generated rows use the home device),
away-from-home rows keep the source device city.
Generated rows get features from the same extraction as real rows: each module is rebuilt on a train half plus the
generated events of that half (density.halves), `pairs` on the full table; only the generated rows are kept.

usage (cwd=work): python gen_events.py OUT.parquet [--mult 1] [--seed 0]
OUT.parquet: matrix rows (same columns as make_matrix.py) + gen_ev, src_a, src_b, fam, half. OUT.raw.parquet: raw rows.
"""
import argparse, importlib
import numpy as np
import polars as pl
from common import DATA, LABEL2ID, load_all, CACHE
from density import halves
from make_matrix import MODULES
from quick_eval import base_features

VERSION = "bait-v2"
c = pl.col
NUIS = ["sim_age_hours", "device_age_days", "pin_retries", "login_to_txn_sec", "ip_risk_score"]
BAIT = (c("txn_type") == "PAY") & (c("amount") <= 10) & (c("amount") % 1 == 0)
PAIR = [pl.min_horizontal("sender_id", "receiver_id").alias("a"), pl.max_horizontal("sender_id", "receiver_id").alias("b")]


def source_events(tx: pl.DataFrame) -> pl.DataFrame:
    """Train P2P rows of bait events: pairs with >= 1 bait row among rows labelled phishing (6) / fake_cashback (9);
    legit-labelled rows of the pair within 48 h of the event are kept (label noise). Events inside one month only."""
    p = tx.filter(~c("is_test") & (c("receiver_type") == "P2P")).with_columns(PAIR)
    ev = (p.filter(c("y").is_in([6, 9])).group_by("a", "b")
           .agg(BAIT.sum().alias("nb"), c("y").mode().min().alias("fam"), c("ts").min().alias("t0"), c("ts").max().alias("t1"))
           .filter(c("nb") >= 1).with_row_index("src_ev"))
    r = p.join(ev, on=["a", "b"]).filter(c("ts").is_between(c("t0") - 48 * 3600, c("t1") + 48 * 3600))
    one = r.group_by("src_ev").agg(c("month").n_unique().alias("nm")).filter(c("nm") == 1)
    return r.join(one.select("src_ev"), on="src_ev", how="semi")


def generate(tx: pl.DataFrame, mult: float, seed: int) -> pl.DataFrame:
    """Raw generated rows (train.parquet columns) + gen_ev, src_ev, fam."""
    rng = np.random.default_rng(seed)
    raw_cols = pl.read_parquet_schema(DATA / "train.parquet").names()
    S = source_events(tx)
    acc = pl.read_parquet(DATA / "accounts.parquet").with_columns(c("opened_on").str.to_date())
    app = tx.group_by("sender_id").agg(c("app").first())  # app is constant per account
    used = set(S["a"]) | set(S["b"])
    # accounts are drawn like legit-row senders (weight = rows sent): real event accounts are as active as legit senders,
    # a uniform draw over accounts gave too few counterparties and devices (scratch_gen/fid.py)
    pool = (acc.join(app, left_on="account_id", right_on="sender_id").filter(~c("account_id").is_in(list(used)))
               .join(tx.group_by("sender_id").len(), left_on="account_id", right_on="sender_id"))
    pw = (pool["len"] / pool["len"].sum()).to_numpy()
    month_lo = {m: tx.filter(c("month") == m)["ts"].min() // 86400 * 86400 for m in (7, 8, 9)}
    month_hi = {m: month_lo[m] + d * 86400 for m, d in ((7, 31), (8, 31), (9, 30))}
    donors = {m: tx.filter(~c("is_test") & (c("month") == m) & (c("y") == 0) & (c("receiver_type") == "P2P") & (c("txn_type") == "PAY"))
              .select(NUIS) for m in (7, 8, 9)}
    out, g = [], 0
    for (sid,), e in S.group_by("src_ev", maintain_order=True):
        for _ in range(int(mult) + (rng.random() < mult % 1)):
            m = int(e["month"][0]); t0, t1 = e["ts"].min(), e["ts"].max()
            ks = [k for k in range(-31, 32) if k and t0 + k * 86400 >= month_lo[m] and t1 + k * 86400 < month_hi[m]]
            if not ks:
                continue
            dt = int(rng.choice(ks)) * 86400
            new = pool[rng.choice(pool.height, 2, replace=False, p=pw).tolist()]
            mp = pl.DataFrame({"old": [e["a"][0], e["b"][0]], "new_id": new["account_id"], "home": new["home_city"],
                               "napp": new["app"], "opened": new["opened_on"]})
            d = donors[m][rng.integers(0, donors[m].height, e.height).tolist()]
            x = (e.join(mp.rename({"new_id": "ns", "home": "nhome", "napp": "nsapp", "opened": "nopen"}), left_on="sender_id", right_on="old", maintain_order="left")
                  .join(mp.select("old", c("new_id").alias("nr")), left_on="receiver_id", right_on="old", maintain_order="left"))
            ts = x["ts"] + dt
            x = x.with_columns(
                c("ns").alias("sender_id"), c("nr").alias("receiver_id"), ("DV0" + c("ns").str.slice(2)).alias("device_id"),
                pl.when(c("device_city") == c("snd_home_city")).then(c("nhome")).otherwise(c("device_city")).alias("device_city"),
                c("nsapp").alias("app"), pl.from_epoch(ts, "s").cast(pl.Datetime("us")).alias("timestamp"),
                ((ts // 86400 - c("nopen").cast(pl.Int64)).cast(pl.Float64)).alias("sender_account_age_days"),
                *[d[k].alias(k) for k in NUIS],
                pl.Series("txn_id", [f"GA{g:05d}{i:03d}" for i in range(e.height)]),
                pl.lit(g).alias("gen_ev"), pl.lit(sid).alias("src_ev"))
            out.append(x.select(raw_cols + ["gen_ev", "src_ev", "fam"])); g += 1
    G = pl.concat(out)
    assert G["txn_id"].n_unique() == G.height and not set(G["txn_id"]) & set(tx["txn_id"])
    return G


def decorate(G: pl.DataFrame, tx: pl.DataFrame) -> pl.DataFrame:
    """Same joins and derived columns as common.load_all, for generated raw rows; column order of tx."""
    acc = pl.read_parquet(DATA / "accounts.parquet").with_columns(c("opened_on").str.to_date())
    mer = pl.read_parquet(DATA / "merchants.parquet").with_columns(c("onboarded_on").str.to_date())
    x = G.with_columns(pl.lit(False).alias("is_test"))
    x = x.join(acc.rename({k: f"snd_{k}" for k in acc.columns if k != "account_id"}), left_on="sender_id", right_on="account_id", how="left")
    x = x.join(acc.rename({k: f"rcv_{k}" for k in acc.columns if k != "account_id"}), left_on="receiver_id", right_on="account_id", how="left")
    x = x.join(mer.rename({k: f"mer_{k}" for k in mer.columns if k != "merchant_id"}), left_on="receiver_id", right_on="merchant_id", how="left")
    x = x.with_columns(c("timestamp").dt.epoch("s").alias("ts"), c("timestamp").dt.month().alias("month"),
                       c("label").replace_strict(LABEL2ID, default=None, return_dtype=pl.Int8).alias("y"))
    return x.select(tx.columns).cast(dict(tx.schema))


def features(tx: pl.DataFrame, X: pl.DataFrame, half: np.ndarray) -> pl.DataFrame:
    """Matrix rows of the generated rows X (tx format): module builds on train half k + test + generated events of half k."""
    F = base_features(X)
    hs = {ml: halves(tx, mlink=ml) for ml in {ml for _, ml in MODULES}}
    for name, ml in MODULES:
        mod = importlib.import_module(f"feats.{name}"); parts = []
        for k in (0, 1):
            sub = pl.concat([tx.filter((hs[ml] == k).fill_null(False) | tx["is_test"]), X.filter(pl.Series(half == k))]).sort("ts")
            parts.append(mod.build(sub).join(X.filter(pl.Series(half == k)).select("txn_id"), on="txn_id", how="semi"))
        F = F.join(pl.concat(parts), on="txn_id", how="left", maintain_order="left"); print(name, F.shape, flush=True)
    pairs = importlib.import_module("feats.pairs").build(pl.concat([tx, X]).sort("ts"))
    F = F.join(pairs, on="txn_id", how="left", maintain_order="left")
    feats = [k for k in F.columns if k != "txn_id"]
    return X.select("txn_id", "month", "y", "is_test").join(F.with_columns([c(k).cast(pl.Float32) for k in feats]), on="txn_id", how="left", maintain_order="left")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--mult", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tx = load_all()
    G = generate(tx, a.mult, a.seed)
    print("generated events", G["gen_ev"].n_unique(), "rows", G.height, G.group_by("fam", "label").len().sort("fam", "len", descending=[False, True]).rows())
    X = decorate(G.drop("gen_ev", "src_ev", "fam"), tx)
    half = np.random.default_rng(a.seed + 1).integers(0, 2, G["gen_ev"].max() + 1)[G["gen_ev"].to_numpy()]  # whole events per half
    M = features(tx, X, half).with_columns(G["gen_ev"], G["src_ev"], G["fam"], pl.Series("half", half), pl.lit(VERSION).alias("gen_version"))
    ref = pl.read_parquet_schema(CACHE / "matrix_m5dm2_pairsfd.parquet").names()
    assert M.columns[:len(ref)] == ref, [k for k in ref if k not in M.columns][:5]
    G.write_parquet(a.out.replace(".parquet", ".raw.parquet")); M.write_parquet(a.out)
    print("written", a.out, M.shape)
