"""Device-block post-rule on h6 (notes/typesplit_victim.md).

Fresh phones (device n >= 420000) are numbered per fraud class in fixed class order, per period (train / test). The class
blocks are recovered label-free: a monotone segmentation (DP) of the period's fraud-region devices, scored by the model's
own decoded predictions, then sim_swap | account_takeover split by the device-gap changepoint (sim block dense ~1.4,
ATO sparse ~3.8; preds cannot split them since ~30% of each block is the other archetype).
Outputs: subs/ts_victim.csv      h6_ppq, rows decoded sim/ATO on a sim/ATO-block phone get the block's type
         subs/ts_victim_blk.csv  h6_ppq, every row on a block-core phone gets the block's class (incl. rows decoded legit)
Check: blocks fit on Aug OOF preds only, scored on Sep (noisy and cleaned labels) + Sep takeover-event accuracy.
usage (cwd=work): python ts_victim_post.py
"""
import numpy as np
import polars as pl
from common import ROOT, LABELS, load_all, run_feature
from postproc import apply
from train import tune_bias, decode, macro_f1, write_sub, SAMPLE
from clean_labels import clean

c = pl.col
SIM, ATO = 4, 5
ORDER = [1, 2, 3, 45, 6, 7, 8, 9, 10, 11]  # 45 = sim_swap+ATO zone, split afterwards by gap density


def segment(dn, pr):
    """dn: sorted unique region device numbers, pr: their decoded class (0 = legit). Returns {class: (lo, hi)} block cores."""
    n, K = len(dn), len(ORDER)
    S = np.zeros((n, K))
    for j, k in enumerate(ORDER):
        hit = np.isin(pr, [SIM, ATO]) if k == 45 else pr == k
        S[:, j] = np.where(hit, 1.0, np.where(pr > 0, -1.0, 0.0))
    dp = np.cumsum(S, 0); arg = np.zeros((n, K), int)
    for i in range(1, n):                                  # dp[i, j] = best score of devices 0..i with device i in block j
        best = np.maximum.accumulate(dp[i - 1]); bj = np.array([int(np.argmax(dp[i - 1][:j + 1])) for j in range(K)])
        dp[i] = best + S[i]; arg[i] = bj
    blk = np.zeros(n, int); j = int(dp[-1].argmax())
    for i in range(n - 1, -1, -1):
        blk[i] = j; j = arg[i, j]
    cores = {}
    for j, k in enumerate(ORDER):
        m = np.flatnonzero((blk == j) & (S[:, j] > 0))
        if len(m) < 2: continue
        lo, hi = dn[m[0]], dn[m[-1]]
        if k != 45: cores[k] = (lo, hi); continue
        d = dn[(dn >= lo) & (dn <= hi)]; g = np.diff(d).astype(float)
        ll = [(-i * np.log(g[:i].mean()) - (len(g) - i) * np.log(g[i:].mean()), i) for i in range(5, len(g) - 5)]
        i = max(ll)[1]
        cores[SIM] = (lo, d[i]); cores[ATO] = (d[i + 1], hi)
    return cores


def block_of(dn, cores):
    out = np.zeros(len(dn), int)
    for k, (lo, hi) in cores.items():
        out[(dn >= lo) & (dn <= hi)] = k
    return out


def fit_cores(D, pred):
    """D: rows with dn, inreg; pred: decoded class per row -> cores from device-level majority non-legit pred."""
    x = D.with_columns(pl.Series("p", pred)).filter(c("inreg"))
    g = x.group_by("dn").agg(c("p").filter(c("p") > 0).mode().min().fill_null(0).alias("p")).sort("dn")
    return segment(g["dn"].to_numpy(), g["p"].to_numpy())


def overrides(pred, blk):
    a = pred.copy(); m = np.isin(pred, [SIM, ATO]) & np.isin(blk, [SIM, ATO]); a[m] = blk[m]
    b = pred.copy(); b[blk > 0] = blk[blk > 0]
    return a, b


def main():
    tx = load_all()
    f = run_feature("ts_victim", tx)
    base = tx.select("txn_id", "month", "sender_id", "ts", c("device_id").str.slice(2).cast(pl.Int64).alias("dn")).join(
        f.select("txn_id", c("ts_victim__blkpos").is_not_null().alias("inreg")), on="txn_id", how="left")
    ids = pl.read_parquet(ROOT / "oof" / "h6_ids.parquet")
    tid = pl.read_csv(SAMPLE).select(c("id").alias("txn_id"))
    y = ids["y"].to_numpy().astype(np.int64); fold = ids["fold"].to_numpy()
    yc = ids.join(tx.select("txn_id", clean(tx).alias("yc")), on="txn_id", how="left", maintain_order="left")["yc"].to_numpy().astype(np.int64)
    P1 = apply(np.load(ROOT / "oof" / "h6.npy"), ids, tx, True)
    T1 = apply(np.load(ROOT / "preds" / "h6_test.npy"), tid, tx, True)
    b = tune_bias(P1, y); po, pt = decode(P1, b), decode(T1, b)
    ref = pl.read_csv(ROOT / "subs" / "h6_ppq.csv")["label"].to_numpy()
    print("decoded test == subs/h6_ppq.csv:", bool((np.array(LABELS)[pt] == ref).all()))

    # ---- train check: cores from Aug (fold 0) OOF preds only, scored on Sep (fold 1)
    Do = ids.select("txn_id").join(base, on="txn_id", how="left", maintain_order="left")
    m0, m1 = fold == 0, fold == 1
    cores = fit_cores(Do.filter(pl.Series(m0)), po[m0])
    print("train cores (Aug preds):", {LABELS[k]: v for k, v in sorted(cores.items())})
    blk = block_of(Do["dn"].to_numpy(), cores) * Do["inreg"].fill_null(False).to_numpy()
    a, bb = overrides(po, blk)
    for name, p in (("h6_ppq", po), ("simato", a), ("blk", bb)):
        fs, fc = macro_f1(y[m1], p[m1]), macro_f1(yc[m1], p[m1])
        print(f"Sep {name:7s} noisy {fs.mean():.4f} (sim {fs[SIM]:.3f} ato {fs[ATO]:.3f})  cleaned {fc.mean():.4f} (sim {fc[SIM]:.3f} ato {fc[ATO]:.3f})"
              f"  rows changed {int((p[m1] != po[m1]).sum())}")
    print("Sep cleaned per class blk-h6:", np.round(macro_f1(yc[m1], bb[m1]) - macro_f1(yc[m1], po[m1]), 3).tolist())
    inb = blk[m1] > 0
    print(f"Sep rows on block phones {int(inb.sum())}: block==label {np.mean(blk[m1][inb] == y[m1][inb]):.3f}  ==cleaned {np.mean(blk[m1][inb] == yc[m1][inb]):.3f}")

    # ---- Sep takeover-event accuracy: event label = majority of sim/ATO rows (ev.parquet sessions)
    ev = pl.read_parquet(ROOT / "scratch_ts_victim" / "ev.parquet").filter(c("lab").is_not_null() & (c("month") == 9))
    eb = block_of(ev["edev"].str.slice(2).cast(pl.Int64).to_numpy(), cores)
    arch = np.where((ev["simf"] >= .5) & (ev["devy"] < .5), SIM, np.where(ev["devy"] >= .5, ATO,
                    np.where((ev["night"] >= .5) & (ev["ecity"] == ev["home"]), SIM, ATO)))
    rule = np.where(np.isin(eb, [SIM, ATO]), eb, arch); lab = ev["lab"].to_numpy()
    ext = np.isin(eb, [SIM, ATO])
    print(f"Sep takeover events {len(lab)}: arch acc {np.mean(arch == lab):.3f}  block rule acc {np.mean(rule == lab):.3f}  "
          f"(on block phones {ext.sum()}: {np.mean(rule[ext] == lab[ext]):.3f}; own-phone {(~ext).sum()}: {np.mean(rule[~ext] == lab[~ext]):.3f})")

    # ---- test
    Dt = tid.join(base, on="txn_id", how="left", maintain_order="left")
    ct = fit_cores(Dt, pt)
    print("test cores:", {LABELS[k]: v for k, v in sorted(ct.items())})
    bt = block_of(Dt["dn"].to_numpy(), ct) * Dt["inreg"].fill_null(False).to_numpy()
    at, btt = overrides(pt, bt)
    for tag, p in (("ts_victim", at), ("ts_victim_blk", btt)):
        print(f"{tag}: test rows changed vs h6_ppq {int((p != pt).sum())}  (from legit {int(((p != pt) & (pt == 0)).sum())})")
        oh = np.eye(len(LABELS), dtype=np.float32)[p]
        write_sub(oh, np.zeros(len(LABELS)), tag)


if __name__ == "__main__":
    main()
