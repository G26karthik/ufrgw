"""Agent A follow-up: are trace-free fraud labels placed independently per row, or on chosen entities / moments?
cwd=work, PYTHONPATH=.  python scratch_atomic/representation/09_placement.py > scratch_atomic/representation/09_out.txt
Supplied labels of July / August / September only; no model output.
Cohort A (trace-free, = C3): home device, no stamp, app_version >= 4, ip < July q95, non-integer amount.
Cohort B: all home-device rows. Fraud flag = supplied label > 0.
Null: the fraud flag is shuffled among cohort rows inside month x txn_type x receiver_type (200 shuffles, seed 2026), so every
entity's activity, every day's and hour's volume and every sender's row times are preserved exactly.
T1 entities with >= 2 / >= 3 fraud rows inside a month.  T2 P(entity has a fraud row in month m+1 | it had one in month m) and
union(Jul, Aug) -> Sep, plus the label-informed predicate 'entity held a cohort fraud label in Jul or Aug': lift and coverage on Sep.
T3 same-class share of fraud-row pairs on one entity against shuffling the class among the month's cohort fraud rows.
T4 chi-square of fraud counts per day and per hour of day against the stratum-preserving expectation; gaps between consecutive
fraud rows of one sender."""
import numpy as np
import polars as pl
from common import load_all

c = pl.col; NP = 200; H = 86400
tx = load_all().filter(~c("is_test") & c("month").is_in([7, 8, 9]))
q95 = float(tx.filter(c("month") == 7)["ip_risk_score"].quantile(.95))
num = lambda x: c(x).str.extract(r"(\d+)$").cast(pl.Int64, strict=False)
code = lambda e: e.cast(pl.Categorical).to_physical().cast(pl.Int64)
tx = tx.with_columns(
    (num("device_id") == num("sender_id")).fill_null(False).alias("home_dev"),
    (~((c("txn_type") != "QR") & (c("qr_vpa_mismatch") == 0)).fill_null(False) & (c("app_version") >= 4) & (c("ip_risk_score") < q95)
     & (c("amount") != c("amount").round(0))).alias("tracefree"))
ENT = ["sender", "receiver", "device", "merchant", "pair"]


def run(name, D):
    D = D.sort("sender_id", "ts").with_columns(
        code(c("sender_id")).alias("sender"), code(c("receiver_id")).alias("receiver"), code(c("device_id")).alias("device"),
        pl.when(c("receiver_type") == "MERCHANT").then(code(c("receiver_id"))).otherwise(-1).alias("merchant"),
        code(pl.concat_str("sender_id", pl.lit("|"), "receiver_id")).alias("pair"),
        code(pl.concat_str(c("month").cast(pl.String), "txn_type", "receiver_type")).alias("st"),
        c("timestamp").dt.day().alias("day"), c("timestamp").dt.hour().alias("hour"))
    n = D.height; mo = D["month"].to_numpy(); st = D["st"].to_numpy(); ts = D["ts"].to_numpy(); y = D["y"].to_numpy().astype(np.int64)
    E = {e: D[e].to_numpy() for e in ENT}; day = D["day"].to_numpy(); hour = D["hour"].to_numpy(); snd = E["sender"]
    f0 = y > 0
    strata = [np.where(st == s)[0] for s in np.unique(st)]; ks = [int(f0[i].sum()) for i in strata]
    # stratum-preserving expectation of fraud counts per (month, day) and (month, hour)
    rate = np.zeros(n)
    for i, k in zip(strata, ks): rate[i] = k / len(i)
    exp_d = {m: np.bincount(day[mo == m], rate[mo == m], 32) for m in (7, 8, 9)}; exp_h = {m: np.bincount(hour[mo == m], rate[mo == m], 24) for m in (7, 8, 9)}
    in_m = {m: mo == m for m in (7, 8, 9)}

    def stats(f):
        r = {}
        fm = {m: f & in_m[m] for m in (7, 8, 9)}
        for e in ENT:
            v = E[e]; S = {}
            for m in (7, 8, 9):
                x = v[fm[m]]; x = x[x >= 0]; cnt = np.bincount(x); S[m] = np.nonzero(cnt)[0]
                r[f"T1 {e} m{m} n>=2"] = int((cnt >= 2).sum()); r[f"T1 {e} m{m} n>=3"] = int((cnt >= 3).sum())
            for a, b in ((7, 8), (8, 9)):
                r[f"T2 {e} P(m{b}|m{a})"] = len(np.intersect1d(S[a], S[b], assume_unique=True)) / max(len(S[a]), 1)
            U = np.union1d(S[7], S[8]); r[f"T2 {e} P(m9|m7 or m8)"] = len(np.intersect1d(U, S[9], assume_unique=True)) / max(len(U), 1)
            # label-informed predicate on September rows: entity held a cohort fraud label in July or August
            hit = np.isin(v, U) & in_m[9] & (v >= 0); base = fm[9].sum() / in_m[9].sum()
            r[f"T2 {e} Sep predicate rows"] = int(hit.sum()); r[f"T2 {e} Sep predicate fraud rows"] = int((hit & f).sum())
            r[f"T2 {e} Sep predicate lift"] = ((hit & f).sum() / max(hit.sum(), 1)) / base
            r[f"T2 {e} Sep predicate coverage"] = (hit & f).sum() / max(fm[9].sum(), 1)
        for m in (7, 8, 9):
            od = np.bincount(day[fm[m]], minlength=32); ok = exp_d[m] > 0; r[f"T4 day chi2/df m{m}"] = float(((od[ok] - exp_d[m][ok]) ** 2 / exp_d[m][ok]).sum() / (ok.sum() - 1))
            oh = np.bincount(hour[fm[m]], minlength=24); ok = exp_h[m] > 0; r[f"T4 hour chi2/df m{m}"] = float(((oh[ok] - exp_h[m][ok]) ** 2 / exp_h[m][ok]).sum() / (ok.sum() - 1))
        i = np.nonzero(f)[0]; same = snd[i][1:] == snd[i][:-1]; g = (ts[i][1:] - ts[i][:-1])[same]   # rows are sorted by sender, ts
        r["T4 same-sender consecutive fraud pairs"] = int(same.sum())
        for lab, w in (("<1h", 3600), ("<1d", H), ("<7d", 7 * H)): r[f"T4 sender gap {lab}"] = int((g < w).sum())
        r["T4 sender gap median days"] = float(np.median(g) / H) if len(g) else float("nan")
        return r

    def t3(cls):   # same-class share of fraud-row pairs on one entity, per month
        r = {}
        for e in ENT:
            v = E[e]
            for m in (7, 8, 9):
                k = f0 & in_m[m] & (v >= 0); x = v[k]; cl = cls[k]
                cnt = np.bincount(x); tot = (cnt * (cnt - 1) // 2).sum()
                _, c2 = np.unique(x * 16 + cl, return_counts=True); r[f"T3 {e} m{m} same-class pair share"] = float((c2 * (c2 - 1) // 2).sum() / tot) if tot else float("nan")
                r[f"T3 {e} m{m} pairs"] = int(tot)
        return r

    obs = stats(f0); rng = np.random.default_rng(2026); null = []
    for _ in range(NP):
        f = np.zeros(n, bool)
        for i, k in zip(strata, ks): f[rng.choice(i, k, replace=False)] = True
        null.append(stats(f))
    o3 = t3(y); n3 = []
    for _ in range(NP):
        cls = y.copy()
        for m in (7, 8, 9):
            k = np.nonzero(f0 & in_m[m])[0]; cls[k] = y[rng.permutation(k)]
        n3.append(t3(cls))
    print(f"\n######## cohort {name}: rows {n}, fraud-labelled {int(f0.sum())} (Jul {int((f0 & in_m[7]).sum())}, Aug {int((f0 & in_m[8]).sum())}, Sep {int((f0 & in_m[9]).sum())}); strata {len(strata)}; {NP} shuffles")
    print("statistic | observed | null mean | null 2.5% .. 97.5% | z")
    for O, N_ in ((obs, null), (o3, n3)):
        for k in O:
            if k.endswith(" pairs") and k.startswith("T3"):
                print(f"{k} | {O[k]}"); continue
            a = np.array([d[k] for d in N_], float); sd = np.nanstd(a)
            fmt = (lambda v: f"{v:.0f}") if isinstance(O[k], int) else (lambda v: f"{v:.4f}")
            print(f"{k} | {fmt(O[k])} | {fmt(np.nanmean(a))} | {fmt(np.nanpercentile(a, 2.5))} .. {fmt(np.nanpercentile(a, 97.5))} | {(O[k] - np.nanmean(a)) / sd if sd > 0 else float('nan'):+.1f}")


H_ = tx.filter(c("home_dev"))
run("A trace-free", H_.filter(c("tracefree")))
run("B all home-device rows", H_)
