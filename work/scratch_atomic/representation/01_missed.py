"""Agent A: missed-fraud groups, matched controls, raw-field scan. cwd=work, PYTHONPATH=.
python scratch_atomic/representation/01_missed.py > scratch_atomic/representation/01_out.txt
Missed fraud row = y_true > 0 and final decoder label == 0. September: dec files. August: aug_dec.py-style decode
(bias tuned on the other event-grouped half). July: no held-out predictions exist (fold 0 fits July) -> all fraud events.
Every predicate below is label-free (computed on load_all() incl. test rows)."""
import numpy as np
import polars as pl
from scipy.stats import rankdata
import train as T
from common import load_all, LABELS, LABEL2ID as L
from clean_labels import fraud_events
from feats.ts_event import TR_EDGES, zone_ids
from ts_judge_post import zones, override

c = pl.col; OUT = "scratch_atomic/representation/"; SEED = 2026
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260); pl.Config.set_fmt_str_lengths(40)
tx = load_all()
_, FE = fraud_events(tx.filter(~c("is_test")), g=48 * 3600)

# ---------------------------------------------------------------- label-free raw-derived columns (NOT in the 190-col matrix)
day = 86400
R = tx.with_columns(
    c("timestamp").dt.second().alias("sec"), c("timestamp").dt.microsecond().alias("usec"),
    (c("ip_risk_score") - c("ip_risk_score").median().over("sender_id", "month")).alias("ip_dev"),
    (c("ip_risk_score").rank() / pl.len()).over("month").alias("ip_mrank"),
    ((c("timestamp").dt.date() - c("rcv_opened_on")).dt.total_days()).alias("rcv_age_days"),
    ((c("timestamp").dt.date() - c("mer_onboarded_on")).dt.total_days()).alias("mer_age_days"),
    c("sender_id").n_unique().over("device_id").alias("dev_nsnd"),
    c("device_city").n_unique().over("device_id").alias("dev_ncity"),
    c("device_id").n_unique().over("sender_id").alias("snd_ndev"),
    pl.len().over("sender_id", "month").alias("snd_n_month"),
    ((c("ts").rank("ordinal").over("sender_id", "device_id") == 1) & (c("ts").rank("ordinal").over("sender_id") > 1)).alias("sd_first"),
    (c("device_city") != c("device_city").shift(1).over("sender_id")).fill_null(False).alias("city_chg"),
    (c("ts").rank("ordinal").over("receiver_id") == 1).alias("rcv_first"),
    (c("amount") > c("balance_before")).alias("overdraft"),
    (c("snd_bank_code") == c("rcv_bank_code")).alias("same_bank"),
    (c("device_city") == c("rcv_home_city")).alias("dcity_eq_rcvhome"),
    (c("snd_home_city") == c("rcv_home_city")).alias("home_eq_rcvhome"),
    (c("snd_os") == c("rcv_os")).alias("same_os"),
    ((c("balance_before") - (c("balance_before") - c("amount")).shift(1).over("sender_id")) / c("balance_before").shift(1).over("sender_id")).alias("bal_jump_rel"),
    (c("app_version") - c("app_version").max().over("sender_id", "month")).alias("appv_below_max"),
    (c("app") != c("app").shift(1).over("sender_id")).fill_null(False).alias("app_chg"),
    (c("ts") - c("ts").shift(1).over("sender_id")).alias("snd_gap_s"),
)
jul = R.filter(c("month") == 7)
q = {k: float(jul[k].quantile(p)) for k, p in (("ip_risk_score", .95), ("ip_dev", .95))}
print("frozen on July all rows (label-free): ip_risk_score q95 = %.4f, ip_dev q95 = %.4f" % (q["ip_risk_score"], q["ip_dev"]))
PRED = {  # exact label-free predicates
    "ip_ge_julq95": c("ip_risk_score") >= q["ip_risk_score"],
    "ipdev_ge_julq95": c("ip_dev") >= q["ip_dev"],
    "ip_mrank_ge95": c("ip_mrank") >= .95,
    "appv_lt4": c("app_version") < 4,
    "appv_below_snd_month_max": c("appv_below_max") < 0,
    "sd_first": c("sd_first"),
    "dev_shared_ge2snd": c("dev_nsnd") >= 2,
    "dev_ncity_ge2": c("dev_ncity") >= 2,
    "city_chg_vs_prev_row": c("city_chg"),
    "app_chg_vs_prev_row": c("app_chg"),
    "rcv_first_seen": c("rcv_first"),
    "overdraft": c("overdraft"),
    "usec_eq0": c("usec") == 0,
    "sec_eq0": c("sec") == 0,
    "same_bank": c("same_bank").fill_null(False),
    "dcity_eq_rcvhome": c("dcity_eq_rcvhome").fill_null(False),
    "home_eq_rcvhome": c("home_eq_rcvhome").fill_null(False),
    "snd_acct_age_lt30": c("sender_account_age_days") < 30,
    "rcv_age_lt30": (c("rcv_age_days") < 30).fill_null(False),
    "mer_age_lt90": (c("mer_age_days") < 90).fill_null(False),
    "bal_jump_up_gt5pct": (c("bal_jump_rel") > .05).fill_null(False),
    "bal_jump_down_gt5pct": (c("bal_jump_rel") < -.05).fill_null(False),
    "snd_gap_lt60s": (c("snd_gap_s") < 60).fill_null(False),
}
NUM = ["ip_risk_score", "ip_dev", "ip_mrank", "app_version", "appv_below_max", "sender_account_age_days", "rcv_age_days", "mer_age_days",
       "sec", "usec", "dev_nsnd", "dev_ncity", "snd_ndev", "bal_jump_rel", "snd_gap_s", "snd_n_month",
       "login_to_txn_sec", "sim_age_hours", "device_age_days", "pin_retries"]  # last 4 are in the matrix (reference)
R = R.with_columns([e.cast(pl.Int8).alias("P_" + k) for k, e in PRED.items()]).with_columns(
    c("amount").cut([100, 500, 1000, 2000, 5000, 10000]).cast(pl.String).alias("amt_band"),
    c("snd_n_month").cut([2, 5, 10, 20]).cast(pl.String).alias("act_band"),
    c("device_id").str.slice(2).cast(pl.Int64).alias("dn"))
PC = ["P_" + k for k in PRED]
print("\n== predicate rate over ALL rows per month (Oct = test, label-free drift check)")
print(R.group_by("month").agg(pl.len().alias("n"), *[c(p).mean().round(4) for p in PC]).sort("month"))


# ---------------------------------------------------------------- held-out decodes
def force(P, D):
    Q = P.copy()
    for col, k in (("takeover__arch_sim", L["sim_swap"]), ("takeover__arch_ato", L["account_takeover"])):
        h = D.select((c(col).fill_nan(None) == 1).fill_null(False)).to_series().to_numpy(); Q[h] = 0; Q[h, k] = 1
    return Q


def marg(F, b):
    Lg = np.log(F + 1e-12) + b
    return Lg[:, 1:].max(1) - Lg[:, 0]


def held(B, ref, decf):
    Da, Ds = (pl.read_parquet(f"scratch_gap/{B}_{k}_rows.parquet") for k in ("aug", "sep"))
    Pa, Ps = np.load(f"scratch_gap/P{B}_{ref}_aug.npy"), np.load(f"scratch_gap/P{B}_{ref}_sep.npy")
    ya = Da["y_true"].to_numpy().astype(np.int64); Fa, Fs = force(Pa, Da), force(Ps, Ds)
    z = np.load(f"scratch_gap/{decf}.npz"); qs, ps = z[f"{ref}_q"], z[f"{ref}_p1"]
    bs = T.tune_bias(Fa, ya); assert (T.decode(Fs, bs) == ps).mean() > .9999, "Sep decode does not reproduce the dec file"
    sep = Ds.select("txn_id", "y_true", "takeover__arch_sim", "takeover__arch_ato", "ts_bait__rule").with_columns(
        pl.Series("q", qs.astype(np.int64)), pl.Series("p1", ps.astype(np.int64)), pl.Series("marg", marg(Fs, bs).astype(np.float64)), pl.Series("pf", (1 - Ps[:, 0]).astype(np.float64)))
    # August: aug_dec.py halves
    n = len(ya)
    ev = Da.select("txn_id").join(FE.select("txn_id", "ev"), on="txn_id", how="left", maintain_order="left")["ev"].to_numpy().astype(float)
    rng = np.random.default_rng(2026); evh = rng.integers(0, 2, int(np.nanmax(ev)) + 1)
    half = np.where(np.isnan(ev), rng.integers(0, 2, n), evh[np.nan_to_num(ev).astype(np.int64)])
    rule = Da["ts_bait__rule"].fill_null(0).to_numpy(); qa = np.zeros(n, np.int64); pa = np.zeros(n, np.int64); ma = np.zeros(n)
    for a_, b_ in ((0, 1), (1, 0)):
        A_, B_ = half == a_, half == b_
        b = T.tune_bias(Fa[A_], ya[A_]); p1 = T.decode(Fa, b); zz, _ = zones(Da, p1, TR_EDGES[0], TR_EDGES[-1]); qq = override(p1, zz, rule)
        qa[B_] = qq[B_]; pa[B_] = p1[B_]; ma[B_] = marg(Fa, b)[B_]
    aug = Da.select("txn_id", "y_true", "takeover__arch_sim", "takeover__arch_ato", "ts_bait__rule").with_columns(
        pl.Series("q", qa), pl.Series("p1", pa), pl.Series("marg", ma), pl.Series("pf", (1 - Pa[:, 0]).astype(np.float64)))
    print(f"\n#### benchmark {B} ({ref}): macro-F1 after decoder  Aug {T.macro_f1(ya, qa).mean():.4f}  Sep {T.macro_f1(Ds['y_true'].to_numpy().astype(np.int64), qs.astype(np.int64)).mean():.4f}")
    return pl.concat([aug.with_columns(pl.lit(8).alias("m")), sep.with_columns(pl.lit(9).alias("m"))])


def auc(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5: return float("nan")
    r = rankdata(np.concatenate([a, b]))
    return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


RAWC = [x for x in tx.columns if x not in ("label", "y", "is_test")]
KEEP = list(dict.fromkeys(RAWC + NUM + PC + ["amt_band", "act_band", "dn", "sd_first", "snd_gap_s"]))
legit_pool = R.filter(~c("is_test") & (c("y") == 0)).select(KEEP)
summ = []
for B, ref, decf in (("N", "C", "dec_N_wt"), ("D", "h6", "dec_D_sq6")):
    H = held(B, ref, decf).join(R.select(KEEP), on="txn_id", how="left").join(FE.select("txn_id", "ev"), on="txn_id", how="left")
    zt = zone_ids(H["dn"].to_numpy(), np.zeros(H.height, bool), [2 ** 62] * 12)
    H = H.with_columns(pl.Series("zone", zt.astype(np.int64)), (c("y_true") > 0).alias("fraud"), ((c("y_true") > 0) & (c("q") == 0)).alias("miss"),
                       pl.when(c("marg") > -1).then(pl.lit("a_borderline(>-1)")).when(c("marg") > -3).then(pl.lit("b_mid(-3..-1)")).otherwise(pl.lit("c_confident_legit(<-3)")).alias("mbin"),
                       ((c("takeover__arch_sim").fill_nan(None) == 1) | (c("takeover__arch_ato").fill_nan(None) == 1)).fill_null(False).alias("sig_arch"),
                       (c("ts_bait__rule").fill_null(0) > 0).alias("sig_bait"), (c("qr_vpa_mismatch") == 1).fill_null(False).alias("sig_qrmm"))
    H = H.with_columns((c("sig_arch") | c("sig_bait") | c("sig_qrmm") | (c("zone") > 0)).alias("sig_any"),
                       (c("device_age_days") < 3).alias("dev_new"), (c("dev_nsnd") >= 2).alias("dev_shared"))
    H.select("txn_id", "m", "y_true", "q", "p1", "marg", "pf", "ev", "sender_id", "receiver_id", "device_id", "ts").write_parquet(f"{OUT}held_{B}.parquet")
    F = H.filter(c("fraud"))
    for m in (8, 9):
        Fm = F.filter(c("m") == m)
        E = Fm.group_by("ev").agg(c("y_true").mode().first().alias("cls"), pl.len().alias("n"), c("miss").sum().alias("nmiss"))
        print(f"\n== {B} month {m}: fraud rows {Fm.height}, missed rows {int(Fm['miss'].sum())}; events {E.height}, events with >=1 missed row {E.filter(c('nmiss') > 0).height}, fully missed {E.filter(c('nmiss') == c('n')).height}")
        print(E.group_by("cls").agg(pl.len().alias("events"), (c("nmiss") > 0).sum().alias("ev_any_miss"), (c("nmiss") == c("n")).sum().alias("ev_full_miss"),
                                    c("n").sum().alias("rows"), c("nmiss").sum().alias("rows_missed")).sort("cls")
              .with_columns(pl.Series("class", [LABELS[i] for i in sorted(E["cls"].unique().to_list())]), (c("rows_missed") / c("rows")).round(3).alias("row_miss_rate")))
        M = Fm.filter(c("miss"))
        print("missed rows by margin bin x class:"); print(M.group_by("y_true", "mbin").len().pivot(on="mbin", index="y_true", values="len", sort_columns=True).sort("y_true"))
        print("missed rows: signatures present (should be forced/caught, yet legit):", {k: int(M[k].sum()) for k in ("sig_arch", "sig_bait", "sig_qrmm", "sig_any")}, "zone>0:", int((M["zone"] > 0).sum()))
        print("fraud rows by signature: miss rate with sig_any %.3f (n=%d) | without %.3f (n=%d)" % (
            Fm.filter(c("sig_any"))["miss"].mean(), Fm.filter(c("sig_any")).height, Fm.filter(~c("sig_any"))["miss"].mean(), Fm.filter(~c("sig_any")).height))
        print("fraud rows by context (n, miss rate):")
        print(Fm.group_by("txn_type", "receiver_type").agg(pl.len().alias("n"), c("miss").mean().round(3).alias("miss_rate"), c("miss").sum().alias("missed")).sort("missed", descending=True))
        print(Fm.group_by("dev_new", "dev_shared").agg(pl.len().alias("n"), c("miss").mean().round(3).alias("miss_rate"), c("miss").sum().alias("missed")).sort("missed", descending=True))
        # ---- one row per event: first missed row (missed events) / first row of events with no missed row (detected)
        rep_m = M.sort("ts").group_by("ev", maintain_order=True).first()
        det = Fm.join(E.filter(c("nmiss") == 0).select("ev"), on="ev").sort("ts").group_by("ev", maintain_order=True).first()
        # ---- matched legit controls: month, txn_type, receiver_type, amount band, sender activity band; 5 per event, fixed seed
        rng = np.random.default_rng(SEED); key = ["month", "txn_type", "receiver_type", "amt_band", "act_band"]
        pool = legit_pool.filter(c("month") == m).with_row_index("pi")
        cells = pool.group_by(key).agg(c("pi")); want = rep_m.group_by(key).agg(c("txn_id").alias("case_ids")).join(cells, on=key, how="left")
        pick, case = [], []; nomatch = 0
        for ids, pis in zip(want["case_ids"].to_list(), want["pi"].to_list()):
            if pis is None: nomatch += len(ids); continue
            for t in ids:
                s = rng.choice(pis, size=min(5, len(pis)), replace=False); pick += s.tolist(); case += [t] * len(s)
        ctl = pl.DataFrame({"pi": pick, "case_txn_id": case}).with_columns(c("pi").cast(pl.UInt32)).join(pool, on="pi").drop("pi")
        print(f"missed events {rep_m.height} (unmatched {nomatch}), matched legit controls {ctl.height}, detected events {det.height}, legit rows of month {pool.height}")
        rows = []
        for p in PC:
            a, b_, d_, l_ = rep_m[p].mean(), ctl[p].mean(), det[p].mean(), pool[p].mean()
            # borderline vs confident split among missed events
            cf = rep_m.filter(c("marg") <= -3)
            rows.append(dict(bench=B, month=m, predicate=p[2:], missed_ev_n=int(rep_m[p].sum()), missed_ev_rate=round(a, 4), conf_legit_ev_n=int(cf[p].sum()),
                             conf_legit_ev_rate=round(cf[p].mean(), 4) if cf.height else None, matched_ctl_rate=round(b_, 4), all_legit_rate=round(l_, 4),
                             detected_ev_rate=round(d_, 4), lift_vs_ctl=round(a / max(b_, 1e-9), 2)))
        summ += rows
        print(pl.DataFrame(rows).drop("bench", "month").sort("lift_vs_ctl", descending=True))
        print("numeric raw fields: AUC(missed event rep rows vs matched controls) | AUC(detected vs controls) | medians missed/ctl/detected")
        print(pl.DataFrame([dict(col=k, auc_missed_vs_ctl=round(auc(rep_m[k].cast(pl.Float64).to_numpy(), ctl[k].cast(pl.Float64).to_numpy()), 3),
                                 auc_detected_vs_ctl=round(auc(det[k].cast(pl.Float64).to_numpy(), ctl[k].cast(pl.Float64).to_numpy()), 3),
                                 med_missed=rep_m[k].median(), med_ctl=ctl[k].median(), med_det=det[k].median(),
                                 null_missed=round(rep_m[k].null_count() / max(rep_m.height, 1), 3), null_ctl=round(ctl[k].null_count() / max(ctl.height, 1), 3)) for k in NUM]))
        print("per class AUC ip_risk_score / ip_dev / app_version (missed rep rows of the class vs their matched controls):")
        for k_ in sorted(rep_m["y_true"].unique().to_list()):
            rm = rep_m.filter(c("y_true") == k_); cc = ctl.filter(c("case_txn_id").is_in(rm["txn_id"].to_list()))
            print(f"  {LABELS[k_]:20s} events {rm.height:5d} conf-legit {rm.filter(c('marg') <= -3).height:5d} " + " ".join(
                f"{x}={auc(rm[x].cast(pl.Float64).to_numpy(), cc[x].cast(pl.Float64).to_numpy()):.3f}" for x in ("ip_risk_score", "ip_dev", "ip_mrank", "app_version", "sender_account_age_days", "bal_jump_rel", "snd_gap_s")))
        cols = ["txn_id", "ev", "y_true", "q", "p1", "marg", "pf", "mbin", "zone", "sig_any"] + KEEP[1:]
        smp = rep_m.sample(n=min(300, rep_m.height), seed=SEED, shuffle=True).select(cols).with_columns(pl.lit("missed").alias("role"))
        dsm = det.sample(n=min(150, det.height), seed=SEED, shuffle=True).select(cols).with_columns(pl.lit("detected").alias("role"))
        smp.vstack(dsm).write_parquet(f"{OUT}sample_{B}_m{m}_events.parquet")
        ctl.filter(c("case_txn_id").is_in(smp["txn_id"].to_list())).write_parquet(f"{OUT}sample_{B}_m{m}_controls.parquet")
        rep_m.select(cols).write_parquet(f"{OUT}missed_events_{B}_m{m}.parquet")
        ctl.select("txn_id", "case_txn_id").write_parquet(f"{OUT}controls_{B}_m{m}.parquet")
        det.select(cols).write_parquet(f"{OUT}detected_events_{B}_m{m}.parquet")
pl.DataFrame(summ).write_csv(f"{OUT}predicate_rates.csv")

# ---------------------------------------------------------------- July: no held-out preds; all fraud events (first row) vs all legit
J = R.filter((c("month") == 7) & ~c("is_test")).join(FE.select("txn_id", "ev"), on="txn_id", how="left")
jf = J.filter(c("y") > 0).sort("ts").group_by("ev", maintain_order=True).first(); jl = J.filter(c("y") == 0)
print(f"\n== July (no held-out predictions): all fraud events {jf.height} (first row) vs all legit rows {jl.height}")
print(pl.DataFrame([dict(predicate=p[2:], fraud_ev_n=int(jf[p].sum()), fraud_ev_rate=round(jf[p].mean(), 4), all_legit_rate=round(jl[p].mean(), 4)) for p in PC]))
