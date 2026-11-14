"""A1 report: masked stamped non-QR held rows, original vs masked features, same boosters.
Primary decode = per-class bias tuned on the UNMASKED August predictions (bench_eval `pre`), applied to both sides.
August rows: bias tuned on the other event-grouped half (aug_dec halves), so no row is decoded with a bias fitted on itself.
Also: September after the full decoder (forcing, zones, bait) on both sides, and natural detection on unstamped fraud rows.
usage (cwd=work, PYTHONPATH=.): python scratch_stamp/a1_report.py"""
import numpy as np
import polars as pl
import train as T
from common import load_all, LABEL2ID as L, LABELS
from clean_labels import fraud_events
from feats.ts_event import TR_EDGES
from ts_judge_post import zones, override

c = pl.col; K = T.K
Da, Ds = (pl.read_parquet(f"scratch_gap/N_{k}_rows.parquet") for k in ("aug", "sep"))
ya, ys = Da["y_true"].to_numpy().astype(np.int64), Ds["y_true"].to_numpy().astype(np.int64)
Pa, Pam, Ps, Psm = (np.load(f"scratch_stamp/P_{a}_{m}.npy") for m in ("aug", "sep") for a in ("orig", "mask"))
ma, ms = np.load("scratch_stamp/mask_aug.npy"), np.load("scratch_stamp/mask_sep.npy")
for D, m in ((Da, ma), (Ds, ms)):  # the saved mask is the stamped non-QR set of the rows table
    assert (D.select(((c("txn_type") != "QR") & (c("qr_vpa_mismatch") == 0)).fill_null(False)).to_series().to_numpy() == m).all()

tx = load_all().filter(~c("is_test")); _, FE = fraud_events(tx, g=48 * 3600)
ev = Da.select("txn_id").join(FE.select("txn_id", "ev"), on="txn_id", how="left", maintain_order="left")["ev"].to_numpy().astype(float)
rng = np.random.default_rng(2026); evh = rng.integers(0, 2, int(np.nanmax(ev)) + 1)
half = np.where(np.isnan(ev), rng.integers(0, 2, len(ya)), evh[np.nan_to_num(ev).astype(np.int64)])
da, dam = np.zeros(len(ya), np.int64), np.zeros(len(ya), np.int64)
for a, b in ((0, 1), (1, 0)):
    bb = T.tune_bias(Pa[half == a], ya[half == a]); da[half == b] = T.decode(Pa[half == b], bb); dam[half == b] = T.decode(Pam[half == b], bb)
bias = T.tune_bias(Pa, ya); ds, dsm = T.decode(Ps, bias), T.decode(Psm, bias)
print("August-tuned bias (unmasked August):", dict(zip([l[:6] for l in LABELS], bias.round(1).tolist())))


def line(name, s, y, P, Pm, d, dm):
    n = int(s.sum())
    if n == 0: return f"{name:22s} n=0"
    pf, pfm = 1 - P[s, 0], 1 - Pm[s, 0]
    return (f"{name:22s} n={n:5d} | decoded non-legit {np.mean(d[s] > 0):.3f} -> {np.mean(dm[s] > 0):.3f} | mean P(fraud) {pf.mean():.3f} -> {pfm.mean():.3f}"
            f" | P(fraud)>=.5 {np.mean(pf >= .5):.3f} -> {np.mean(pfm >= .5):.3f} | decoded = label {np.mean(d[s] == y[s]):.3f} -> {np.mean(dm[s] == y[s]):.3f}"
            f" | argmax non-legit {np.mean(P[s].argmax(1) > 0):.3f} -> {np.mean(Pm[s].argmax(1) > 0):.3f}")


def table(tag, y, P, Pm, d, dm, m):
    print(f"\n=== {tag}: masked rows {int(m.sum())} (fraud-labelled {int((m & (y > 0)).sum())}, legit-labelled {int((m & (y == 0)).sum())}); before -> after masking ===")
    print(line("fraud-labelled", m & (y > 0), y, P, Pm, d, dm)); print(line("legit-labelled", m & (y == 0), y, P, Pm, d, dm))
    for k in range(1, K): print(line("  " + LABELS[k], m & (y == k), y, P, Pm, d, dm))
    for nm, s in (("fraud-labelled", m & (y > 0)), ("legit-labelled", m & (y == 0))):
        print(f"decoded type on masked {nm} rows, before: " + " ".join(f"{LABELS[k][:6]} {int((d[s] == k).sum())}" for k in range(K)))
        print(f"decoded type on masked {nm} rows, after : " + " ".join(f"{LABELS[k][:6]} {int((dm[s] == k).sum())}" for k in range(K)))
        ch = s & (d != dm); tr = {}
        for i in np.flatnonzero(ch): tr[(d[i], dm[i])] = tr.get((d[i], dm[i]), 0) + 1
        print(f"  decoded label changed on {int(ch.sum())} rows: fraud->legit {int((ch & (d > 0) & (dm == 0)).sum())}, legit->fraud {int((ch & (d == 0) & (dm > 0)).sum())},"
              f" type->type {int((ch & (d > 0) & (dm > 0)).sum())}; top: " + ", ".join(f"{LABELS[a][:6]}->{LABELS[b][:6]} {n}" for (a, b), n in sorted(tr.items(), key=lambda t: -t[1])[:8]))
    u = ~m
    print(f"rows NOT masked: {int(u.sum())}; P changed on {int((np.abs(P - Pm).max(1) > 0)[u].sum())}; decoded label changed on {int((d != dm)[u].sum())}")
    print(f"macro-F1 of the held month, bias only: original {T.macro_f1(y, d).mean():.4f}, masked {T.macro_f1(y, dm).mean():.4f} (counterfactual, not a score)")
    # context: naturally unstamped fraud-labelled rows (no masking), same decode
    nq = D_[tag].select(c("txn_type") != "QR").to_series().to_numpy()
    qs = D_[tag].select(((c("txn_type") == "QR") & (c("receiver_type") == "MERCHANT") & c("qr_vpa_mismatch").is_null())).to_series().to_numpy()
    print(line("natural: unstamped non-QR fraud-labelled", nq & ~m & (y > 0), y, P, P, d, d).replace(" -> ", " / "))
    print(line("natural: unstamped non-QR legit-labelled", nq & ~m & (y == 0), y, P, P, d, d).replace(" -> ", " / "))
    print(f"natural counts: non-QR rows {int(nq.sum())}, stamped {int(m.sum())}, QR-null stamp rows {int(qs.sum())} (fraud-labelled {int((qs & (y > 0)).sum())}), fraud-labelled rows {int((y > 0).sum())}, of them non-QR {int((nq & (y > 0)).sum())}")


D_ = {"August (fold 0, bias from the other half)": Da, "September (fold 1, August bias)": Ds}
table("August (fold 0, bias from the other half)", ya, Pa, Pam, da, dam, ma)
table("September (fold 1, August bias)", ys, Ps, Psm, ds, dsm, ms)


def force(P, D):
    Q = P.copy()
    for col, k in (("takeover__arch_sim", L["sim_swap"]), ("takeover__arch_ato", L["account_takeover"])):
        h = D.select((c(col).fill_nan(None) == 1).fill_null(False)).to_series().to_numpy(); Q[h] = 0; Q[h, k] = 1
    return Q


b2 = T.tune_bias(force(Pa, Da), ya); rule = Ds["ts_bait__rule"].fill_null(0).to_numpy(); q = {}
for nm, P in (("orig", Ps), ("mask", Psm)):
    p1 = T.decode(force(P, Ds), b2); z, e = zones(Ds, p1, TR_EDGES[0], TR_EDGES[-1]); q[nm] = override(p1, z, rule); print(f"zone edges ({nm}):", e)
m = ms; y = ys
print(f"\n=== September after the full decoder (forcing, zones, bait); bias re-tuned on unmasked forced August ===")
for nm, s in (("fraud-labelled", m & (y > 0)), ("legit-labelled", m & (y == 0))):
    print(f"masked {nm} n={int(s.sum())}: decoded non-legit {np.mean(q['orig'][s] > 0):.3f} -> {np.mean(q['mask'][s] > 0):.3f}; decoded = label {np.mean(q['orig'][s] == y[s]):.3f} -> {np.mean(q['mask'][s] == y[s]):.3f}")
print(f"rows NOT masked whose after-decoder label changed (zone edges moved): {int((q['orig'] != q['mask'])[~m].sum())}")
print(f"September macro-F1 after decoder: original {T.macro_f1(y, q['orig']).mean():.4f} (reference .5070), masked {T.macro_f1(y, q['mask']).mean():.4f} (counterfactual)")
print(f"September macro-F1 before decoder: original {T.macro_f1(y, ds).mean():.4f} (reference .4815)")
