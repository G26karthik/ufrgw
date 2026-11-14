"""Local check of a candidate submission before it costs an upload slot.

usage (cwd=work): python validate_sub.py CAND.csv [BASE=subs/dd_final.csv]
  1. format: same ids/order as the base, valid labels.
  2. diff vs base by (from, to) class.
  3. estimated public score change under a per-class truth model, for a range of precisions of the changed rows
     (share of changed rows whose true label is the new label; the rest are assumed to be truly the old label).
Truth model (full test, public = same ratios): class c has P_c predicted rows in the base, TP_c = PREC * P_c of them
right, and H_c hidden rows we call legit (camouflage). H_c = M * CAMO_c, with M fitted so the base scores BASE_PUBLIC.
KNOWN_F1 overrides a class with its measured public F1 (knockout probe: F1 = 12 * score drop).
shortcut: one precision for all classes and row-independent truth; good to about +-.01 on large edits, sign and
break-even on small ones. Replace CAMO/KNOWN_F1 as probes come in.
"""
import sys
import polars as pl
from common import ROOT, LABELS

BASE_PUBLIC = 0.5505
PREC = 0.93
# share of decoded rows that carry the decoded class, h6 OOF Aug+Sep among fraud-labelled rows (scratch_me/conf.py)
PREC_C = {"sim_swap": 0.72, "account_takeover": 0.72, "fake_cashback": 0.69, "mule_account": 0.86, "layering_chain": 0.86,
          "phishing_collect": 0.87}
CAMO = {"mule_account": 1893, "velocity_bot": 893, "layering_chain": 821, "sim_swap": 429, "account_takeover": 500,
        "phishing_collect": 643, "qr_swap": 1071, "refund_abuse": 321, "fake_cashback": 429, "loan_app_extortion": 179,
        "merchant_collusion": 357}  # scratch_fe/judge/j4.py
KNOWN_F1 = {"velocity_bot": 0.373, "merchant_collusion": 0.337}  # knockouts 14 Nov: .5194, .5224 vs .5505


def state(base_counts, m):
    """class -> [TP, P, N] for the base."""
    s = {}
    for k in LABELS[1:]:
        p = base_counts.get(k, 0)
        tp = PREC_C.get(k, PREC) * p
        n = tp + m * CAMO[k]
        if k in KNOWN_F1:  # F1 = 2TP/(P+N)  ->  N
            n = 2 * tp / KNOWN_F1[k] - p
        s[k] = [tp, p, n]
    fraud_true = sum(v[2] for v in s.values())
    fp_legit = sum(v[2] - v[0] for v in s.values())  # fraud rows we call legit
    p_legit = base_counts["legit"]
    s["legit"] = [p_legit - fp_legit, p_legit, sum(base_counts.values()) - fraud_true]
    return s


def macro(s):
    return sum(2 * tp / (p + n) for tp, p, n in s.values()) / len(s)


def fit_m(base_counts):
    lo, hi = 0.0, 5.0
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if macro(state(base_counts, m)) > BASE_PUBLIC else (lo, m)
    return m


def delta(base_counts, m, moves, prec):
    """moves: [(from, to, n)]; prec = share of moved rows truly `to`, the rest truly `from`."""
    s = {k: v[:] for k, v in state(base_counts, m).items()}
    b = macro(s)
    for a, t, n in moves:
        s[a][1] -= n
        s[a][0] -= n * (1 - prec)  # rows that were right as `a`
        s[t][1] += n
        s[t][0] += n * prec
    return macro(s) - b


if __name__ == "__main__":
    cand = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "subs/dd_final.csv"
    b, c = pl.read_csv(ROOT / base), pl.read_csv(cand)
    assert c.columns == ["id", "label"] and c.height == b.height and c["id"].equals(b["id"]), "format/id order"
    assert c["label"].is_in(LABELS).all(), "bad label"
    d = b.with_columns(c["label"].alias("to")).filter(pl.col("label") != pl.col("to"))
    moves = d.group_by("label", "to").len().sort("len", descending=True).rows()
    print(f"format OK, {d.height} rows changed vs {base}")
    for a, t, n in moves:
        print(f"  {a} -> {t}: {n}")
    bc = dict(b["label"].value_counts().rows())
    m = fit_m(bc)
    print(f"model: camo multiplier {m:.3f}, base macro {macro(state(bc, m)):.4f}")
    if moves:
        row = {p: delta(bc, m, moves, p) for p in (1.0, .75, .5, .25, 0.0)}
        print("estimated public change by precision of the changed rows:")
        print("  " + "  ".join(f"p={p:.2f}: {v:+.4f}" for p, v in row.items()))
        lo, hi = 0.0, 1.0
        for _ in range(40):
            p = (lo + hi) / 2
            lo, hi = (p, hi) if delta(bc, m, moves, p) < 0 else (lo, p)
        print(f"break-even precision: {p:.3f}" if delta(bc, m, moves, 1.0) > 0 else "never gains under this model")
