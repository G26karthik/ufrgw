"""Score budget for the gap to the reported leader (0.6338 - 0.5505 = 0.0833 macro = 0.9996 summed class F1).
Known: dd_final predicted counts per class, public score 0.5505, two public class F1 from knockout uploads
(velocity_bot 0.373, merchant_collusion 0.337).
Assumed (scenario, not test truth): per-class precision of the current predictions (transported from the Aug+Sep OOF
decode) and the per-class true supports implied by it (validate_sub.state: hidden rows scaled so the base scores 0.5505).
Exact identity used for the break-even: adding rows to class c raises its F1 iff their precision exceeds F1_c / 2.
usage (cwd=work, PYTHONPATH=.): python scratch_gap/budget.py"""
import polars as pl
import validate_sub as V
from common import ROOT, LABELS

bc = dict(pl.read_csv(ROOT / "subs/dd_final.csv")["label"].value_counts().rows())
GAP = 0.6338 - 0.5505


def scenario(name, prec, prec_c):
    V.PREC, V.PREC_C = prec, prec_c
    m = V.fit_m(bc); s = V.state(bc, m)
    f1 = {k: 2 * tp / (p + n) for k, (tp, p, n) in s.items()}
    hidden = {k: s[k][2] - s[k][0] for k in LABELS[1:]}  # true rows of class k not predicted k
    print(f"\n[{name}] fitted base {V.macro(s):.4f}; implied fraud rows in test {sum(v[2] for k, v in s.items() if k != 'legit'):.0f}; "
          f"true rows of each class that we do not call that class: {sum(hidden.values()):.0f}")
    print("  class F1 now / break-even precision for added rows (F1/2): " + " ".join(f"{k[:5]} {f1[k]:.2f}/{f1[k]/2:.2f}" for k in LABELS[1:]))
    # all current fraud predictions typed right, nothing new found
    t = {k: [v[1] if k != "legit" else v[0], v[1], v[2]] for k, v in s.items()}
    print(f"  every current fraud prediction right, no new row found: {V.macro(t):.4f} ({V.macro(t) - V.macro(s):+.4f})")
    for q in (1.0, 0.75, 0.5, 0.35):
        line = []
        for target in (GAP, GAP / 2, GAP / 4):
            lo, hi = 0.0, 1.0
            for _ in range(50):  # f = share of each class's missed rows that the new signal flags correctly
                f = (lo + hi) / 2
                u = {k: v[:] for k, v in s.items()}
                for k in LABELS[1:]:
                    tp = f * hidden[k]; flagged = tp / q
                    u[k][0] += tp; u[k][1] += flagged; u["legit"][1] -= flagged; u["legit"][0] -= flagged - tp
                lo, hi = (lo, f) if V.macro(u) - V.macro(s) > target else (f, hi)
            tot = sum(f * hidden[k] for k in LABELS[1:])
            line.append(f"+{target:.4f}: {f:.0%} of missed rows = {tot:.0f} correct, {tot / q:.0f} flagged" if f < .999 else f"+{target:.4f}: not reachable")
        print(f"  precision {q:.2f}  " + " | ".join(line))


POST = dict(mule_account=.871, velocity_bot=.966, layering_chain=.875, sim_swap=.780, account_takeover=.969, phishing_collect=.869,
            qr_swap=.951, refund_abuse=.941, fake_cashback=.749, loan_app_extortion=.968, merchant_collusion=.938)  # compartments.py N
scenario("P: type precision of the final decode on the label-hidden Sep benchmark", 0.93, POST)
PRE = dict(sim_swap=.72, account_takeover=.72, fake_cashback=.69, mule_account=.86, layering_chain=.86, phishing_collect=.87)  # scratch_me/conf.py
scenario("A: type precision of the model decode before zones (Aug+Sep OOF)", 0.93, PRE)
scenario("B: every class 10 points less precise than A", 0.83, {k: v - 0.10 for k, v in PRE.items()})
scenario("C: all classes at precision 0.70", 0.70, {})
