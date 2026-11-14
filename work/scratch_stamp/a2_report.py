"""A2 report: September recall and precision on UNSTAMPED and stamped rows, variants vs C, from a bench_eval decode file.
stamped = (non-QR and mismatch == 0) or (merchant QR and mismatch null); unstamped = every other row.
recall  = of fraud-labelled rows in the group: share decoded non-legit (detect) and share decoded as the labelled class (exact).
precision = of rows in the group decoded non-legit: share fraud-labelled (detect) and share with the labelled class (exact).
usage (cwd=work, PYTHONPATH=.): python scratch_stamp/a2_report.py N dec_N_stamp.npz C,m50,m50b35"""
import sys
import numpy as np
import polars as pl

c = pl.col; B, f, VARS = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
z = np.load("scratch_gap/" + f); y = z["y"]; D = pl.read_parquet(f"scratch_gap/{B}_sep_rows.parquet"); assert D.height == len(y)
st = D.select((((c("txn_type") != "QR") & (c("qr_vpa_mismatch") == 0)).fill_null(False)
               | ((c("txn_type") == "QR") & (c("receiver_type") == "MERCHANT") & c("qr_vpa_mismatch").is_null()))).to_series().to_numpy()
nq = D.select(c("txn_type") != "QR").to_series().to_numpy()
print(f"benchmark {B} September rows {len(y)}: stamped {int(st.sum())} (fraud-labelled {int((st & (y > 0)).sum())}), unstamped fraud-labelled {int((~st & (y > 0)).sum())} (non-QR {int((~st & nq & (y > 0)).sum())})")
for stage in ("pre", "q"):
    print(f"--- {'before decoder (bias only)' if stage == 'pre' else 'after decoder'} ---")
    for nm, g in (("unstamped", ~st), ("unstamped non-QR", ~st & nq), ("stamped", st)):
        for v in VARS:
            p = z[f"{v}_{stage}"]; fr = g & (y > 0); dec = g & (p > 0); r = z[f"{VARS[0]}_{stage}"]
            print(f"{nm:17s} {v:7s} fraud-labelled {int(fr.sum()):5d} | recall detect {np.mean(p[fr] > 0):.4f} ({int((p[fr] > 0).sum())}) exact {np.mean(p[fr] == y[fr]):.4f} ({int((p[fr] == y[fr]).sum())})"
                  f" | decoded non-legit {int(dec.sum()):5d} precision detect {np.mean(y[dec] > 0):.4f} exact {np.mean(p[dec] == y[dec]):.4f}"
                  f" | vs {VARS[0]}: missed fraud now detected {int((fr & (r == 0) & (p > 0)).sum())}, detected fraud now missed {int((fr & (r > 0) & (p == 0)).sum())},"
                  f" legit newly called fraud {int((g & (y == 0) & (r == 0) & (p > 0)).sum())}, legit no longer called fraud {int((g & (y == 0) & (r > 0) & (p == 0)).sum())}")
