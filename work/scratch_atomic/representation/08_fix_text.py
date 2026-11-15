"""One-off text corrections to FINDINGS.md and field_coverage.md after review (no numbers recomputed). cwd=work."""
D = "scratch_atomic/representation/"
p = D + "FINDINGS.md"; s = open(p, encoding="utf-8").read()
R = [
    ("Updated 08:00 IST 15 Nov (clock from `date`).", "Updated 08:03 IST 15 Nov (clock from `date`)."),
    ("- New and measured: a **model-free and label-free count of fraud-path rows that the final decoder calls legit**.",
     "- New and measured: a **count of fraud-path rows among the fraud-labelled rows that the final decoder calls legit**. The trace is\n"
     "  label-free and model-free; the count uses the decode, and its baseline is the trace rate on legit-labelled decoded-legit rows of\n"
     "  the same month, so it cannot see fraud-path rows that carry a legit label."),
    ("the script re-derives the September forced decode and asserts that it equals the dec file.",
     "the script re-derives the September forced decode and asserts at least 99.99% agreement with the dec file."),
    ("5. Frozen on July (threshold .441, reference rates), checked on August and September, benchmark N, home-device rows",
     "5. Threshold .441 frozen on July; legit and reference rates are recomputed inside each month (they are stable: item 2). Checked on\n"
     "   August and September, benchmark N, home-device rows"),
    ("6. For agent C (executable): score a candidate decode with the `06_lost_cells.py` count. Fraud-path rows lost =\n"
     "   (J count among decoded-legit rows - .0066 x n) / .273.",
     "6. For agent C (executable): score a candidate decode by running `06_lost_cells.py` on it (it reads `held_<B>.parquet`: txn_id, q).\n"
     "   Fraud-path rows lost = (J count among decoded-legit rows - rJ x n) / (rF - rJ), with rJ = J rate on legit-labelled decoded-legit\n"
     "   rows and rF = J rate on rows fraud by label and decode, both taken from the SAME decode and month. Do not plug in a constant:\n"
     "   the excess is about 1% of the expected count, so a fixed .0066 gives wrong numbers. The count covers fraud-labelled rows only."),
    ("and `(txn_type == \"COLLECT\") & (receiver_type == \"P2P\") & (new_payee == 1) & (amount == round(amount))`.",
     "(A COLLECT-P2P new-payee integer cell was also read; it is dropped: see item 4.)"),
    ("COLLECT cell: Aug 24 in 2,063 against 13.8 (z 2.7);\n   Sep 17 in 2,149 against 14.0 (z 0.8).",
     "In the QR cell 7 of the 11 September J rows carry a fraud label; August has 2 J rows. COLLECT cell: only 4 of 24 (Aug) and 3 of 17 (Sep)\n"
     "   J rows carry a fraud label, so its excess is mostly legit-labelled rows and the month-wide baseline is not valid per cell:\n"
     "   **not evaluated adequately**. The phishing-topped fraud-labelled J rows sit in integer PAY-P2P cells (Aug 22 with new_payee = 0, 13 with new_payee = 1)."),
    ("5. Cells were read on August and September together, so this is not a frozen test. September supports the QR cell, not the COLLECT cell.",
     "5. Cells were read on August and September together, so this is not a frozen test. The QR cell rests on September."),
]
for a, b in R:
    assert a in s, a[:60]
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
p = D + "field_coverage.md"; s = open(p, encoding="utf-8").read()
a = "| snd_home_city | transformed |"
assert a in s
s = s.replace(a, "| account_id | join key only | common.py:30-35 | joined to sender_id and receiver_id; number equals the home device number (`DV0` + number) |\n" + a, 1)
a = "| mer_category | preserved (code) |"
assert a in s
s = s.replace(a, "| merchant_id | join key only | common.py:37-38 | joined to receiver_id |\n" + a, 1)
open(p, "w", encoding="utf-8").write(s); print("ok")
