"""One-off text corrections to section 8 of FINDINGS.md after review and after 12_sender_pairs.py. cwd=work. argv[1] = HH:MM from `date`."""
import sys
p = "scratch_atomic/representation/FINDINGS.md"; s = open(p, encoding="utf-8").read(); t = sys.argv[1]
R = [
    ("REFUND-to-merchant rows (5 / 11 / 10 rows in Jul / Aug / Sep, class refund_abuse)",
     "REFUND-to-merchant rows (5 / 11 / 10 rows in Jul / Aug / Sep, top class refund_abuse; two classes in Jul and Aug)"),
    ("- Reading: the sender excess is the same refund residue (one sender, several REFUND rows). Not a second mechanism. I did not\n"
     "  separate the sender excess by txn_type, so this reading is not evaluated adequately.",
     "- Check (`12_sender_pairs.py`, `12_out.txt`): same-class sender pairs observed / expected under class shuffle: Jul 29 / 26.0,\n"
     "  Aug 39 / 24.2, Sep 30 / 21.1. REFUND-to-merchant refund_abuse pairs: 4, 11, 8 (about half to the same merchant). Without them:\n"
     "  25, 28, 22. So refund episodes explain 11 of the 15 excess pairs in August and 8 of 9 in September; the rest (about +4 and +1) is\n"
     "  inside sampling noise (sd about 4.6). No QR-to-P2P pair is involved. Not a second mechanism."),
    ("| 3 class agreement | no detected difference beyond the refund residue | supported |",
     "| 3 class agreement | no detected difference once the refund episodes are removed (sender z +3.2 / +2.1 before removal; 1 of about 100 statistics read) | supported |"),
    ("supported for P2P receivers; merchants at the null; senders weak (Sep predicate lift 1.15 against .97..1.04)",
     "supported for P2P receivers; senders weak (Sep predicate lift 1.15 against .97..1.04); merchants and pairs recur LESS than the shuffle (see 8.5)"),
    ("- Exact predicate for October (**label-informed",
     "- Merchants and pairs in cohort B go the other way: merchants flagged in July or August hold 6,207 September fraud rows against\n"
     "  6,388..6,534 under the shuffle (lift .763 against .793, z -7); pair P(Sep | Jul or Aug) .0011 against .0014..0024. Plausible reason:\n"
     "  fraud-path merchant rows sit on small merchants and on pairs that are not used again. Not tested further.\n"
     "- Exact predicate for October (**label-informed"),
    ("- For agent C: nothing to score for detection.",
     "- Not reached: the number of October rows that match the predicate.\n- For agent C: nothing to score for detection."),
]
for a, b in R:
    assert a in s, a[:50]
    s = s.replace(a, b)
import re
s = re.sub(r"Updated \d\d:\d\d IST 15 Nov", f"Updated {t} IST 15 Nov", s, count=1)
open(p, "w", encoding="utf-8").write(s); print("ok", t)
