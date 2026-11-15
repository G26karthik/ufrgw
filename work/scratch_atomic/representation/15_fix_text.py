"""One-off corrections to section 9 of FINDINGS.md after the tie fix in 14_hist_class.py (a tie is no longer counted as
agreement on legit-labelled rows). cwd=work. argv[1] = HH:MM from `date`."""
import re
import sys
p = "scratch_atomic/representation/FINDINGS.md"; s = open(p, encoding="utf-8").read(); t = sys.argv[1]
R = [
    ("| N Aug | 5,435 | 1,428 | .877 | .745 | .181 |", "| N Aug | 5,435 | 1,428 | .877 | .745 | .169 |"),
    ("| N Sep | 9,712 | 2,359 | .881 | .769 | .195 |", "| N Sep | 9,712 | 2,359 | .881 | .769 | .172 |"),
    ("| D Aug | 5,435 | 1,393 | .896 | .765 | .182 |", "| D Aug | 5,435 | 1,393 | .896 | .765 | .172 |"),
    ("| D Sep | 9,712 | 2,307 | .896 | .782 | .195 |", "| D Sep | 9,712 | 2,307 | .896 | .782 | .174 |"),
    ("Observed .18 to .20:\n  the earlier class carries a small amount of information (about +.07 to +.09), far below the decoder (.75 to .78).",
     "Observed .169 to .174:\n  the earlier class carries a small amount of information (about +.055 to +.07), far below the decoder (.75 to .78).\n"
     "  A tie (`hist_class` = none) never counts as agreement."),
    ("decoder == label .745 / .744; hist == label .168 / .226;", "decoder == label .745 / .744; hist == label .157 / .216;"),
    ("decoder .772 / .760; hist .191 / .207; 838 / 265 against 12 / 9.", "decoder .772 / .760; hist .168 / .183; 838 / 265 against 12 / 9."),
    ("hist == label .188 / .131 / .186; decoder right where they differ 657 / 20 / 59,", "hist == label .188 / .046 / .178; decoder right where they differ 657 / 20 / 59,"),
    ("N Sep (1,599 / 460 / 300): .212 / .141 / .183; 934 / 56 / 113 against 19 / 0 / 2.",
     "N Sep (1,599 / 460 / 300): .212 / .046 / .150; 934 / 56 / 113 against 19 / 0 / 2. Most `hist_n = 2` rows are ties (two different classes)."),
    ("hist == label .181 / .485 / .110; decoder right where they differ", "hist == label .181 / .485 / .000; decoder right where they differ"),
    ("N Sep (1,770 / 142 / 447): .207 / .218 / .136; 1,018 / 81 / 4 against 20 / 1 / 0.", "N Sep (1,770 / 142 / 447): .207 / .218 / .016; 1,018 / 81 / 4 against 20 / 1 / 0."),
    ("- D is within +-.04 of N on every share.", "- D is within +-.04 of N on every share in lines with about 90 or more rows; smaller cells differ more."),
    ("so its earlier class is close to uninformative (.18 to .20 against\n  .10 to .12 under independence).",
     "so its earlier class is close to uninformative (.17 against\n  .10 to .12 under independence). Camouflage labels in the history cannot explain this: of the 1,624 September home-device P2P\n"
     "  predicate fraud rows about 500 are expected under the shuffle (section 8.5), so about two thirds are real recurrence, and a\n"
     "  persistent class would put the agreement well above .4."),
    ("was not tested further (for example by class pair, or by whether the earlier label was a fraud-path row).",
     "was not tested further. Not reached: the `hist_class` x label crosstab, and whether the earlier label was a fraud-path row."),
]
for a, b in R:
    assert a in s, a[:60]
    s = s.replace(a, b)
s = re.sub(r"Updated \d\d:\d\d IST 15 Nov", f"Updated {t} IST 15 Nov", s, count=1)
open(p, "w", encoding="utf-8").write(s); print("ok", t)
