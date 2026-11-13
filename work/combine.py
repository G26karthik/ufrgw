"""Stack small validated post-rule deltas onto a base submission.
Each delta = rows where subs/<rule>.csv differs from subs/<ref>.csv (the rule was built and gated on top of ref).
shortcut: transplants row diffs instead of re-running each rule on the new base; fine while the rules touch disjoint rows
(asserted below). Upgrade to re-running the rule scripts on the base tag if they start to overlap.
usage (cwd=work): python combine.py BASE OUT ref:rule [ref:rule ...]   e.g. ts_judge_c2s6 final ts_judge_h6:art_acct
"""
import sys
import polars as pl
from common import ROOT

base, out, pairs = sys.argv[1], sys.argv[2], [p.split(":") for p in sys.argv[3:]]
rd = lambda n: pl.read_csv(ROOT / "subs" / f"{n}.csv")
B = rd(base); lab = B["label"].to_list(); touched = set()
for ref, rule in pairs:
    R, U = rd(ref), rd(rule)
    assert R["id"].equals(B["id"]) and U["id"].equals(B["id"])
    idx = [i for i, (a, b) in enumerate(zip(R["label"], U["label"])) if a != b]
    assert not touched & set(idx), f"{rule} overlaps an earlier rule"
    touched |= set(idx)
    for i in idx:
        lab[i] = U["label"][i]
    print(f"{rule}: {len(idx)} rows from {ref}")
S = pl.DataFrame({"id": B["id"], "label": lab})
S.write_csv(ROOT / "subs" / f"{out}.csv")
print(f"wrote subs/{out}.csv; rows changed vs {base}: {sum(a != b for a, b in zip(B['label'], lab))}")
