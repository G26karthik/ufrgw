"""alt_c2 rule chain with the fallback declared in scratch_mix/DECLARED.md (Build): if an assertion inside the last layer
(dd_rules_both, expected-id guard) refuses the new base, that layer's accepted rows (id and target as produced by
alt_c2.chain on subs/ts_judge_h6.csv) are applied instead; rows whose current label differs from the recorded source
label are listed as conflicts. No assertion is relaxed; the refusal is printed. Self-check first: the unpatched chain on
ts_judge_h6 must give the bytes of dd_final. usage (cwd=work, PYTHONPATH=.): python scratch_mix/chain_fb.py BASE.csv NAME
  writes subs/NAME.csv (never overwrites), notes/NAME_layers.csv, notes/NAME_diff.csv"""
import hashlib, sys
import polars as pl
import alt_c2
from common import ROOT, load_all

c = pl.col; inp, name = sys.argv[1:3]; out = ROOT / "subs" / f"{name}.csv"
assert not out.exists(), f"{out} exists; never overwrite"
tx = load_all(); h6b = pl.read_csv(ROOT / "subs" / "ts_judge_h6.csv")
h6, log_h6 = alt_c2.chain(h6b, tx)
assert alt_c2.csv_bytes(h6) == (ROOT / "subs" / "dd_final.csv").read_bytes(), "self-check failed"
print("self-check: driver on ts_judge_h6 == subs/dd_final.csv (byte-identical)")
acc = log_h6.filter(c("layer") == "DD"); real = alt_c2.DD


class Fallback:
    @staticmethod
    def apply(s, tx):
        try: return real.apply(s, tx)
        except AssertionError as e:
            print("DD layer refused this base:", e, "-> declared fallback: its", acc.height, "accepted rows from the dd_final chain")
            cur = dict(zip(s["id"], s["label"])); m = dict(zip(acc["id"], acc["to"]))
            conf = [(i, cur[i], f, t) for i, f, t in zip(acc["id"], acc["from"], acc["to"]) if cur[i] != f]
            print("conflicts (id, label here, recorded source label, target):", conf)
            return s.with_columns(pl.Series("label", [m.get(i, l) for i, l in zip(s["id"], s["label"])]))


alt_c2.DD = Fallback
base = pl.read_csv(ROOT / inp); new, log = alt_c2.chain(base, tx)
assert new.height == h6.height and new["id"].equals(h6["id"])
for k in log_h6["layer"].unique(maintain_order=True):
    a, b = log_h6.filter(c("layer") == k), log.filter(c("layer") == k)
    print(f"{k:8s} h6 base {a.height:3d} rows | this base {b.height:3d} rows | same rows and targets: {a.sort('id').equals(b.sort('id'))}"
          f" | only here: {sorted(set(b['id']) - set(a['id']))} | only h6: {sorted(set(a['id']) - set(b['id']))}")
diff = h6.rename({"label": "dd_final"}).with_columns(new["label"].alias(name), base["label"].alias("base_before_rules"), h6b["label"].alias("h6_before_rules")).filter(c("dd_final") != c(name))
print("rows that differ from dd_final:", diff.height, diff.group_by("dd_final", name).len().sort("len", descending=True).rows())
new.write_csv(out); log.write_csv(ROOT / "notes" / f"{name}_layers.csv"); diff.write_csv(ROOT / "notes" / f"{name}_diff.csv")
print("wrote", out, "sha256", hashlib.sha256(out.read_bytes()).hexdigest())
