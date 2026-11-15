"""Independent end check of the full run. Run from work/: PYTHONPATH=. python scratch_llm/verify_full.py"""
import collections
import json
from pathlib import Path

import polars as pl

from common import LABELS
from train import SAMPLE

HERE = Path(__file__).resolve().parent
ids = pl.read_csv(SAMPLE)["id"].to_list()
rows = [json.loads(l) for l in open(HERE / "full_predictions.jsonl")]
lab = {r["txn_id"]: r["label"] for r in rows}
print("sample rows", len(ids), "| prediction lines", len(rows), "| unique ids", len(lab),
      "| ids not in sample", len(set(lab) - set(ids)), "| missing", len(set(ids) - set(lab)),
      "| labels outside vocabulary", sum(v not in LABELS for v in lab.values()))
cfg, valid, bad, cost = collections.Counter(), 0, collections.Counter(), 0.0
for l in open(HERE / "full_requests.jsonl", encoding="utf-8"):
    r = json.loads(l)
    cfg[(r["api"], r["model"], r["mode"], r["builder"], r["static_hash"])] += 1
    cost += r["cost"]
    if r.get("valid"):
        valid += 1
    else:
        bad[(r.get("http"), str(r.get("error"))[:80])] += 1
print("request attempts", sum(cfg.values()), "| valid", valid, "| cost $%.4f" % cost)
print("configurations (must be one):", dict(cfg))
print("failed attempts:", dict(bad))
print("label counts:", collections.Counter(lab.values()).most_common())
csv = HERE / "llm_direct.csv"
if csv.exists():
    d = pl.read_csv(csv)
    print("llm_direct.csv:", d.columns, d.height, "| ids in sample order", d["id"].to_list() == ids,
          "| labels equal predictions", d["label"].to_list() == [lab.get(t) for t in ids])
else:
    print("llm_direct.csv: not written")
