"""Blend saved probs from several train.py tags, re-tune biases on pooled OOF, write submission.

usage (cwd=work): python blend.py --tags a,b,c [--weights 1,1,2] [--geo] --out NAME
Needs oof/<tag>.npy + oof/<tag>_ids.parquet for every tag; preds/<tag>_test.npy too unless --val-only.
"""
import argparse, json
import numpy as np
import polars as pl
from common import ROOT
from train import tune_bias, report, write_sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True); ap.add_argument("--weights", default="")
    ap.add_argument("--geo", action="store_true", help="weighted mean of log-probs instead of probs")
    ap.add_argument("--out", required=True); ap.add_argument("--val-only", action="store_true")
    a = ap.parse_args()
    tags = a.tags.split(",")
    w = np.array([float(x) for x in a.weights.split(",")] if a.weights else [1.0] * len(tags)); w /= w.sum()
    assert len(w) == len(tags), f"{len(w)} weights for {len(tags)} tags"
    ids0 = pl.read_parquet(ROOT / "oof" / f"{tags[0]}_ids.parquet")
    for t in tags[1:]:
        assert pl.read_parquet(ROOT / "oof" / f"{t}_ids.parquet")["txn_id"].equals(ids0["txn_id"]), f"{t}: OOF rows differ"

    def mix(paths):
        if a.geo:
            P = np.exp(sum(wi * np.log(np.load(p) + 1e-12) for wi, p in zip(w, paths)))
            return (P / P.sum(1, keepdims=True)).astype(np.float32)
        return sum(wi * np.load(p) for wi, p in zip(w, paths)).astype(np.float32)

    y = ids0["y"].to_numpy().astype(np.int64); fold = ids0["fold"].to_numpy()
    P = mix([ROOT / "oof" / f"{t}.npy" for t in tags])
    for t in tags:
        report(f"[{t}] OOF", y, np.load(ROOT / "oof" / f"{t}.npy"), np.array(json.loads((ROOT / "oof" / f"{t}_meta.json").read_text())["bias"]))
    b = tune_bias(P, y)
    res = {"tags": tags, "weights": w.tolist(), "geo": a.geo, "bias": b.tolist()}
    res["pooled_argmax"], res["pooled_biased"] = report("BLEND pooled OOF", y, P, b)
    for fi in np.unique(fold):
        res[f"fold{fi}_biased"] = report(f"BLEND fold{fi}", y[fold == fi], P[fold == fi], b)[1]
    np.save(ROOT / "oof" / f"{a.out}.npy", P); ids0.write_parquet(ROOT / "oof" / f"{a.out}_ids.parquet")
    (ROOT / "oof" / f"{a.out}_meta.json").write_text(json.dumps(res, indent=1))
    if not a.val_only:
        for t in tags:
            assert (ROOT / "preds" / f"{t}_test.npy").stat().st_mtime > (ROOT / "oof" / f"{t}.npy").stat().st_mtime, \
                f"{t}: test preds older than OOF; rerun train.py --mode full"
        PT = mix([ROOT / "preds" / f"{t}_test.npy" for t in tags])
        np.save(ROOT / "preds" / f"{a.out}_test.npy", PT)
        write_sub(PT, b, a.out)


if __name__ == "__main__":
    main()
