"""Can LLM answers improve the baseline label? Override rules on top of C_q, scored on the two sealed pilots.

  python scratch_llm/check/hybrid_eval.py            (cwd = work; writes check/hybrid_eval.md)
  python scratch_llm/check/hybrid_eval.py --selftest

No API call. Uses: pilot sample ids, sealed pilot predictions, the Decisions pilot's logged `probabilities`, true September
labels and the baseline C_q (scratch_gap/dec_N_wt.npz). Families, each on top of the baseline label, LLM confidence
(probability of the LLM's chosen label) >= t:
  D   baseline legit, LLM fraud class           -> LLM label
  T   both fraud, different classes             -> LLM class
  V   baseline fraud, LLM legit                 -> legit
  ALL any disagreement                          -> LLM label
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import HERE, LLM, WORK, find_predictions, read_jsonl, sha256  # noqa: E402
from evaluate_pilot import K, LABELS, L2I, confusion, load_samples, summary  # noqa: E402

THRESH = [0, .5, .7, .9, .95, .99]
FAMILIES = {"D": lambda b, l: (b == 0) & (l > 0), "T": lambda b, l: (b > 0) & (l > 0) & (b != l),
            "V": lambda b, l: (b > 0) & (l == 0), "ALL": lambda b, l: b != l}


def hybrid(b, l, conf, fam, t):
    """Hybrid labels and the override mask."""
    m = FAMILIES[fam](b, l) & (conf >= t)
    return np.where(m, l, b), m


def counts(y, b, h, m):
    return dict(overridden=int(m.sum()), corrected=int((m & (b != y) & (h == y)).sum()), damaged=int((m & (b == y) & (h != y)).sum()),
                unchanged_wrong=int((m & (b != y) & (h != y)).sum()))


def macro(y, p, w=None):
    C = confusion(y, p)
    if w is not None:
        C = C * w[:, None]
    s = summary(C)
    return s["macro_present"]


def table(y, b, l, conf, nat, w, thresholds):
    """Rows of the result table and the list of (family, t) that pass the support rule."""
    rows, ok = [], []
    mb_n, mb_p = macro(y[nat], b[nat]), macro(y, b, w)
    for fam in FAMILIES:
        for t in thresholds:
            h, m = hybrid(b, l, conf, fam, t)
            cn, cp = counts(y[nat], b[nat], h[nat], m[nat]), counts(y, b, h, m)
            rows.append((fam, t, cn, macro(y[nat], h[nat]) - mb_n, cp, macro(y, h, w) - mb_p))
            if cn["corrected"] > cn["damaged"] and cp["corrected"] > cp["damaged"] and cp["corrected"] >= 5:
                ok.append((fam, t))
    return rows, ok, mb_n, mb_p


def fmt(rows, mb_n, mb_p, with_t=True):
    s = [f"Baseline macro-F1: natural {mb_n:.4f} (classes present, unweighted), pooled reweighted {mb_p:.4f}.\n",
         "| family | t | natural: overridden | corrected | damaged | unchanged-wrong | macro-F1 hybrid (change) | pooled: overridden | corrected | damaged | unchanged-wrong | reweighted macro-F1 hybrid (change) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for fam, t, cn, dn, cp, dp in rows:
        s.append(f"| {fam} | {t if with_t else 'label only'} | {cn['overridden']} | {cn['corrected']} | {cn['damaged']} | {cn['unchanged_wrong']} | {mb_n + dn:.4f} ({dn:+.4f}) "
                 f"| {cp['overridden']} | {cp['corrected']} | {cp['damaged']} | {cp['unchanged_wrong']} | {mb_p + dp:.4f} ({dp:+.4f}) |")
    return "\n".join(s)


def selftest():
    y = np.array([0, 0, 1, 1, 2, 2, 0, 3])
    b = np.array([0, 0, 0, 1, 1, 2, 2, 3])
    l = np.array([1, 0, 1, 2, 2, 0, 0, 3])
    conf = np.array([.6, 1, .95, .8, .99, .7, .4, 1])
    h, m = hybrid(b, l, conf, "D", 0)
    assert m.tolist() == [True, False, True, False, False, False, False, False] and counts(y, b, h, m) == dict(overridden=2, corrected=1, damaged=1, unchanged_wrong=0)
    h, m = hybrid(b, l, conf, "D", .9)
    assert counts(y, b, h, m) == dict(overridden=1, corrected=1, damaged=0, unchanged_wrong=0)
    h, m = hybrid(b, l, conf, "T", 0)
    assert counts(y, b, h, m) == dict(overridden=2, corrected=1, damaged=1, unchanged_wrong=0)       # rows 3 (damaged) and 4 (corrected)
    h, m = hybrid(b, l, conf, "V", .5)
    assert counts(y, b, h, m) == dict(overridden=1, corrected=0, damaged=1, unchanged_wrong=0)       # row 5; row 6 below threshold
    h, m = hybrid(b, l, conf, "ALL", 0)
    assert int(m.sum()) == 6 and counts(y, b, h, m)["corrected"] == 3
    assert abs(macro(y, y) - 1.0) < 1e-12 and macro(y, b) < 1
    rows, ok, _, _ = table(y, b, l, conf, np.ones(8, bool), np.ones(K), [0, .9])
    assert len(rows) == 8 and ok == []                    # nothing reaches 5 corrected rows
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import polars as pl
    files = {"decisions": (LLM / "pilot_predictions_decisions.jsonl", LLM / "pilot_requests_decisions.jsonl"),
             "astra": (LLM / "pilot_predictions_astra_full.jsonl", None)}
    for name, (pf, _) in files.items():
        seal = HERE / f"{pf.stem}.sha256"
        if not seal.exists() or seal.read_text().split()[0] != sha256(pf):
            raise SystemExit(f"REFUSED: {pf.name} is not the sealed file")
    samples = load_samples(LLM / "pilot_sample_ids.csv")
    ids = sorted({i for v in samples.values() for i in v})
    rows = pl.read_parquet(WORK / "scratch_gap" / "N_sep_rows.parquet", columns=["txn_id"])["txn_id"].to_list()
    z = np.load(WORK / "scratch_gap" / "dec_N_wt.npz")
    pos = {t: k for k, t in enumerate(rows)}
    idx = np.array([pos[i] for i in ids])
    y, b = z["y"][idx], z["C_q"][idx]
    pop = np.bincount(z["y"], minlength=K).astype(float)
    cnt = np.bincount(y, minlength=K).astype(float)
    w = np.divide(pop, cnt, out=np.zeros(K), where=cnt > 0)
    nat = np.isin(ids, samples["natural"])
    out = ["# Hybrid rules: LLM overrides on top of the baseline label C_q (pilot rows, September)\n",
           f"Rows: natural {int(nat.sum())} ({int((y[nat] > 0).sum())} fraud), pooled {len(ids)} ({int((y > 0).sum())} fraud) reweighted per true class to the "
           f"September distribution (weight = population count / pooled count, legit {w[0]:.0f}). Counts are pilot rows, not reweighted. "
           "corrected = baseline wrong, hybrid right; damaged = baseline right, hybrid wrong; unchanged-wrong = overridden, wrong before and after."]
    verdicts = {}
    for name, (pf, rf) in files.items():
        lab = dict(p for _, rec in read_jsonl(pf) for p in find_predictions(rec))
        l = np.array([L2I[lab[i]] for i in ids])
        conf = np.ones(len(ids))
        if rf is not None:       # probability of the chosen label, from the logged answers
            pr = {}
            for _, rec in read_jsonl(rf):
                if rec.get("valid"):
                    for x in rec["response"]["answers"]:
                        pr[x["name"]] = (x["choice"], next(d["probability"] for d in x["probabilities"] if d["value"] == x["choice"]), x.get("confidence"))
            assert all(pr[i][0] == lab[i] for i in ids), "logged choice differs from the predictions file"
            conf = np.array([pr[i][1] for i in ids])
            cf = np.array([np.nan if pr[i][2] is None else pr[i][2] for i in ids], dtype=float)
            out.append(f"\n## Decisions lean pilot (`gpt-6-luna`, the setup of the full run)\n\nConfidence = logged probability of the chosen label. "
                       f"Distribution over the 525 rows: " + ", ".join(f">= {t}: {int((conf >= t).sum())}" for t in THRESH[1:])
                       + f"; minimum {conf.min():.3f}. Logged `confidence` field equals it on {int(np.isclose(cf, conf).sum())} rows"
                       + f" (mean {np.nanmean(cf):.3f} against {conf.mean():.3f}).")
            out.append(f"Disagreements with the baseline: {int((l != b).sum())} rows; of them D {int(FAMILIES['D'](b, l).sum())}, T {int(FAMILIES['T'](b, l).sum())}, V {int(FAMILIES['V'](b, l).sum())}.\n")
            rows_t, ok, mb_n, mb_p = table(y, b, l, conf, nat, w, THRESH)
            out.append(fmt(rows_t, mb_n, mb_p))
        else:
            out.append(f"\n## astra full-context pilot (`gpt-6-astra`), label only (no probabilities exist for it)\n\n"
                       f"Disagreements with the baseline: {int((l != b).sum())} rows; of them D {int(FAMILIES['D'](b, l).sum())}, T {int(FAMILIES['T'](b, l).sum())}, V {int(FAMILIES['V'](b, l).sum())}.\n")
            rows_t, ok, mb_n, mb_p = table(y, b, l, conf, nat, w, [0])
            out.append(fmt(rows_t, mb_n, mb_p, with_t=False))
        verdicts[name] = ok
        win = np.flatnonzero((l == y) & (b != y))
        out.append(f"\nRows where the LLM is right and the baseline wrong: {len(win)}"
                   + "".join(f"\n- {ids[k]} ({'natural' if nat[k] else 'enriched'}): true {LABELS[y[k]]}, baseline {LABELS[b[k]]}, LLM confidence "
                             + (f"{conf[k]:.3f}" if rf is not None else "n/a") for k in win))
        lose = (b == y) & (l != y)
        out.append(f"Rows where the baseline is right and the LLM wrong: {int(lose.sum())}"
                   + (f"; their LLM confidence: >= 0.9 on {int((conf[lose] >= .9).sum())}, >= 0.99 on {int((conf[lose] >= .99).sum())}, median {np.median(conf[lose]):.3f}" if rf is not None and lose.any() else ""))
    out.append("\n## Verdict\n")
    for name, ok in verdicts.items():
        out.append(f"- {name}: family / threshold with corrected > damaged on BOTH the natural and the pooled view and at least 5 corrected rows: "
                   + (", ".join(f"{f} at t={t}" for f, t in ok) if ok else "NONE"))
    if not any(verdicts.values()):
        out.append("\nNo hybrid rule is supported by the pilot.")
    out.append("\nLimits: 525 rows (17 fraud rows in the natural sample, 15-20 per fraud class pooled); thresholds are chosen and scored on the same rows, "
               "which favours the rules, and they still do not pass; pilot batches were 33% fraud against about 4% in the full run, so LLM confidences "
               "in the full run may be shifted; the baseline C_q here is the September label-hidden decode, not the October submission `dd_final.csv`.")
    txt = "\n".join(out)
    (HERE / "hybrid_eval.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
