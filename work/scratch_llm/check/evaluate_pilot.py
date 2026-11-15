"""Pilot evaluation (September, labels hidden from the LLM). Run with cwd = work.

Step 1 (before any label is read):  python scratch_llm/check/evaluate_pilot.py --seal
Step 2:                             python scratch_llm/check/evaluate_pilot.py [--allow-missing]
Self-test:                          python scratch_llm/check/evaluate_pilot.py --selftest

Step 2 refuses to run unless the predictions file hashes to the sealed SHA-256. Truth = raw train label
(== `y` of scratch_gap/dec_N_wt.npz, checked). Baseline = `C_q` of the same file on exactly the same rows.
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import HERE, LLM, WORK, find_predictions, read_jsonl, sha256, wilson  # noqa: E402

LABELS = ["legit", "mule_account", "velocity_bot", "layering_chain", "sim_swap", "account_takeover",
          "phishing_collect", "qr_swap", "refund_abuse", "fake_cashback", "loan_app_extortion", "merchant_collusion"]
L2I = {l: i for i, l in enumerate(LABELS)}
K = len(LABELS)
INVALID = K  # extra prediction column: missing / out-of-vocabulary / conflicting, always counted as wrong


def load_preds(path):
    return [p for _, rec in read_jsonl(path) for p in find_predictions(rec)]


def load_samples(path):
    """{sample name: [txn ids]} from A's ids file. Needs an id column and a sample-membership column."""
    import polars as pl
    d = pl.read_csv(path, infer_schema_length=0)
    idc = next((c for c in ("txn_id", "id") if c in d.columns), None)
    if idc is None:
        raise SystemExit(f"FAIL: no txn_id/id column in {path}: {d.columns}")
    leaked = [c for c in d.columns if c != idc and set(d[c].drop_nulls().unique().to_list()) & set(LABELS)]
    if leaked:
        print(f"CONTRACT VIOLATION: sample ids file carries label values in columns {leaked}")
    sc = next((c for c in ("subset", "sample", "set", "split", "group", "kind", "part", "source") if c in d.columns), None)
    if sc is None:
        raise SystemExit(f"FAIL: no sample-membership column in {path}: {d.columns}. Cannot separate natural from enriched.")
    out = defaultdict(list)
    for i, s in zip(d[idc].to_list(), d[sc].to_list()):
        for part in str(s).replace("+", "|").replace(",", "|").split("|"):   # one id may be in both samples
            out["natural" if "nat" in part.lower() else "enriched" if "enr" in part.lower() else part].append(i)
    return dict(out)


def coverage(pairs, ids):
    """Returns ({id: label id or INVALID}, [problem strings]). Every sample id must appear exactly once."""
    want, seen, prob = set(ids), defaultdict(list), []
    for i, l in pairs:
        seen[i].append(l)
    extra = set(seen) - want
    missing = want - set(seen)
    dup = {i for i, v in seen.items() if len(v) > 1 and i in want}
    conflict = {i for i in dup if len(set(map(str, seen[i]))) > 1}
    oov = {i for i, v in seen.items() if i in want and any(l not in L2I for l in v)}
    if len(set(ids)) != len(ids):
        prob.append(f"sample id list itself has {len(ids) - len(set(ids))} duplicate ids")
    for name, s in (("missing", missing), ("extra (not in sample)", extra), ("duplicated", dup),
                    ("conflicting duplicates", conflict), ("label out of vocabulary", oov)):
        if s:
            prob.append(f"{name}: {len(s)} ids, e.g. {sorted(s)[:3]}")
    got = {}
    for i in want:
        v = set(map(str, seen.get(i, [])))
        got[i] = L2I[next(iter(v))] if len(v) == 1 and next(iter(v)) in L2I else INVALID
    return got, prob


def confusion(t, p):
    C = np.zeros((K, K + 1), dtype=float)
    np.add.at(C, (np.asarray(t), np.asarray(p)), 1)
    return C


def prf(C):
    """Per-class precision, recall, F1 from a (K, K+1) confusion matrix (rows true, cols predicted)."""
    tp = np.diag(C[:, :K])
    ps, rs = C[:, :K].sum(0), C.sum(1)
    P = np.divide(tp, ps, out=np.zeros(K), where=ps > 0)
    R = np.divide(tp, rs, out=np.zeros(K), where=rs > 0)
    F = np.divide(2 * P * R, P + R, out=np.zeros(K), where=(P + R) > 0)
    return P, R, F


def summary(C):
    P, R, F = prf(C)
    sup = C.sum(1)
    pres = sup > 0
    tpf = C[1:, 1:K].sum()                       # fraud row predicted as any fraud class
    fp, fn = C[0, 1:K].sum(), C[1:, 0].sum() + C[1:, K].sum()
    bp = tpf / (tpf + fp) if tpf + fp else 0.0
    br = tpf / (tpf + fn) if tpf + fn else 0.0
    return dict(P=P, R=R, F=F, sup=sup, macro_present=float(F[pres].mean()) if pres.any() else 0.0,
                macro_12=float(F.mean()), acc=float(np.diag(C[:, :K]).sum() / max(C.sum(), 1)),
                bin_p=bp, bin_r=br, bin_f=2 * bp * br / (bp + br) if bp + br else 0.0)


def fmt_conf(C, title, ints=True):
    cols = [l[:6] for l in LABELS] + ["INVAL"]
    s = [f"\n{title} (rows = true, columns = predicted)\n", "| true \\ pred | " + " | ".join(cols) + " |",
         "|---|" + "---|" * len(cols)]
    for k in range(K):
        if C[k].sum() > 0:
            s.append(f"| {LABELS[k]} | " + " | ".join((f"{int(round(v))}" if ints else f"{v:.0f}") if v else "." for v in C[k]) + " |")
    return "\n".join(s)


def table(Cl, Cb, title, ci=True):
    a, b = summary(Cl), summary(Cb)
    s = [f"\n{title}\n", "| class | n | LLM P | LLM R | LLM F1 | base P | base R | base F1 |" + (" LLM recall 95% CI |" if ci else ""),
         "|---|---|---|---|---|---|---|---|" + ("---|" if ci else "")]
    for k in range(K):
        if a["sup"][k] == 0 and Cl[:, k].sum() == 0 and Cb[:, k].sum() == 0:
            continue
        row = f"| {LABELS[k]} | {a['sup'][k]:.0f} | {a['P'][k]:.3f} | {a['R'][k]:.3f} | {a['F'][k]:.3f} | {b['P'][k]:.3f} | {b['R'][k]:.3f} | {b['F'][k]:.3f} |"
        if ci:
            lo, hi = wilson(int(round(Cl[k, k])), int(round(a["sup"][k])))
            row += f" {lo:.2f}-{hi:.2f} |" if a["sup"][k] else " n/a |"
        s.append(row)
    s.append(f"\n- macro-F1 over classes with support > 0: LLM {a['macro_present']:.4f}, baseline {b['macro_present']:.4f}")
    s.append(f"- macro-F1 over all 12 classes (absent class = 0): LLM {a['macro_12']:.4f}, baseline {b['macro_12']:.4f}")
    s.append(f"- accuracy: LLM {a['acc']:.4f}, baseline {b['acc']:.4f}")
    s.append(f"- fraud versus legit (any fraud class counts): LLM P {a['bin_p']:.3f} R {a['bin_r']:.3f} F1 {a['bin_f']:.3f}; "
             f"baseline P {b['bin_p']:.3f} R {b['bin_r']:.3f} F1 {b['bin_f']:.3f}")
    return "\n".join(s)


def report(name, ids, got, truth, base, pop=None):
    """ids: sample ids; got/truth/base: dicts id -> label id. pop: September class counts -> adds the reweighted view."""
    ids = sorted(set(ids))
    t = np.array([truth[i] for i in ids]); p = np.array([got[i] for i in ids]); b = np.array([base[i] for i in ids])
    Cl, Cb = confusion(t, p), confusion(t, b)
    n, nf = len(ids), int((t > 0).sum())
    out = [f"\n## {name} sample: {n} rows, {nf} fraud, {n - nf} legit, {int((p == INVALID).sum())} without a valid LLM label"]
    out.append(table(Cl, Cb, f"### {name}: per class, LLM against baseline C_q, same rows (unweighted)"))
    out.append(fmt_conf(Cl, f"### {name}: LLM confusion counts"))
    out.append(fmt_conf(Cb, f"### {name}: baseline confusion counts"))
    lw, bw = int(((p == t) & (b != t)).sum()), int(((p != t) & (b == t)).sum())
    out.append(f"\n- rows where LLM and baseline differ: {int((p != b).sum())}; LLM right and baseline wrong: {lw}; "
               f"baseline right and LLM wrong: {bw}; both wrong, different labels: {int(((p != t) & (b != t) & (p != b)).sum())}")
    nl = int((t == 0).sum())
    fpl, fpb = int(((t == 0) & (p != 0)).sum()), int(((t == 0) & (b != 0)).sum())
    lo, hi = wilson(fpl, nl); lob, hib = wilson(fpb, nl)
    out.append(f"- legit rows called fraud: LLM {fpl}/{nl} (95% CI {lo:.4f}-{hi:.4f}); baseline {fpb}/{nl} (95% CI {lob:.4f}-{hib:.4f})")
    if nf:
        kl, kb = int(((t > 0) & (p == t)).sum()), int(((t > 0) & (b == t)).sum())
        lo, hi = wilson(kl, nf); lob, hib = wilson(kb, nf)
        out.append(f"- fraud rows with the exact class: LLM {kl}/{nf} (95% CI {lo:.2f}-{hi:.2f}); baseline {kb}/{nf} (95% CI {lob:.2f}-{hib:.2f})")
    if pop is not None:
        cnt = np.bincount(t, minlength=K).astype(float)
        w = np.divide(pop, cnt, out=np.zeros(K), where=cnt > 0)
        out.append(f"\n### {name}: reweighted to the September class distribution (weight = population count / sample count per true class)\n")
        out.append("Weights: " + ", ".join(f"{LABELS[k]} {w[k]:.0f}" for k in range(K) if cnt[k]))
        out.append(table(Cl * w[:, None], Cb * w[:, None], f"### {name}: reweighted per class (n = September population)", ci=False))
        out.append(fmt_conf(Cl * w[:, None], f"### {name}: LLM confusion, reweighted counts"))
        out.append(fmt_conf(Cb * w[:, None], f"### {name}: baseline confusion, reweighted counts"))
        lo, hi = wilson(fpl, nl)
        swamp = f", more than any fraud class has rows ({int(pop[1:].max())} at most)" if w[0] > pop[1:].max() else ""
        out.append(f"\nLimits of this sample. The unweighted macro-F1 above is NOT representative: fraud is {nf}/{n} of the sample "
                   f"but {pop[1:].sum() / pop.sum():.2%} of September. Recall per class rests on {int(cnt[1:][cnt[1:] > 0].min()) if nf else 0}-"
                   f"{int(cnt[1:].max()) if nf else 0} rows (CI column above). Precision after reweighting rests on {nl} legit rows, each "
                   f"standing for {w[0]:.0f} rows: one legit row called fraud adds {w[0]:.0f} false positives{swamp}. "
                   f"With {fpl}/{nl} legit errors the 95% interval for the false-positive rate is "
                   f"{lo:.4f}-{hi:.4f}, that is {lo * pop[0]:.0f}-{hi * pop[0]:.0f} false positives against {int(pop[1:].sum())} fraud rows. "
                   + (f"So this sample estimates recall only; its reweighted precision and macro-F1 hang on {fpl} legit error(s)."
                      if nl < 100 else "The reweighted macro-F1 is an estimate with wide error: per-class recall from 15-20 rows, precision from this false-positive interval."))
    diff = [(i, LABELS[truth[i]], LABELS[got[i]] if got[i] < K else "INVALID", LABELS[base[i]]) for i in ids if got[i] != base[i]]
    return "\n".join(out), diff


def selftest():
    import json, tempfile, os
    assert find_predictions({"response": json.dumps({"predictions": [{"txn_id": "TX1", "label": "legit"}]})}) == [("TX1", "legit")]
    got, prob = coverage([("a", "legit"), ("b", "qr_swap")], ["a", "b"])
    assert not prob and got == {"a": 0, "b": 7}
    got, prob = coverage([("a", "legit"), ("a", "qr_swap"), ("c", "legit"), ("d", "nonsense")], ["a", "b", "d"])
    assert len(prob) == 5 and got == {"a": INVALID, "b": INVALID, "d": INVALID}, prob   # missing, extra, dup, conflict, oov
    C = confusion([0, 0, 1, 1, 7], [0, 1, 1, 0, INVALID])
    P, R, F = prf(C)
    assert (P[0], R[0], P[1], R[1], R[7]) == (0.5, 0.5, 0.5, 0.5, 0.0)
    s = summary(C)
    assert abs(s["macro_present"] - (0.5 + 0.5 + 0) / 3) < 1e-9 and abs(s["bin_p"] - 0.5) < 1e-9 and abs(s["bin_r"] - 1 / 3) < 1e-9
    lo, hi = wilson(0, 60)
    assert lo == 0 and 0.055 < hi < 0.065
    truth = {f"t{i}": (0 if i < 6 else 1) for i in range(9)}
    got = {**{f"t{i}": 0 for i in range(9)}, "t0": 1, "t6": 1}
    rep, diff = report("x", list(truth), got, truth, {k: 0 for k in truth}, pop=np.array([600, 30] + [0.] * 10))
    assert "legit 100" in rep and "mule_account 10" in rep and len(diff) == 2
    Cw = confusion([truth[k] for k in sorted(truth)], [got[k] for k in sorted(truth)]) * np.array([100, 10] + [0.] * 10)[:, None]
    assert Cw[0, 1] == 100 and Cw[1, 1] == 10 and abs(prf(Cw)[0][1] - 10 / 110) < 1e-9   # one legit error swamps precision
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "ids.csv"), "w") as f:
        f.write("txn_id,sample\nA,natural\nB,enriched\nC,natural|enriched\n")
    assert load_samples(os.path.join(d, "ids.csv")) == {"natural": ["A", "C"], "enriched": ["B", "C"]}
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(LLM / "pilot_predictions.jsonl"))
    ap.add_argument("--ids", default=str(LLM / "pilot_sample_ids.csv"))
    ap.add_argument("--seal", action="store_true", help="write the SHA-256 of the predictions file, read no labels")
    ap.add_argument("--allow-missing", action="store_true", help="score missing/invalid predictions as wrong instead of stopping")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not Path(a.pred).exists():
        raise SystemExit(f"REFUSED: predictions file {a.pred} does not exist")
    h = sha256(a.pred)
    stem = Path(a.pred).stem                       # one seal and one report per predictions file
    SEAL = HERE / f"{stem}.sha256"
    if a.seal:
        if SEAL.exists() and SEAL.read_text().split()[0] != h:
            raise SystemExit(f"REFUSED: seal exists with a different hash ({SEAL.read_text().split()[0]} != {h}); predictions changed after sealing")
        SEAL.write_text(f"{h}  {Path(a.pred).name}\n")
        print(f"sealed {h}")
        return
    if not SEAL.exists():
        raise SystemExit("REFUSED: run --seal first (SHA-256 of the predictions must be recorded before labels are revealed)")
    if SEAL.read_text().split()[0] != h:
        raise SystemExit(f"REFUSED: predictions file changed after sealing (now {h})")

    samples = load_samples(a.ids)
    all_ids = sorted({i for v in samples.values() for i in v})
    got, prob = coverage(load_preds(a.pred), all_ids)
    out = [f"# Pilot evaluation\n\npredictions sha256 `{h}` (matches seal). Samples: " + ", ".join(f"{k} {len(v)}" for k, v in samples.items())
           + f"; distinct ids {len(all_ids)}; ids in more than one sample {sum(len(v) for v in samples.values()) - len(all_ids)}"]
    out.append("\nCoverage: " + ("PASS (every sample id exactly once, no extras, all labels in vocabulary)" if not prob else "FAIL: " + "; ".join(prob)))
    if prob and not a.allow_missing:
        print("\n".join(out))
        raise SystemExit("FAIL: coverage. Labels NOT revealed. Re-run with --allow-missing to score missing/invalid rows as wrong.")

    # ---- labels are read only from here on ----
    import polars as pl
    rows = pl.read_parquet(WORK / "scratch_gap" / "N_sep_rows.parquet", columns=["txn_id"])["txn_id"].to_list()
    z = np.load(WORK / "scratch_gap" / "dec_N_wt.npz")
    y, cq = z["y"], z["C_q"]
    raw = pl.read_parquet(WORK / "data" / "train.parquet", columns=["txn_id", "label"])
    raw = dict(zip(raw["txn_id"].to_list(), raw["label"].to_list()))
    assert all(LABELS[y[k]] == raw[r] for k, r in enumerate(rows)), "npz y is not the raw train label"
    truth, base = dict(zip(rows, y.tolist())), dict(zip(rows, cq.tolist()))
    nob = [i for i in all_ids if i not in truth]
    if nob:
        raise SystemExit(f"FAIL: {len(nob)} sample ids are not September baseline rows, e.g. {nob[:3]}")
    pop = np.bincount(y, minlength=K).astype(float)
    sb = summary(confusion(y, cq))
    out.append(f"\nTruth = raw September train label (509,996 rows, {int(pop[1:].sum())} fraud = {pop[1:].sum() / pop.sum():.2%}). "
               f"Baseline C_q on ALL of September: macro-F1 {sb['macro_12']:.4f}, fraud-vs-legit P {sb['bin_p']:.3f} R {sb['bin_r']:.3f}. "
               f"Compare with its score on each sample below to see the sampling noise.")
    diffs = []
    for name, ids in samples.items():
        rep, diff = report(name, ids, got, truth, base, pop=None if name == "natural" else pop)
        out.append(rep)
        diffs += [(name,) + d for d in diff]
        if name == "natural":
            nf = sum(truth[i] > 0 for i in set(ids))
            out.append(f"\nLimits of the natural sample: {len(set(ids))} rows hold {nf} fraud rows across 11 fraud classes, so per-class "
                       f"fraud F1 here is 0-or-1 noise and macro-F1 moves by about {1 / max(nf, 1) / 2:.2f}-{1 / max(nf, 1):.2f} per single row. "
                       f"It can show: whether the LLM over-flags legit rows (false-positive rate, interval above) and gross failure. "
                       f"It cannot rank LLM against baseline on fraud classes or predict leaderboard macro-F1.")
    if len(samples) > 1:
        # both samples are random within each true class, so pooled rows reweighted per class give the best single estimate
        rep, _ = report("pooled (natural + enriched, post-stratified by true class)", all_ids, got, truth, base, pop=pop)
        out.append(rep)
    pl.DataFrame(diffs, schema=["sample", "txn_id", "true", "llm", "baseline"], orient="row").write_csv(HERE / f"{stem}_diff_rows.csv")
    out.append(f"\nRows where LLM and baseline differ: `check/{stem}_diff_rows.csv` ({len(diffs)} rows; true / llm / baseline: "
               + ", ".join(f"{k[0]}/{k[1]}/{k[2]} x{v}" for k, v in Counter((d[2], d[3], d[4]) for d in diffs).most_common(15)) + ")")
    txt = "\n".join(out)
    (HERE / f"{stem}_eval.md").write_text(txt, encoding="utf-8")
    print(txt)
    if prob:
        raise SystemExit("FAIL: coverage (scored with missing/invalid counted as wrong)")


if __name__ == "__main__":
    main()
