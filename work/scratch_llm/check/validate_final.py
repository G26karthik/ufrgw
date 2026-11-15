"""Integrity check of a finished llm_direct.csv. Fails loudly (exit 1) on any problem, including partial coverage.

  python scratch_llm/check/validate_final.py --csv scratch_llm/llm_direct.csv --log "scratch_llm/<full run log>*.jsonl"
  python scratch_llm/check/validate_final.py --selftest

Every row must be backed by exactly one distinct valid label in the LLM logs and the CSV must carry that label:
no default, no fill from dd_final.csv, no conflicting predictions for one id.
"""
import argparse
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import N_TEST, WORK, find_predictions, find_usage, read_jsonl, sha256  # noqa: E402

LABELS = ["legit", "mule_account", "velocity_bot", "layering_chain", "sim_swap", "account_takeover",
          "phishing_collect", "qr_swap", "refund_abuse", "fake_cashback", "loan_app_extortion", "merchant_collusion"]


def read_csv(path):
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        r = list(csv.reader(f))
    return r[0], r[1:]


def logged(paths):
    """{id: Counter(label -> number of valid log records)}, and the ids whose records carry no usage object.
    Only the model's answer is read (not the prompt), and only from attempts that A accepted (valid is not False)."""
    by, nousage = defaultdict(Counter), set()
    for p in paths:
        for _, rec in read_jsonl(p):
            if isinstance(rec, dict) and (rec.get("valid") is False or rec.get("error")):
                continue
            has_u = bool(find_usage(rec))
            body = {k: v for k, v in rec.items() if k not in ("prompt", "request", "body")} if isinstance(rec, dict) else rec
            for i, l in set(find_predictions(body)):     # one record may hold raw and parsed copies of the same answer
                by[i][str(l)] += 1
                if not has_u:
                    nousage.add(i)
    return by, nousage


def choice_vs_argmax(paths):
    """Decisions API: (answers with probabilities, answers whose `choice` is not the top-probability value). Information
    only: the exported label must be the returned `choice`, whatever the probabilities say."""
    n = off = 0
    for p in paths:
        for _, rec in read_jsonl(p):
            ans = (rec.get("response") or {}).get("answers") if isinstance(rec, dict) and rec.get("valid") is not False else None
            for x in ans or []:
                pr = x.get("probabilities")
                if isinstance(pr, list) and pr:
                    n += 1
                    off += max(pr, key=lambda d: d.get("probability", 0)).get("value") != x.get("choice")
    return n, off


def backing(pred_path, log_paths):
    """A's predictions file against the logged API responses. Returns (fail list, info list)."""
    by, nousage = logged(log_paths)
    pred = [p for _, rec in read_jsonl(pred_path) for p in find_predictions(rec)]
    cnt = Counter(i for i, _ in pred)
    f = []
    dup = [i for i, c in cnt.items() if c > 1]
    nolog = [i for i, l in pred if not by.get(i)]
    diff = [i for i, l in pred if by.get(i) and set(by[i]) != {l}]
    multi = [i for i in cnt if sum(by.get(i, Counter()).values()) > 1]
    if dup:
        f.append(f"{len(dup)} ids more than once in {Path(pred_path).name}, e.g. {dup[:3]}")
    if nolog:
        f.append(f"{len(nolog)} predictions without a logged valid API response, e.g. {nolog[:3]}")
    if diff:
        f.append(f"{len(diff)} predictions differ from (or conflict within) the logged response label / choice, e.g. {diff[:3]}")
    if multi:
        f.append(f"{len(multi)} ids answered by more than one valid logged response, e.g. {multi[:3]}")
    nu = len(nousage & set(cnt))
    if nu:
        f.append(f"{nu} ids backed by a record without a usage object")
    return f, [f"{Path(pred_path).name}: {len(pred)} lines, {len(cnt)} distinct ids; ids with a logged valid response {len(cnt) - len(set(nolog))}; "
               f"label equal to the logged response {len(pred) - len(diff) - len(nolog)}"]


def validate(csv_path, log_paths, sample_path, dd_path, n=N_TEST, require_usage=False):
    fails, info = [], []
    head, rows = read_csv(csv_path)
    shead, srows = read_csv(sample_path)
    ids, labs = [r[0] for r in rows], [r[1] if len(r) > 1 else "" for r in rows]
    sids = [r[0] for r in srows]
    if head != ["id", "label"]:
        fails.append(f"header is {head}, expected ['id', 'label']")
    if len(rows) != n or len(sids) != n:
        fails.append(f"row count {len(rows)} (sample {len(sids)}), expected {n}")
    if ids != sids:
        fails.append(f"ids / order differ from the sample submission: {sum(a != b for a, b in zip(ids, sids))} positions differ, "
                     f"{len(set(sids) - set(ids))} sample ids absent, {len(set(ids) - set(sids))} extra ids")
    if len(set(ids)) != len(ids):
        fails.append(f"{len(ids) - len(set(ids))} duplicate ids in the CSV")
    oov = Counter(l for l in labs if l not in LABELS)
    if oov:
        fails.append(f"{sum(oov.values())} labels out of vocabulary: {dict(oov.most_common(5))}")
    by, nousage = logged(log_paths)
    valid = {i: {l for l in c if l in LABELS} for i, c in by.items()}
    backed = {i for i in set(ids) if len(valid.get(i, ())) == 1}
    none = [i for i in ids if not valid.get(i)]
    conflict = [i for i in ids if len(valid.get(i, ())) > 1]
    mismatch = [i for i, l in zip(ids, labs) if i in backed and l not in valid[i]]
    dup = sum(1 for i in backed if sum(by[i][l] for l in valid[i]) > 1)
    info.append(f"distinct ids with exactly one valid logged LLM label: {len(backed)} of {len(set(ids))}")
    if none:
        fails.append(f"PARTIAL COVERAGE: {len(none)} rows have no valid logged LLM prediction (default or fill?), e.g. {none[:3]}")
    if conflict:
        fails.append(f"{len(conflict)} ids have conflicting logged labels, e.g. {conflict[:3]}")
    if mismatch:
        fails.append(f"{len(mismatch)} CSV labels differ from the logged LLM label, e.g. {mismatch[:3]}")
    if dup:
        fails.append(f"{dup} ids were predicted in more than one log record (same label each time): duplicate requests")
    nu = len(nousage & set(ids))
    info.append(f"ids whose log record carries no usage object: {nu}")
    if nu and require_usage:
        fails.append(f"{nu} ids are backed by records without a usage object (not provably an API response)")
    info.append(f"sha256 {sha256(csv_path)}")
    info.append("per-class counts: " + ", ".join(f"{l} {c}" for l, c in sorted(Counter(labs).items(), key=lambda kv: -kv[1])))
    if dd_path and Path(dd_path).exists():
        _, drows = read_csv(dd_path)
        dd = {r[0]: r[1] for r in drows}
        diff = Counter((dd.get(i), l) for i, l in zip(ids, labs) if dd.get(i) != l)
        info.append(f"rows differing from dd_final.csv: {sum(diff.values())} of {len(ids)}; from -> to: "
                    + ", ".join(f"{a} -> {b} x{c}" for (a, b), c in diff.most_common(40)))
        if ids and not diff:
            fails.append("file is identical to dd_final.csv: not an independent LLM prediction")
    return fails, info


def selftest():
    import json, os, tempfile
    d = tempfile.mkdtemp()
    w = lambda name, txt: (open(os.path.join(d, name), "w", newline="", encoding="utf-8").write(txt), os.path.join(d, name))[1]
    sample = w("sample.csv", "id,label\nA,legit\nB,legit\nC,legit\n")
    dd = w("dd.csv", "id,label\nA,legit\nB,qr_swap\nC,legit\n")
    rec = lambda preds, usage=True: json.dumps({"response": {"usage": {"prompt_tokens": 9, "completion_tokens": 3}} if usage else {},
                                                "raw": json.dumps({"predictions": preds}), "parsed": preds}) + "\n"
    P = lambda i, l: {"txn_id": i, "label": l}
    good_log = w("log.jsonl", rec([P("A", "legit"), P("B", "sim_swap")]) + rec([P("C", "legit")]))
    good = w("good.csv", "id,label\nA,legit\nB,sim_swap\nC,legit\n")
    f, info = validate(good, [good_log], sample, dd, n=3, require_usage=True)
    assert not f and any("legit -> " not in s and "qr_swap -> sim_swap x1" in s for s in info), (f, info)
    assert any("PARTIAL" in x for x in validate(good, [w("l2.jsonl", rec([P("A", "legit"), P("B", "sim_swap")]))], sample, dd, n=3)[0])
    assert any("order" in x for x in validate(w("o.csv", "id,label\nB,sim_swap\nA,legit\nC,legit\n"), [good_log], sample, dd, n=3)[0])
    assert any("row count" in x for x in validate(w("s.csv", "id,label\nA,legit\nB,sim_swap\n"), [good_log], sample, dd, n=3)[0])
    assert any("differ from the logged" in x for x in validate(w("m.csv", "id,label\nA,legit\nB,qr_swap\nC,legit\n"), [good_log], sample, dd, n=3)[0])
    assert any("vocabulary" in x for x in validate(w("v.csv", "id,label\nA,legit\nB,fraud\nC,legit\n"), [good_log], sample, dd, n=3)[0])
    assert any("conflicting" in x for x in validate(good, [good_log, w("l3.jsonl", rec([P("B", "legit")]))], sample, dd, n=3)[0])
    assert any("more than one log record" in x for x in validate(good, [good_log, w("l4.jsonl", rec([P("C", "legit")]))], sample, dd, n=3)[0])
    assert any("usage" in x for x in validate(good, [w("l5.jsonl", rec([P("A", "legit"), P("B", "sim_swap"), P("C", "legit")], usage=False))], sample, dd, n=3, require_usage=True)[0])
    # an out-of-vocabulary logged label does not back a row
    assert any("PARTIAL" in x for x in validate(good, [w("l6.jsonl", rec([P("A", "legit"), P("B", "sim_swap"), P("C", "LEGIT")]))], sample, dd, n=3)[0])
    assert any("identical to dd_final" in x for x in validate(w("d.csv", "id,label\nA,legit\nB,qr_swap\nC,legit\n"),
                                                               [w("l7.jsonl", rec([P("A", "legit"), P("B", "qr_swap"), P("C", "legit")]))], sample, dd, n=3)[0])
    pf = w("pred.jsonl", json.dumps({"txn_id": "A", "label": "legit", "req": "x"}) + "\n" + json.dumps({"txn_id": "B", "label": "sim_swap"}) + "\n" + json.dumps({"txn_id": "C", "label": "legit"}) + "\n")
    assert not backing(pf, [good_log])[0]
    assert any("differ" in x for x in backing(w("p2.jsonl", json.dumps({"txn_id": "A", "label": "qr_swap"}) + "\n"), [good_log])[0])
    assert any("without a logged" in x for x in backing(w("p3.jsonl", json.dumps({"txn_id": "Z", "label": "legit"}) + "\n"), [good_log])[0])
    rej = w("l8.jsonl", json.dumps({"valid": False, "response": {"usage": {"input_tokens": 1}}, "parsed": [P("A", "qr_swap")]}) + "\n")
    assert not backing(pf, [good_log, rej])[0]                      # a rejected attempt neither backs nor conflicts
    dec = w("l9.jsonl", json.dumps({"valid": True, "usage": {"input_tokens": 5}, "response": {"answers": [
        {"name": "TX0000000a", "choice": "qr_swap", "probabilities": [{"value": "legit", "probability": 0.4}, {"value": "qr_swap", "probability": 0.6}], "confidence": 0.6}]}}) + "\n")
    assert not backing(w("p4.jsonl", json.dumps({"txn_id": "TX0000000a", "label": "qr_swap"}) + "\n"), [dec])[0]   # Decisions API shape
    assert backing(w("p5.jsonl", json.dumps({"txn_id": "TX0000000a", "label": "legit"}) + "\n"), [dec])[0]
    assert choice_vs_argmax([dec]) == (1, 0)
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(WORK / "scratch_llm" / "llm_direct.csv"))
    ap.add_argument("--log", action="append", default=[])
    ap.add_argument("--sample", default=str(WORK.parent / "upi-fraud-rings" / "sample_submission.csv"))
    ap.add_argument("--dd", default=str(WORK / "subs" / "dd_final.csv"))
    ap.add_argument("--require-usage", action="store_true")
    ap.add_argument("--pred", help="A's predictions jsonl: every line must equal the label / choice of a logged valid API response")
    ap.add_argument("--backing-only", action="store_true", help="only the --pred check (pilot files)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    paths = [p for g in a.log for p in glob.glob(g)]
    if a.pred:
        fails, info = backing(a.pred, paths)
        print("\n".join(info))
        print("backing: " + ("PASS" if not fails else "FAIL:\n- " + "\n- ".join(fails)))
        n, off = choice_vs_argmax(paths)
        if n:
            print(f"Decisions answers with probabilities: {n}; `choice` differs from the top-probability value: {off} (information only)")
        if fails or a.backing_only:
            raise SystemExit(1 if fails else 0)
    if not Path(a.csv).exists() or not paths:
        raise SystemExit(f"FAIL: csv ({a.csv}) or logs ({a.log}) not found")
    fails, info = validate(a.csv, paths, a.sample, a.dd, require_usage=a.require_usage)
    n, off = choice_vs_argmax(paths)
    if n:
        info.append(f"Decisions answers with probabilities: {n}; `choice` differs from the top-probability value: {off} (information only)")
    print("\n".join(info))
    if fails:
        print("\nFAIL:\n- " + "\n- ".join(fails))
        raise SystemExit(1)
    print(f"\nPASS: {N_TEST} rows, ids and order equal to the sample submission, every row backed by one logged LLM prediction")


if __name__ == "__main__":
    main()
