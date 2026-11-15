"""Label-isolation check of Agent A's request payloads. Run with cwd = work, PYTHONPATH=.

  python scratch_llm/check/check_isolation.py --mode pilot --log "scratch_llm/pilot_requests_<tag>.jsonl"
  python scratch_llm/check/check_isolation.py --mode full  --log "scratch_llm/full_requests.jsonl" [--stride 20] [--max 500]
  python scratch_llm/check/check_isolation.py --selftest

A's payload = a static prompt (instructions + labelled examples, the same for every request of a run) + a dynamic prompt
(target rows) + for the Decisions API one question per target. The log stores the dynamic prompt (pilot / probes) or only
its hash (full run). For every checked request the body is rebuilt with A's own code (requests.post is replaced by a
recorder, the network is never touched) on data whose evaluated-month labels are permuted.
Checks, each PASS / FAIL (exit 1 on any FAIL):
 (h) logged static_hash / prompt_hash equal sha256 of the rebuilt body text (and of the logged prompt when present).
 (a) dynamic prompts hold no label name at all and only lines of the declared format (fixed raw field counts, so no label
     or prediction column can ride along). In the static prompt label names occur only in the instruction section and on
     the `label:` line that closes an example.
 (q) Decisions API: one choice question per target, name = txn_id, the same 12 choices in the same order, instructions
     identical across rows once the txn_id is masked. Responses API: nothing in the body besides static, dynamic, schema.
 (b) every demonstration is a train row of an earlier month than the targets and shares no sender, receiver, device
     or fraud event (clean_labels.fraud_events) with a target. Entity sharing: FAIL for the pilot, count for the full run.
 (c) no model-output column name (feats_cache / feature-matrix schemas, any `x__y` token, `ts_*`, q, pred, proba, C_q,
     oof, dd_final ...) and no score-like key with a numeric value.
 (d) permutation test: dynamic prompts, static prompt and (pilot) demo list / natural sample come out byte-identical
     with evaluated-month labels permuted. Full run: test rows carry no label in the data; the rebuild proves the
     logged hashes belong to prompts made by that label-free builder.
 (k) API key pattern / Authorization header in A's directory: presence only, a match is never printed.
"""
import argparse
import glob
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import HERE, LLM, TX_RE, WORK, read_jsonl  # noqa: E402

LABELS = ["legit", "mule_account", "velocity_bot", "layering_chain", "sim_swap", "account_takeover",
          "phishing_collect", "qr_swap", "refund_abuse", "fake_cashback", "loan_app_extortion", "merchant_collusion"]
LABEL_RE = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(LABELS) + r")(?![A-Za-z0-9_])")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
HARD = {"q", "pred", "preds", "proba", "probs", "prob", "logit", "logits", "C_q", "C_pre", "C_p1", "oof", "y_pred", "yhat",
        "dd_final", "y_clean", "yc", "model_pred", "baseline_pred", "submission_label"}
SOFT_RE = re.compile(r"(?<![A-Za-z0-9_])(probability|probabilities|confidence|score|model_score|fraud_score|rule|rule_hit|"
                     r"rule_output|prediction|predicted)[\"']?\s*[:=]\s*[\"']?-?[0-9]")
KEY_RE = re.compile(rb"(?<![A-Za-z0-9_\-])sk-[A-Za-z0-9_\-]{20,}")
AUTH_RE = re.compile(rb"(?i)authorization|bearer\s+[A-Za-z0-9_\-\.]{16,}")
ORIG = {"txn_id", "timestamp", "sender_id", "receiver_id", "receiver_type", "device_id", "amount", "txn_type",
        "merchant_category", "device_city", "sim_age_hours", "device_age_days", "sender_account_age_days", "new_payee",
        "pin_retries", "login_to_txn_sec", "balance_before", "app", "app_version", "ip_risk_score", "qr_vpa_mismatch",
        "label", "ts", "month", "is_test", "y"}
NF = 22                 # full mode row: rel | dt | 20 raw fields
NF_LEAN, NF_CTX = 32, 9  # lean mode: txn_id | 20 raw fields | 11 account / merchant fields; context: rel | dt | 7 fields
REL_RE = re.compile(r"^(T|S?R?D?)$")
DT_RE = re.compile(r"^[+-]\d+$")
EX_HEAD = "LABELLED EXAMPLES"


def sha16(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def model_tokens(text, forbidden):
    """(hard hits, soft hits) for check (c)."""
    toks = set(TOKEN_RE.findall(text))
    # a feature name that is a plain English word (e.g. `hour`) counts only when used as a key: `hour=3`, `hour: 3`
    plain = {t for t in toks & forbidden if "_" not in t and not re.search(r"(?<![A-Za-z0-9_])" + t + r"\s*[:=]\s*-?[0-9]", text)}
    hard = ((toks & forbidden) - plain) | (toks & HARD) | {t for t in toks if "__" in t.strip("_") or t.startswith("ts_")}
    return sorted(hard), sorted({m.group(1) for m in SOFT_RE.finditer(text)})


def scan_blocks(text, head_word):
    """Full mode: `### <head_word> ...` blocks. Returns (problems, blocks); block = dict(head, label)."""
    prob, blocks = Counter(), []
    for chunk in text.split("### ")[1:]:
        lines = [l for l in chunk.split("\n") if l.strip()]
        b, nt = dict(head=lines[0].split(" ", 1)[-1].strip(), label=None), 0
        if not lines[0].startswith(head_word + " "):
            prob["unexpected_block_head"] += 1
        for k, l in enumerate(lines[1:], 1):
            if k in (1, 2) and l.startswith(("sender ", "receiver ")):
                prob["label_name_in_account_line"] += bool(LABEL_RE.search(l))
                continue
            if l.startswith("label: ") and k == len(lines) - 1 and head_word == "EXAMPLE":
                b["label"] = l[7:].strip()
                continue
            f = l.split("|")
            if len(f) != NF or not f[0] or not REL_RE.match(f[0]) or not DT_RE.match(f[1]):
                prob["line_not_a_raw_row"] += 1
            elif LABEL_RE.search(l):
                prob["label_name_in_row"] += 1
            nt += f[0] == "T"
        prob["block_without_single_T_row"] += nt != 1
        blocks.append(b)
    return +prob, blocks


def scan_lean(text, head_re, with_labels):
    """Lean mode: `<head>|20 fields|11 fields`, then indented context rows ` rel|dt|7 fields`, then (examples) `label: x`."""
    prob, blocks = Counter(), []
    for l in text.split("\n"):
        if not l.strip():
            continue
        f = l.split("|")
        if l.startswith(" "):
            if len(f) != NF_CTX or not f[0].strip() or not REL_RE.match(f[0].strip()) or f[0].strip() == "T" or not DT_RE.match(f[1]) or not blocks:
                prob["line_not_a_raw_row"] += 1
            elif LABEL_RE.search(l):
                prob["label_name_in_row"] += 1
        elif with_labels and l.startswith("label: ") and blocks and blocks[-1]["label"] is None:
            blocks[-1]["label"] = l[7:].strip()
        elif len(f) == NF_LEAN and re.fullmatch(head_re, f[0]):
            prob["label_name_in_row"] += bool(LABEL_RE.search(l))
            blocks.append(dict(head=f[0], label=None))
        else:
            prob["line_not_a_raw_row"] += 1
    return +prob, blocks


def scan_dynamic(dyn, target_ids, mode="full"):
    """Problems of one dynamic prompt (Counter). No label name may occur anywhere in it."""
    if mode == "lean":
        intro, _, body = dyn.partition("\n")
        prob, blocks = scan_lean(body, r"TX[0-9a-f]{8}", False)
        ok_intro = re.fullmatch(r"Label these \d+ TARGET transactions\.", intro)
    else:
        prob, blocks = scan_blocks(dyn, "TARGET")
        ok_intro = re.fullmatch(r"Label these \d+ TARGET transactions: [TX0-9a-f, ]+", dyn.split("### ")[0].strip())
    if LABEL_RE.search(dyn):
        prob["label_name_in_dynamic_prompt"] += len(LABEL_RE.findall(dyn))
    prob["unexpected_intro"] += not ok_intro
    prob["blocks_do_not_match_target_ids"] += [b["head"] for b in blocks] != list(target_ids)
    prob["txn_id_other_than_targets"] += bool(set(TX_RE.findall(dyn)) - set(target_ids))
    return +prob


def scan_static(static, demos, mode="full"):
    """Problems of the static prompt (Counter) and its example blocks."""
    if EX_HEAD not in static:
        return Counter(no_examples_section=1), []
    instr, exs = static.split(EX_HEAD, 1)
    prob = Counter(txn_id_in_static_prompt=len(TX_RE.findall(static)))
    p2, blocks = scan_lean(exs.partition("\n")[2], r"EX\d+", True) if mode == "lean" else scan_blocks(exs, "EXAMPLE")
    prob.update(p2)
    prob["example_labels_differ_from_demo_list"] += [b["label"] for b in blocks] != [d["label"] for d in demos]
    prob["label_name_outside_label_line"] += abs(len(LABEL_RE.findall(exs)) - len(blocks))
    return +prob, blocks


def scan_questions(body, ids):
    """Decisions API body: problems (Counter) of its `questions`; other APIs: no extra payload keys."""
    prob = Counter()
    qs = body.get("questions")
    if qs is None:
        extra = set(body) - {"model", "store", "prompt_cache_options", "input", "reasoning", "max_output_tokens", "text"}
        prob["unexpected_body_key"] += len(extra)
        return +prob
    prob["question_names_differ_from_targets"] += [q.get("name") for q in qs] != list(ids)
    prob["question_not_choice_type"] += sum(q.get("type") != "choice" for q in qs)
    prob["choices_not_the_12_labels_in_fixed_order"] += sum([c.get("value") for c in q.get("choices", [])] != LABELS or
                                                           any(set(c) != {"value"} for c in q.get("choices", [])) for q in qs)
    prob["question_instructions_differ_between_rows"] += len({str(q.get("instructions")).replace(str(q.get("name")), "<ID>") for q in qs}) > 1
    prob["label_name_in_question_instructions"] += sum(bool(LABEL_RE.search(str(q.get("instructions")))) for q in qs)
    prob["unexpected_question_key"] += sum(bool(set(q) - {"type", "name", "instructions", "choices"}) for q in qs)
    prob["unexpected_body_key"] += len(set(body) - {"model", "prompt_cache_options", "input", "questions"})
    return +prob


def key_scan(root=LLM, skip=HERE):
    """{relative file: (key pattern count, authorization count)} for files with any hit. Never returns the match."""
    out = {}
    for p in Path(root).rglob("*"):
        if not p.is_file() or skip in p.parents or p.suffix == ".pyc":
            continue
        k = a = 0
        with open(p, "rb") as f:
            for line in f:
                k += len(KEY_RE.findall(line))
                a += len(AUTH_RE.findall(line))
        if k or a:
            out[str(p.relative_to(root))] = (k, a)
    return out


def forbidden_columns():
    import polars as pl
    names = set()
    for f in list((WORK / "feats_cache").glob("*.parquet")) + list((WORK / "scratch_gap").glob("*_f0.parquet")):
        try:
            names.update(pl.read_parquet_schema(f))
        except Exception as e:   # another session may be writing a cache file
            print(f"note: schema of {f.name} unreadable ({type(e).__name__}), skipped")
    return {n for n in names if n not in ORIG and not n.startswith(("snd_", "rcv_", "mer_"))}


def permuted_data(A, mode, seed=7):
    """A.Data() built on data whose evaluated-month labels are permuted (pilot: September train rows)."""
    import numpy as np
    import polars as pl
    import common
    tx = common.load_all()
    if mode == "pilot":
        lab = np.array(tx["label"].to_list(), dtype=object)
        idx = np.flatnonzero(((tx["month"] == 9) & ~tx["is_test"]).to_numpy())
        before = lab[idx].copy()
        lab[idx] = lab[np.random.default_rng(seed).permutation(idx)]
        print(f"(d) permuted labels of {len(idx)} September rows; {int((lab[idx] != before).sum())} rows changed label")
        tx = tx.with_columns(pl.Series("label", lab.tolist(), dtype=pl.String)).with_columns(
            pl.col("label").replace_strict(common.LABEL2ID, default=None, return_dtype=pl.Int8).alias("y"))
    A.load_all = lambda: tx
    return A.Data()


def rederive_demos(A, D, pilot_ids):
    """Independent re-derivation of A's demonstration list from July/August labels and the pilot rows' entities only."""
    import polars as pl
    m = D.meta.sort("txn_id")
    ent = set()
    for t in pilot_ids:
        i = D.row[t]
        ent.update((D.snd[i], D.rcv[i], D.dev[i]))
    early = m.filter(pl.col("month").is_in([7, 8]) & ~pl.col("is_test"))
    demos = []
    for k, l in enumerate(A.LABELS):
        cand = early.filter(pl.col("label") == l)["txn_id"].to_list()
        random.Random(A.SEED + k).shuffle(cand)
        need = A.N_DEMO_LEGIT if l == "legit" else A.N_DEMO_FRAUD
        for t in cand:
            if need == 0:
                break
            if not (D.entities(D.row[t]) & ent):
                demos.append({"txn_id": t, "label": l})
                need -= 1
    random.Random(A.SEED).shuffle(demos)
    return demos


def lean_pick(demos):
    left = {l: (4 if l == "legit" else 2) for l in LABELS}
    pick = []
    for d in demos:
        if left[d["label"]]:
            left[d["label"]] -= 1
            pick.append(d)
    return pick


def rebuilt_static(A, D, demos, mode, api):
    """Static prompt rebuilt from A's builder pieces on the (permuted-label) data."""
    if mode == "lean":
        s = A.INSTR_LEAN + "\nLABELLED EXAMPLES (training months; labels are noisy)\n" + "\n".join(
            D.lean_block(D.row[d["txn_id"]], f"EX{n + 1}") + f"\nlabel: {d['label']}" for n, d in enumerate(lean_pick(demos)))
    else:
        s = A.INSTRUCTIONS + "\nLABELLED EXAMPLES (training months; labels are noisy)\n\n" + "\n\n".join(
            D.block(D.row[d["txn_id"]], f"### EXAMPLE {n + 1}") + f"\nlabel: {d['label']}" for n, d in enumerate(demos))
    if api == "decisions":
        s = s[:s.index("OUTPUT\n")] + A.DEC_OUTPUT + s[s.index("\nLABELLED EXAMPLES"):]
    return s


def selftest():
    raw = ["09-01 10:00:00", "AC1", "AC2", "P2P", "DV1", "5.0", "PAY"] + [""] * 13
    row = lambda rel, dt: "|".join([rel, dt] + raw)
    tgt = lambda t: f"### TARGET {t}\nsender AC1: home_city=X|bank_code=B|kyc_level=full|opened_on=2020-01-01|os=android\nreceiver AC2: home_city=Y|bank_code=B|kyc_level=min|opened_on=2021-01-01|os=ios\n{row('S', '-30')}\n{row('T', '+0')}\n{row('RD', '+12')}"
    ids = ["TX00000001", "TX00000002"]
    ok = "Label these 2 TARGET transactions: TX00000001, TX00000002\n\n" + tgt(ids[0]) + "\n\n" + tgt(ids[1])
    assert not scan_dynamic(ok, ids), scan_dynamic(ok, ids)
    assert scan_dynamic(ok + "|qr_swap", ids) == Counter(label_name_in_dynamic_prompt=1, line_not_a_raw_row=1)   # extra column
    assert scan_dynamic(ok.replace("PAY", "legit", 1), ids)["label_name_in_row"] == 1
    assert scan_dynamic(ok + "\nhint: probably fraud", ids)["line_not_a_raw_row"] == 1
    assert scan_dynamic(ok, ids[::-1])["blocks_do_not_match_target_ids"] == 1
    assert scan_dynamic(ok.replace("-30", "-30|TX00000009", 1), ids)["txn_id_other_than_targets"] == 1
    assert scan_dynamic(ok.replace(row("T", "+0"), row("S", "+0"), 1), ids)["block_without_single_T_row"] == 1
    ex = lambda n, l: tgt("x").replace("### TARGET x", f"### EXAMPLE {n}") + f"\nlabel: {l}"
    static = "Labels: legit, qr_swap.\n" + EX_HEAD + " (training months)\n\n" + ex(1, "qr_swap") + "\n\n" + ex(2, "legit")
    demos = [{"txn_id": "a", "label": "qr_swap"}, {"txn_id": "b", "label": "legit"}]
    p, b = scan_static(static, demos)
    assert not p and len(b) == 2, p
    assert scan_static(static, demos[::-1])[0]["example_labels_differ_from_demo_list"] == 1
    assert scan_static(static.replace("PAY", "legit", 1), demos)[0]["label_name_in_row"] == 1
    # lean mode
    lt = lambda h: "|".join([h] + raw + ["X"] * 11) + "\n S|-30|AC1|AC9|DV1|5.0|PAY|Pune|1\n RD|+7|AC3|AC2|DV1|9.0|PAY|Pune|0"
    lok = "Label these 2 TARGET transactions.\n" + lt(ids[0]) + "\n" + lt(ids[1])
    assert not scan_dynamic(lok, ids, "lean"), scan_dynamic(lok, ids, "lean")
    assert scan_dynamic(lok + "|0.93", ids, "lean")["line_not_a_raw_row"] == 1
    assert scan_dynamic(lok.replace("|X|", "|X|X|", 1), ids, "lean")["line_not_a_raw_row"] >= 1          # 33 fields on a target line
    assert scan_dynamic(lok.replace("Pune", "mule_account", 1), ids, "lean")["label_name_in_row"] == 1
    assert scan_dynamic(lok + "\nlabel: legit", ids, "lean")["line_not_a_raw_row"] == 1
    ls = "Labels: legit, qr_swap.\n" + EX_HEAD + " (x)\n" + lt("EX1") + "\nlabel: qr_swap\n" + lt("EX2") + "\nlabel: legit"
    p, b = scan_static(ls, demos, "lean")
    assert not p and [x["label"] for x in b] == ["qr_swap", "legit"], p
    assert scan_static(ls.replace("EX2", ids[0]), demos, "lean")[0]["txn_id_in_static_prompt"] == 1
    # questions
    q = lambda t, ins="Which label fits TARGET transaction {}?": {"type": "choice", "name": t, "instructions": ins.format(t), "choices": [{"value": l} for l in LABELS]}
    body = {"model": "m", "prompt_cache_options": {}, "input": [], "questions": [q(ids[0]), q(ids[1])]}
    assert not scan_questions(body, ids)
    assert scan_questions({**body, "questions": [q(ids[0]), q(ids[1], "Likely fraud: which label fits {}?")]}, ids)["question_instructions_differ_between_rows"] == 1
    bad = q(ids[1]); bad["choices"] = bad["choices"][::-1]
    assert scan_questions({**body, "questions": [q(ids[0]), bad]}, ids)["choices_not_the_12_labels_in_fixed_order"] == 1
    assert scan_questions({**body, "metadata": {"hint": 1}}, ids)["unexpected_body_key"] == 1
    assert scan_questions({**body, "questions": [q(ids[0]), {**q(ids[1]), "prior": "legit"}]}, ids)["unexpected_question_key"] == 1
    assert [d["label"] for d in lean_pick([{"label": "legit"}] * 6 + [{"label": "qr_swap"}] * 3)] == ["legit"] * 4 + ["qr_swap"] * 2
    assert model_tokens("x flow__out_deg=3 ts_bait q=0.9 ip_risk_score=0.4", {"flow__out_deg"}) == (["flow__out_deg", "q", "ts_bait"], [])
    assert model_tokens("in the last hour", {"hour"}) == ([], []) and model_tokens("hour=3", {"hour"}) == (["hour"], [])
    assert model_tokens("ip_risk_score: a score from a vendor; score: 0.9", set()) == ([], ["score"])
    assert KEY_RE.search(b"key=sk-" + b"a" * 24) and not KEY_RE.search(b"sk-short") and AUTH_RE.search(b'"Authorization": "x"')
    assert not KEY_RE.search(b"abcDsk-" + b"a" * 24)      # inside a base64 blob, not a key
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot", help="pilot: targets = pilot ids (September); full: test rows")
    ap.add_argument("--log", action="append", default=[], help="request log (jsonl); repeatable, globs allowed")
    ap.add_argument("--ids", default=str(LLM / "pilot_sample_ids.csv"))
    ap.add_argument("--stride", type=int, default=1, help="check every n-th request (stated in the output)")
    ap.add_argument("--max", type=int, default=0, help="stop after this many checked requests")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import polars as pl
    res, fail = {"mode": a.mode}, []

    def verdict(name, bad, evidence=""):
        print(f"{name}: {'FAIL' if bad else 'PASS'} {evidence}", flush=True)
        res[name] = dict(ok=not bad, evidence=str(evidence))
        if bad:
            fail.append(name)

    ks = key_scan()
    verdict("(k) no API key pattern / Authorization header in logs", any(k for k, _ in ks.values()) or any(au for f, (_, au) in ks.items() if ".py" not in f),
            f"files with hits (key pattern count, authorization count): {ks or 'none'}; hits in run_llm*.py are the header name in code")
    paths = sorted(p for g in a.log for p in glob.glob(g))
    if not paths:
        raise SystemExit("FAIL: no request log given / found")
    demos = json.loads((LLM / "demos.json").read_text())
    if a.mode == "pilot":
        targets = pl.read_csv(a.ids, infer_schema_length=0)["txn_id"].to_list()
    else:
        targets = pl.read_csv(WORK.parent / "upi-fraud-rings" / "sample_submission.csv")["id"].to_list()
    tset = set(targets)

    sys.path.insert(0, str(LLM))
    sys.dont_write_bytecode = True      # importing A's module must not write __pycache__ into A's directory
    import run_llm as A
    A.api_key = lambda *x, **k: (_ for _ in ()).throw(RuntimeError("checker must never read the key"))
    sent = []
    A.requests.post = lambda url, headers=None, json=None, timeout=None: sent.append((url, json))   # recorder: no network
    D = permuted_data(A, a.mode)
    forb = forbidden_columns()
    statics, variants = {}, Counter()

    def static_for(mode, api):
        if (mode, api) not in statics:
            statics[(mode, api)] = rebuilt_static(A, D, demos, mode, api)
        return statics[(mode, api)]

    c, ex, seen = Counter(), {}, set()
    n_req = 0
    for p in paths:
        for _, r in read_jsonl(p):
            if not isinstance(r, dict) or "target_ids" not in r:
                continue
            n_req += 1
            c["requests"] += 1
            seen.update(r["target_ids"])
            c["request_with_non_target_ids"] += bool(set(r["target_ids"]) - tset)
            if n_req % a.stride or (a.max and c["requests_checked"] >= a.max):
                continue
            c["requests_checked"] += 1
            mode, api, ids = r.get("mode", "full"), r.get("api", "responses"), r["target_ids"]
            variants[(mode, api, r.get("model"), r.get("static_hash"))] += 1
            static = static_for(mode, api)
            sent.clear()
            if api == "decisions":
                A.post_decisions("KEY", static, A.dynamic_prompt(D, ids, mode), ids, 1)
            else:
                A.post("KEY", static, A.dynamic_prompt(D, ids, mode), 100, "low", r.get("model") or A.MODEL, 1)
            body = sent[0][1]
            b_static, re_dyn = body["input"][0]["content"][0]["text"], body["input"][1]["content"]
            dyn = r.get("prompt", re_dyn)
            c["with_logged_prompt"] += "prompt" in r
            bad = Counter()
            bad["h_static_hash_mismatch"] += r.get("static_hash") != sha16(b_static)
            bad["h_prompt_hash_mismatch"] += r.get("prompt_hash") != sha16(b_static + dyn)
            bad["d_prompt_differs_under_permuted_labels"] += dyn != re_dyn
            for k, v in scan_dynamic(dyn, ids, mode).items():
                bad["a_" + k] += v
            for k, v in scan_questions(body, ids).items():
                bad["q_" + k] += v
            hard, soft = model_tokens(dyn + json.dumps(body.get("questions", "")), forb)
            bad["c_model_tokens"] += bool(hard or soft)
            for k, v in (+bad).items():
                c[k] += v
                ex.setdefault(k, [])
                if len(ex[k]) < 5:
                    ex[k].append(r.get("req") if k != "c_model_tokens" else (r.get("req"), hard[:5], soft[:5]))
    print(f"requests in log {c['requests']}, checked {c['requests_checked']} (stride {a.stride}, max {a.max or 'none'}), with logged prompt "
          f"{c['with_logged_prompt']}; targets in logged requests {len(seen & tset)} of {len(tset)}; variants (mode, api, model, static_hash): {dict(variants)}")
    grp = lambda pre: {k: v for k, v in c.items() if k.startswith(pre)}
    verdict("(h) logged hashes equal the rebuilt body text", grp("h_"), (grp("h_"), ex) if grp("h_") else f"{c['requests_checked']} static and prompt hashes match")
    verdict("(a) dynamic prompts: no label name, only raw rows of the declared field count", grp("a_") or c["request_with_non_target_ids"],
            (grp("a_"), ex) if grp("a_") else f"0 label names in {c['requests_checked']} dynamic prompts; requests with non-target ids {c['request_with_non_target_ids']}")
    verdict("(q) questions / body carry no per-row hint", grp("q_"), (grp("q_"), ex) if grp("q_") else
            f"{c['requests_checked']} request bodies rebuilt with A's post functions: keys, question names, 12 fixed choices, masked instructions identical")

    # ---- static prompts used by the logged requests
    from clean_labels import fraud_events
    T = common_true()
    _, F = fraud_events(T)
    meta = T.join(F.select("txn_id", "ev"), on="txn_id", how="left")
    tm = meta.filter(pl.col("txn_id").is_in(targets))
    t_acc = set(tm["sender_id"].to_list()) | set(tm["receiver_id"].to_list())
    t_dev = set(tm["device_id"].to_list())
    t_ev = set(tm["ev"].drop_nulls().to_list())
    for (mode, api), static in statics.items():
        tag = f"[{mode}/{api} static {sha16(static)}, {len(static)} chars]"
        (HERE / f"prompt_static_checked_{sha16(static)}.txt").write_text(static, encoding="utf-8")     # snapshot of what was checked
        dl = lean_pick(demos) if mode == "lean" else demos
        ps, blocks = scan_static(static, dl, mode)
        verdict(f"(a) static prompt {tag}: label names only in instructions and on example `label:` lines", bool(ps), dict(ps) or f"{len(blocks)} examples")
        hard, soft = model_tokens(static, forb)
        verdict(f"(c) no model-output columns {tag} + dynamic prompts", bool(hard or soft or c["c_model_tokens"]),
                dict(static_hard=hard[:10], static_soft=soft, dynamic=ex.get("c_model_tokens")) if (hard or soft or c["c_model_tokens"])
                else f"{len(forb)} feature-matrix column names + patterns searched, 0 hits")
        dm = meta.filter(pl.col("txn_id").is_in([d["txn_id"] for d in dl]))
        lab_ok = dict(zip(dm["txn_id"].to_list(), dm["label"].to_list()))
        b = Counter()
        b["demo_label_differs_from_train_label"] = sum(lab_ok.get(d["txn_id"]) != d["label"] for d in dl)
        b["demo_is_test_row"] = int(dm["is_test"].sum())
        b["demo_not_earlier_month"] = int((dm["mk"] >= tm["mk"].min()).sum())
        b["demo_shares_entity_with_target"] = dm.filter(pl.col("sender_id").is_in(list(t_acc)) | pl.col("receiver_id").is_in(list(t_acc))
                                                        | pl.col("device_id").is_in(list(t_dev))).height
        b["demo_shares_fraud_event_with_target"] = int(dm["ev"].is_in(list(t_ev)).fill_null(False).sum()) if t_ev else 0
        hard_b = [k for k in b if k != "demo_shares_entity_with_target" or a.mode == "pilot"]
        verdict(f"(b) demonstrations {tag}: earlier month, no shared sender / receiver / device / fraud event", any(b[k] for k in hard_b),
                f"{dm.height} demos, months {sorted(dm['month'].unique().to_list())}, target months {sorted(tm['month'].unique().to_list())}, {dict(b)}"
                + ("" if a.mode == "pilot" else " (entity sharing with test rows is reported, not failed: train labels are known for the test month)"))
        fpath = LLM / f"prompt_static_{mode}.txt"
        on_disk = fpath.read_text(encoding="utf-8") if fpath.exists() else None
        if on_disk is not None and api == "decisions":
            on_disk = on_disk[:on_disk.index("OUTPUT\n")] + A.DEC_OUTPUT + on_disk[on_disk.index("\nLABELLED EXAMPLES"):]
        res[f"static {tag} equals A's file"] = on_disk == static
        print(f"    info: rebuilt static {tag} equals A's {fpath.name}" + (" with the Decisions OUTPUT swap" if api == "decisions" else "") + f": {on_disk == static}")

    d_bad = dict(dynamic_prompts_differ=c["d_prompt_differs_under_permuted_labels"], static_hash_differs=c["h_static_hash_mismatch"])
    if a.mode == "pilot":
        d_bad["demo_list_differs"] = int(rederive_demos(A, D, targets) != demos)
        m = D.meta.sort("txn_id")
        nat = set(m.filter((pl.col("month") == 9) & ~pl.col("is_test")).sample(300, seed=A.SEED, shuffle=True)["txn_id"].to_list())
        sub = pl.read_csv(a.ids, infer_schema_length=0)
        d_bad["natural_sample_not_the_seeded_label_free_draw"] = int(nat != set(sub.filter(pl.col("subset") == "natural")["txn_id"].to_list()))
    verdict("(d) permutation test: prompts identical with evaluated-month labels permuted", any(d_bad.values()),
            f"{d_bad}; {c['requests_checked']} request bodies and the static prompt rebuilt with A's builder {A.VERSION}"
            + (", demo list and natural-sample draw re-derived, on September labels permuted" if a.mode == "pilot"
               else " (test rows carry no label in the data, nothing to permute; this is a rebuild-and-hash check)"))
    res.update(counts=dict(c), examples=ex, fail=fail, logs=paths, stride=a.stride, variants={str(k): v for k, v in variants.items()})
    (HERE / f"isolation_{a.mode}_{Path(paths[0]).stem}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    if fail:
        raise SystemExit("FAIL: " + "; ".join(fail))
    print("ALL ISOLATION CHECKS PASS")


_TRUE = None


def common_true():
    """Unpermuted rows straight from the parquet files (checker-side only): ids, entities, month, true label."""
    global _TRUE
    if _TRUE is None:
        import polars as pl
        from common import LABEL2ID
        cols = ["txn_id", "timestamp", "sender_id", "receiver_id", "receiver_type", "device_id"]
        tr = pl.read_parquet(WORK / "data" / "train.parquet", columns=cols + ["label"]).with_columns(pl.lit(False).alias("is_test"))
        te = pl.read_parquet(WORK / "data" / "test.parquet", columns=cols).with_columns(pl.lit(None, pl.String).alias("label"), pl.lit(True).alias("is_test"))
        _TRUE = pl.concat([tr, te.select(tr.columns)]).with_columns(
            pl.col("timestamp").dt.epoch("s").alias("ts"), pl.col("timestamp").dt.month().alias("month"),
            (pl.col("timestamp").dt.year() * 12 + pl.col("timestamp").dt.month()).alias("mk"),
            pl.col("label").replace_strict(LABEL2ID, default=None, return_dtype=pl.Int8).alias("y"))
    return _TRUE


if __name__ == "__main__":
    main()
