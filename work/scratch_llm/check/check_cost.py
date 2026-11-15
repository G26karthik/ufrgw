"""Independent token / cost recomputation from logged `usage` objects, and projection to the full test set.

  python scratch_llm/check/check_cost.py --log scratch_llm/<log>.jsonl --price-in 0.25 --price-cached 0.025 --price-out 2.0 \
         [--a-cost 1.234] [--rows 630] [--deadline 10:00]
  python scratch_llm/check/check_cost.py --selftest

Prices are USD per 1M tokens. Every usage object in the log is billed, including failed and retried calls.
Rows per second = rows with a valid prediction / (last log timestamp - first log timestamp), not summed latencies.
"""
import argparse
import datetime as dt
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import N_TEST, find_predictions, find_usage, read_jsonl, usage_tokens  # noqa: E402

LABELS = {"legit", "mule_account", "velocity_bot", "layering_chain", "sim_swap", "account_takeover",
          "phishing_collect", "qr_swap", "refund_abuse", "fake_cashback", "loan_app_extortion", "merchant_collusion"}
TIME_KEYS = ("ts", "time", "timestamp", "t", "t0", "t1", "t_start", "t_end", "start", "end", "started", "finished",
             "started_at", "finished_at", "created", "created_at", "logged_at")


def to_epoch(v):
    if isinstance(v, (int, float)) and 1.6e9 < v < 2.2e9:
        return float(v)
    if isinstance(v, (int, float)) and 1.6e12 < v < 2.2e12:
        return v / 1000.0
    if isinstance(v, str):
        try:
            return dt.datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def rec_times(rec):
    """Epoch seconds from top-level time keys and response.created only (prompts hold transaction timestamps)."""
    out = [to_epoch(rec.get(k)) for k in TIME_KEYS if isinstance(rec, dict) and k in rec]
    r = rec.get("response") if isinstance(rec, dict) else None
    if isinstance(r, dict) and "created" in r:
        out.append(to_epoch(r["created"]))
    return [t for t in out if t]


def tally(records):
    """Agent A's request log: one record per API attempt (req = name-batch-attempt, time, latency, target_ids, valid, cost)."""
    s = dict(records=0, calls=0, tin=0, tcached=0, twrite=0, tout=0, errors=0, retried=0, no_usage=0, invalid_label=0,
             a_cost=0.0, a_cost_no_usage=0.0, tmin=None, tmax=None)
    ok_ids, sent_ids = set(), set()
    for rec in records:
        s["records"] += 1
        us = find_usage(rec)
        s["calls"] += len(us)
        s["no_usage"] += not us
        for u in us:
            a, c, w, o = usage_tokens(u)
            s["tin"] += a; s["tcached"] += c; s["twrite"] += w; s["tout"] += o
        if not isinstance(rec, dict):
            continue
        bad = bool(rec.get("error")) or rec.get("ok") is False or rec.get("valid") is False
        s["errors"] += bad
        # req = name-batch-attempt (v1) or kind-tag-p<pass>-batch-attempt.n429 (v2): any non-zero pass / attempt / 429 count is a retry
        m = re.search(r"-p(\d+)-\d+-(\d+)\.(\d+)$", str(rec.get("req", ""))) or re.search(r"-(\d+)$", str(rec.get("req", "")))
        s["retried"] += bool(m and any(int(g) > 0 for g in m.groups()))
        if isinstance(rec.get("cost"), (int, float)):
            s["a_cost"] += rec["cost"]
            s["a_cost_no_usage"] += rec["cost"] if not us else 0.0
        sent_ids.update(rec.get("target_ids") or [])
        if not bad:
            preds = set(find_predictions({k: v for k, v in rec.items() if k not in ("prompt", "request", "body")}))
            sent_ids.update(i for i, _ in preds)
            ok_ids.update(i for i, l in preds if l in LABELS)
            s["invalid_label"] += sum(l not in LABELS for _, l in preds)
        ts = rec_times(rec)
        if ts and isinstance(rec.get("latency"), (int, float)):
            ts.append(max(ts) + rec["latency"])
        for t in ts:
            s["tmin"] = t if s["tmin"] is None else min(s["tmin"], t)
            s["tmax"] = t if s["tmax"] is None else max(s["tmax"], t)
    s["rows_valid"], s["rows_seen"] = len(ok_ids), len(sent_ids)
    return s


def cost(s, pin, pcached, pout, pwrite=None):
    return (s["tin"] * pin + s["tcached"] * pcached + s["twrite"] * (pin if pwrite is None else pwrite) + s["tout"] * pout) / 1e6


def project(s, usd, rows, n=N_TEST, now=None, deadline=None):
    out = dict(tokens_per_row=(s["tin"] + s["tcached"] + s["twrite"] + s["tout"]) / rows, usd_per_row=usd / rows, usd_full=usd / rows * n)
    if s["tmin"] is not None and s["tmax"] > s["tmin"]:
        out["rows_per_sec"] = rows / (s["tmax"] - s["tmin"])
        out["wall_sec_full"] = n / out["rows_per_sec"]
        if now is not None and deadline is not None:
            out["finish"] = now + dt.timedelta(seconds=out["wall_sec_full"])
            left = (deadline - now).total_seconds()
            out["speedup_needed"] = out["wall_sec_full"] / left if left > 0 else float("inf")
    return out


def selftest():
    u1 = {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_tokens_details": {"cached_tokens": 400},
          "completion_tokens_details": {"reasoning_tokens": 60}}
    u2 = {"input_tokens": 500, "output_tokens": 50, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 200}}
    assert usage_tokens(u1) == (600, 400, 0, 100) and usage_tokens(u2) == (300, 0, 200, 50)
    out = lambda preds: {"output": [{"content": [{"text": json.dumps({"predictions": preds})}]}]}
    A, B, C = "TX0000000a", "TX0000000b", "TX0000000c"
    recs = [{"req": "p-0-0", "time": 1_800_000_000, "latency": 4.0, "target_ids": [A, B], "valid": False, "error": "label not in vocabulary", "cost": 0.5,
             "usage": u1, "response": {**out([{"txn_id": A, "label": "legit"}, {"txn_id": B, "label": "bogus"}]), "usage": u1}},
            {"req": "p-0-1", "time": 1_800_000_010, "latency": 5.0, "target_ids": [A, B], "valid": True, "error": None, "cost": 0.25,
             "usage": u2, "response": {**out([{"txn_id": A, "label": "legit"}, {"txn_id": B, "label": "qr_swap"}]), "usage": u2},
             "prompt": "### TARGET TX0000000a timestamp 2026-09-01"},
            {"req": "p-1-0", "time": 1_800_000_012, "latency": 8.0, "target_ids": [C], "valid": False, "error": "Timeout", "cost": 0.125}]
    s = tally(recs)
    assert (s["tin"], s["tcached"], s["twrite"], s["tout"], s["calls"]) == (900, 400, 200, 150, 2), s     # rec.usage == response.usage counted once
    assert (s["errors"], s["retried"], s["no_usage"], s["rows_valid"], s["rows_seen"]) == (2, 1, 1, 2, 3), s
    v2 = lambda req: tally([{"req": req, "cost": 0.0}])["retried"]
    assert (v2("full-decisions_lean-p0-5-0.0"), v2("full-decisions_lean-p0-5-1.0"), v2("full-decisions_lean-p0-5-0.2"), v2("full-x-p1-0-0.0")) == (0, 1, 1, 1)
    assert s["a_cost"] == 0.875 and s["a_cost_no_usage"] == 0.125 and s["tmax"] - s["tmin"] == 20.0
    usd = cost(s, 1.0, 0.1, 10.0, 1.25)
    assert abs(usd - (900 * 1 + 400 * 0.1 + 200 * 1.25 + 150 * 10) / 1e6) < 1e-12
    p = project(s, usd, 2, n=200)
    assert abs(p["usd_full"] - usd * 100) < 1e-12 and p["tokens_per_row"] == 825 and p["rows_per_sec"] == 0.1
    assert to_epoch("2027-01-15T08:00:20+00:00") == dt.datetime(2027, 1, 15, 8, 0, 20, tzinfo=dt.timezone.utc).timestamp()
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append", default=[])
    ap.add_argument("--price-in", type=float); ap.add_argument("--price-cached", type=float); ap.add_argument("--price-out", type=float)
    ap.add_argument("--price-write", type=float, help="cache-write input price (default: the uncached input price)")
    ap.add_argument("--a-cost", type=float, help="Agent A's own cost figure (USD) for the same log")
    ap.add_argument("--rows", type=int, help="rows processed (default: distinct ids with a valid logged label)")
    ap.add_argument("--expected", type=int, help="rows that were sent (for the missing rate)")
    ap.add_argument("--elapsed", type=float, help="wall seconds, if the log has no timestamps")
    ap.add_argument("--deadline", default="10:00", help="local time HH:MM today")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    paths = [p for g in a.log for p in glob.glob(g)]
    if not paths or None in (a.price_in, a.price_cached, a.price_out):
        raise SystemExit("FAIL: need --log and the three prices (USD per 1M tokens)")
    s = tally(rec for p in paths for _, rec in read_jsonl(p))
    if a.elapsed:
        s["tmin"], s["tmax"] = 0.0, a.elapsed
    rows = a.rows or s["rows_valid"]
    if not rows or not s["calls"]:
        raise SystemExit(f"FAIL: no rows ({rows}) or no usage objects ({s['calls']}) found in {paths}: {s}")
    usd = cost(s, a.price_in, a.price_cached, a.price_out, a.price_write)
    if a.a_cost is None and s["a_cost"]:
        a.a_cost = s["a_cost"] - s["a_cost_no_usage"]     # sum of A's per-record cost fields, records with a usage object
    now = dt.datetime.now()
    h, m = map(int, a.deadline.split(":"))
    p = project(s, usd, rows, now=now, deadline=now.replace(hour=h, minute=m, second=0, microsecond=0))
    print(f"log records {s['records']}, usage objects (billed calls) {s['calls']}, records without usage {s['no_usage']}")
    print(f"tokens: uncached input {s['tin']:,}, cached input {s['tcached']:,}, cache write {s['twrite']:,}, output {s['tout']:,}")
    print(f"recomputed cost: ${usd:.4f} (prices per 1M: in {a.price_in}, cached {a.price_cached}, write {a.price_write}, out {a.price_out})")
    if s["a_cost_no_usage"]:
        print(f"A also booked ${s['a_cost_no_usage']:.4f} as worst case for attempts with unknown outcome (no usage object): not recomputable")
    if a.a_cost is not None:
        d = usd - a.a_cost
        print(f"Agent A's figure: ${a.a_cost:.4f}; difference ${d:+.4f} ({d / a.a_cost:+.2%}) -> {'PASS' if abs(d) <= 0.01 * max(a.a_cost, 1e-9) + 1e-4 else 'FAIL (more than 1% apart)'}")
    exp = a.expected or s["rows_seen"]
    print(f"rows with a valid logged label {s['rows_valid']}; rows sent {exp}; missing {exp - s['rows_valid']} ({(exp - s['rows_valid']) / max(exp, 1):.2%}); "
          f"invalid-label predictions {s['invalid_label']}; records with error {s['errors']} ({s['errors'] / s['records']:.2%}); "
          f"retried records {s['retried']} ({s['retried'] / s['records']:.2%})")
    print(f"per row: {p['tokens_per_row']:.0f} tokens, ${p['usd_per_row']:.6f}. Projection for {N_TEST:,} rows: ${p['usd_full']:.2f}")
    if "rows_per_sec" in p:
        print(f"measured {p['rows_per_sec']:.2f} rows/s over {s['tmax'] - s['tmin']:.0f} s -> {p['wall_sec_full'] / 60:.1f} min for the full set; "
              f"started now ({now:%H:%M}) it ends {p['finish']:%H:%M on %d %b}; deadline {a.deadline} needs {p['speedup_needed']:.1f}x the measured rate")
    else:
        print("no usable timestamps in the log: wall-clock projection NOT done (pass --elapsed seconds)")


if __name__ == "__main__":
    main()
