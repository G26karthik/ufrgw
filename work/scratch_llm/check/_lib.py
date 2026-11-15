"""Shared parsing for Agent B's checkers. All knowledge of Agent A's file formats lives here."""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLM = HERE.parent                    # work/scratch_llm (Agent A's directory, read-only for us)
WORK = LLM.parent
N_TEST = 260582
TX_RE = re.compile(r"TX[0-9a-f]{8}")
# keys whose content is model output, not request payload
RESP_KEYS = {"response", "output", "choices", "usage", "result", "completion", "predictions", "error", "parsed",
             "raw", "content_out", "output_text", "raw_output"}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path):
    """Yields (line_no, record). A complete line that is not JSON raises: a corrupt log must not pass silently.
    Only an unterminated last line is skipped (the writer is still appending to a live log)."""
    with open(path, encoding="utf-8") as f:
        for n, raw in enumerate(f, 1):
            line = raw.strip()
            if line:
                try:
                    yield n, json.loads(line)
                except json.JSONDecodeError as e:
                    if not raw.endswith("\n"):
                        return
                    raise SystemExit(f"FAIL: {path} line {n} is not valid JSON ({e})")


def maybe_json(s):
    if isinstance(s, str) and s.lstrip()[:1] in "{[":
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    return None


def walk(o, path=()):
    """Yields (key_path, value) for every node; JSON held inside a string is parsed and walked too."""
    yield path, o
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, path + (str(k),))
    elif isinstance(o, list):
        for v in o:
            yield from walk(v, path)
    elif isinstance(o, str):
        j = maybe_json(o)
        if j is not None:
            yield from walk(j, path)


def find_predictions(rec):
    """Every (txn id, label) pair anywhere in a record, also inside raw JSON strings.
    Shapes: {txn_id|id, label} (Responses API / A's prediction lines) and {name, choice} (Decisions API)."""
    out = []
    for _, v in walk(rec):
        if not isinstance(v, dict):
            continue
        i = next((v[k] for k in ("txn_id", "id", "name") if isinstance(v.get(k), str) and (k != "name" or TX_RE.fullmatch(v[k]))), None)
        l = next((v[k] for k in ("label", "choice") if isinstance(v.get(k), str)), None)   # strings only: a JSON schema has these keys too
        if i is not None and l is not None:
            out.append((i, l))
    return out


def find_usage(rec):
    """Every usage object in a record: a dict under key 'usage' that carries token counts."""
    out = []
    for p, v in walk(rec):   # A logs the same object as rec.usage and rec.response.usage: count equal objects once
        if p and p[-1] == "usage" and isinstance(v, dict) and any("tokens" in k for k in v) and v not in out:
            out.append(v)
    return out


def usage_tokens(u):
    """(uncached input, cached read, cache write, output) for both OpenAI usage shapes. Reasoning tokens are inside output."""
    inp = u.get("prompt_tokens", u.get("input_tokens", 0)) or 0
    out = u.get("completion_tokens", u.get("output_tokens", 0)) or 0
    det = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
    cached = (det.get("cached_tokens", 0) or 0) if isinstance(det, dict) else 0
    write = (det.get("cache_write_tokens", det.get("cache_creation_tokens", 0)) or 0) if isinstance(det, dict) else 0
    return inp - cached - write, cached, write, out


def request_strings(rec):
    """String leaves of the request payload: everything not under a response-like key."""
    out = []

    def rec_(o, in_resp):
        if isinstance(o, dict):
            for k, v in o.items():
                rec_(v, in_resp or str(k) in RESP_KEYS)
        elif isinstance(o, list):
            for v in o:
                rec_(v, in_resp)
        elif isinstance(o, str) and not in_resp:
            out.append(o)
    rec_(rec, False)
    return out


def wilson(k, n, z=1.96):
    """95% Wilson interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))
