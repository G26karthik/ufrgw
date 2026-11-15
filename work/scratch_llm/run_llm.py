"""Direct LLM classification of UPI transactions with gpt-6-astra (OpenAI Responses API).

Run from work/:  PYTHONPATH=. python scratch_llm/run_llm.py <cmd> ...
  probe0                      one tiny request, no data: checks endpoint / parameters / usage / rate-limit headers
  build-pilot                 pilot sample ids (no labels), demonstrations, static prompt  -> scratch_llm/
  probe  --n 3 --seed 1       one request with n non-pilot September targets (real prompt)
  pilot  --cap USD            the 525 pilot rows -> pilot_predictions.jsonl
  full   --cap USD            all 260,582 test rows -> full_predictions.jsonl  (main agent decides)
No prompt contains model predictions, rule outputs, engineered features or labels of neighbourhood rows.
"""
import argparse, hashlib, json, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import requests

from common import LABELS, load_all

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = "gpt-6-astra"
URL = "https://api.openai.com/v1/responses"
# $ per 1M tokens, developers.openai.com/api/docs/pricing + models/gpt-6-astra (read 15 Nov 2026), <=272K input
# (input, cached input, 30-min cache write, output)
PRICES = {"gpt-6-astra": (10.0, 1.0, 12.5, 50.0), "gpt-6-sol": (2.0, 0.2, 2.5, 10.0), "gpt-6-luna": (0.10, 0.01, 0.125, 0.50)}
VERSION = "v2"          # prompt builder version; v1 = full mode only (run_llm_v1_astra_full.py.txt), v2 adds lean mode
K_LEAN = 2              # lean mode: nearest 2 rows of the sender account and of the receiver, no device lookup
SEED = 20261115
K, W = 8, 72 * 3600     # neighbourhood: nearest K rows per entity (sender, receiver, device) within +-W seconds
N_DEMO_FRAUD, N_DEMO_LEGIT = 4, 6
CHARS_PER_TOKEN = 1.7   # measured 1.84-2.01 chars per token in probe 1; lower value over-estimates the reservation
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["predictions"], "properties": {
    "predictions": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                    "required": ["txn_id", "label"],
                    "properties": {"txn_id": {"type": "string"}, "label": {"type": "string", "enum": LABELS}}}}}}
FIELDS = ["timestamp", "sender_id", "receiver_id", "receiver_type", "device_id", "amount", "txn_type",
          "merchant_category", "device_city", "sim_age_hours", "device_age_days", "sender_account_age_days",
          "new_payee", "pin_retries", "login_to_txn_sec", "balance_before", "app", "app_version", "ip_risk_score",
          "qr_vpa_mismatch"]

INSTRUCTIONS = """You are the risk team of a payments company. The data are synthetic UPI payments in India. Three months (Jul-Sep 2026) are history and one later month is unseen. Label every TARGET transaction with exactly one of 12 labels.

LABELS (organisers' definitions)
- legit: An ordinary payment. About 96% of all transactions.
- mule_account: Victims are tricked into paying a "mule" account (rented or bought by a fraud ring), which quickly passes the money on. Every payment into the mule in that burst, and the mule's own onward transfers, carry this label.
- velocity_bot: Scripted payments: a lot of small payments from one phone in a few minutes, usually to gaming or e-commerce merchants.
- layering_chain: Stolen money hopping through a chain of accounts (A -> B -> C -> ...), a few minutes apart, a little less each hop.
- sim_swap: The fraudster gets a new SIM for the victim's number, then drains the account, often at night.
- account_takeover: The fraudster logs in to the victim's account from a new phone, often far from where the victim lives, and drains it.
- phishing_collect: The victim approves a fake payment request ("collect"), usually for an amount just under a round limit.
- qr_swap: A fake QR code at a shop: the customer thinks they're paying the shop, but the money goes to someone else.
- refund_abuse: One customer claims refunds from online merchants again and again.
- fake_cashback: "You've won cashback!": the fraudster sends Rs 1-10 to look genuine, then asks for a much bigger payment back.
- loan_app_extortion: Victims of illegal loan apps pay again and again, late in the evening, under threats.
- merchant_collusion: A merchant and a few accounts make fake round-number sales to each other.

NOTES FROM THE ORGANISERS
- You don't have to read these descriptions as rules. Some fraud looks completely ordinary, and some fraud types look like each other.
- A single fraudulent payment usually looks normal on its own. Fraud shows itself only in patterns: who else paid that account in the last hour, how many people used that phone, whether the money left again ten minutes later, whether the same customer has asked for six refunds this week.
- Training labels are not perfect: some fraud was never reported (it is labelled legit) and some was filed under the wrong type. The labelled examples below come from the training months, so their labels contain this noise.
- The unseen month is festival season: people spend more, and the payments app pushed a new version to everyone. Fraud rings adapt too. Not every column behaves the same way in the unseen month as in training.
- Questions a fraud investigator asks: How many different people paid this receiver in the last few hours? How many different accounts used this phone? Did the receiver send the money on soon after? Has this pair of accounts ever paid each other the other way? Is this amount just under a round number? Is this phone new, in a different city from the sender's home, with a fresh SIM?

COLUMNS
timestamp: when it happened, IST, shown as MM-DD HH:MM:SS (year 2026)
sender_id: account that paid (AC...)
receiver_id: who got paid: an account (AC...) or a merchant (MR...)
receiver_type: P2P (a person's account) or MERCHANT
device_id: the phone the payment came from
amount: Rs
txn_type: PAY, QR (scanned a QR code), COLLECT (approved someone's payment request), AUTOPAY, REFUND (refund claim)
merchant_category: for merchant payments
device_city: where the phone was at the time
sim_age_hours: hours since the SIM in that phone last changed
device_age_days: days since this account was first used on this phone
sender_account_age_days: age of the sender's account on that day
new_payee: 1 if the sender had never paid this receiver before
pin_retries: wrong UPI PIN entries before this payment
login_to_txn_sec: seconds between opening the app and paying
balance_before: sender's balance before the payment, Rs
app, app_version: which payments app and version
ip_risk_score: a score from a third-party IP-reputation vendor (0 = safe, 1 = risky)
qr_vpa_mismatch: for QR payments: 1 if the QR's payee didn't match the shop's registered ID. Blank when not checked

INPUT FORMAT
Each transaction to label is one block. The block gives the sender's account record (home_city, bank_code, kyc_level, opened_on, os), the receiver's account record or merchant record (category, city, onboarded_on), and a table of rows in time order. Row format, fields separated by "|":
rel|dt|""" + "|".join(FIELDS) + """
- rel = T marks the transaction to label. Other rows are context: the nearest """ + str(K) + """ transactions (within 72 hours before or after) that involve the same sender account (S), the same receiver (R) or the same phone (D); an account counts in either role, sender or receiver. A row can carry several letters.
- dt = seconds after (+) or before (-) the T row.
- An empty field is a missing value. Values are the raw data values.
- Context rows carry no labels; any of them may be legit or fraud.

OUTPUT
Return JSON only: {"predictions":[{"txn_id":"...","label":"..."}]} with exactly one entry per TARGET txn_id, in the order given, and label one of the 12 names above. No explanations.
"""

CTX_FIELDS = ["sender_id", "receiver_id", "device_id", "amount", "txn_type", "device_city", "new_payee"]
_i = INSTRUCTIONS
INSTR_LEAN = _i[:_i.index("COLUMNS\n")] + """FORMAT
Each transaction to label is one line that starts with its txn_id. Fields are separated by "|"; an empty field is a missing value; values are the raw data values:
txn_id|""" + "|".join(FIELDS) + """|s_home_city|s_bank_code|s_kyc_level|s_opened_on|s_os|r_city|r_bank_code|r_kyc_level|r_opened_on|r_os|r_category
- timestamp is IST, MM-DD HH:MM:SS (year 2026). receiver_id is an account (AC..., receiver_type P2P) or a merchant (MR..., MERCHANT). device_id is the phone the payment came from; device_city is where the phone was. txn_type: PAY, QR (scanned a QR code), COLLECT (approved someone's payment request), AUTOPAY, REFUND (refund claim).
- sim_age_hours: hours since the SIM in that phone last changed. device_age_days: days since this account was first used on this phone. new_payee: 1 if the sender had never paid this receiver before. pin_retries: wrong UPI PIN entries before this payment. login_to_txn_sec: seconds between opening the app and paying. balance_before: sender's balance before the payment. ip_risk_score: third-party IP-reputation score (0 = safe, 1 = risky). qr_vpa_mismatch: for QR payments, 1 if the QR's payee didn't match the shop's registered ID; empty when not checked.
- s_* is the sender's account record (home city, bank, KYC level, opening date, phone OS). r_* is the receiver's account record, or for a merchant its city, onboarding date (r_opened_on) and category (r_category).
Indented lines under a transaction are context rows: up to """ + str(K_LEAN) + """ nearest other transactions (within 72 hours before or after) that involve the sender account (S) and up to """ + str(K_LEAN) + """ that involve the receiver (R), in either role; D marks the same phone. Context rows carry no labels; any of them may be legit or fraud. Their format:
 rel|dt|""" + "|".join(CTX_FIELDS) + """
dt = seconds after (+) or before (-) the transaction.

OUTPUT
Return JSON only: {"predictions":[{"txn_id":"...","label":"..."}]} with exactly one entry per TARGET txn_id, in the order given, and label one of the 12 names above. No explanations.
"""


# ---------------------------------------------------------------- data
class Data:
    def __init__(self):
        tx = load_all().sort("ts", "txn_id")
        c = lambda n: pl.col(n).cast(pl.String).fill_null("")
        acc = lambda p: pl.format("home_city={}|bank_code={}|kyc_level={}|opened_on={}|os={}", *[
            pl.col(p + c).cast(pl.String).fill_null("") for c in ("home_city", "bank_code", "kyc_level", "opened_on", "os")])
        mer = pl.format("merchant category={}|city={}|onboarded_on={}", *[
            pl.col("mer_" + c).cast(pl.String).fill_null("") for c in ("category", "city", "onboarded_on")])
        tx = tx.with_columns(
            pl.concat_str([pl.col("timestamp").dt.strftime("%m-%d %H:%M:%S")] +
                          [pl.col(c).cast(pl.String).fill_null("") for c in FIELDS[1:]], separator="|").alias("line"),
            acc("snd_").alias("smeta"),
            pl.when(pl.col("receiver_type") == "MERCHANT").then(mer).otherwise(acc("rcv_")).alias("rmeta"),
            pl.concat_str([c("snd_home_city"), c("snd_bank_code"), c("snd_kyc_level"), c("snd_opened_on"), c("snd_os"),
                           pl.coalesce("rcv_home_city", "mer_city").fill_null(""), c("rcv_bank_code"), c("rcv_kyc_level"),
                           pl.coalesce("rcv_opened_on", "mer_onboarded_on").cast(pl.String).fill_null(""), c("rcv_os"),
                           c("mer_category")], separator="|").alias("lmeta"),
            pl.concat_str([c(x) for x in CTX_FIELDS], separator="|").alias("cline"))
        self.lmeta, self.cline = tx["lmeta"].to_list(), tx["cline"].to_list()
        self.meta = tx.select("txn_id", "month", "is_test", "label")   # label only used by build-pilot
        self.ids = tx["txn_id"].to_list()
        self.row = {t: i for i, t in enumerate(self.ids)}
        self.ts = tx["ts"].to_numpy()
        self.snd, self.rcv, self.dev = (tx[c].to_list() for c in ("sender_id", "receiver_id", "device_id"))
        self.line, self.smeta, self.rmeta = (tx[c].to_list() for c in ("line", "smeta", "rmeta"))
        ent = pl.concat([tx.select(pl.col(c).alias("e"), "ts", pl.int_range(pl.len(), dtype=pl.Int32).alias("i"))
                         for c in ("sender_id", "receiver_id", "device_id")]).sort("e", "ts", "i")
        g = ent.group_by("e", maintain_order=True).len()
        self.start = np.concatenate([[0], np.cumsum(g["len"].to_numpy())]).astype(np.int64)
        self.pos = dict(zip(g["e"].to_list(), range(g.height)))
        self.ets, self.ei = ent["ts"].to_numpy(), ent["i"].to_numpy()

    def lean_block(self, i, head):
        t = int(self.ts[i])
        rows = sorted(self.neigh(i, K_LEAN, False), key=lambda x: (int(self.ts[x[0]]), x[0]))
        return "\n".join([f"{head}|{self.line[i]}|{self.lmeta[i]}"] +
                         [f" {rel}|{int(self.ts[j]) - t:+d}|{self.cline[j]}" for j, rel in rows])

    def neigh(self, i, K=K, dev=True):
        """Label-independent rule: per entity the K rows nearest in time within +-W. Returns [(row, rel)]."""
        t, s, r, d = int(self.ts[i]), self.snd[i], self.rcv[i], self.dev[i]
        sel = set()
        for e in ((s, r, d) if dev else (s, r)):
            a, b = self.start[self.pos[e]], self.start[self.pos[e] + 1]
            ts = self.ets[a:b]
            p = int(np.searchsorted(ts, t))
            cand = [(abs(int(ts[j]) - t), int(self.ei[a + j])) for j in range(max(0, p - K - 1), min(b - a, p + K + 2))]
            sel.update(j for dt, j in sorted(c for c in cand if c[1] != i and c[0] <= W)[:K])
        out = []
        for j in sel:
            rel = ("S" if s in (self.snd[j], self.rcv[j]) else "") + ("R" if r in (self.snd[j], self.rcv[j]) else "") \
                + ("D" if self.dev[j] == d else "")
            out.append((j, rel))
        return out

    def block(self, i, head):
        t = int(self.ts[i])
        rows = sorted(self.neigh(i) + [(i, "T")], key=lambda x: (int(self.ts[x[0]]), x[0]))
        return "\n".join([head, f"sender {self.snd[i]}: {self.smeta[i]}", f"receiver {self.rcv[i]}: {self.rmeta[i]}"] +
                         [f"{rel}|{int(self.ts[j]) - t:+d}|{self.line[j]}" for j, rel in rows])

    def entities(self, i):
        e = set()
        for j in [i] + [j for j, _ in self.neigh(i)]:
            e.update((self.snd[j], self.rcv[j], self.dev[j]))
        return e


def build_lean(D):
    """Lean static prompt: same demonstrations file, first 2 per fraud label and first 4 legit in its order."""
    left = {l: (4 if l == "legit" else 2) for l in LABELS}
    pick = []
    for d in json.loads((HERE / "demos.json").read_text()):
        if left[d["label"]]:
            left[d["label"]] -= 1
            pick.append(d)
    static = INSTR_LEAN + "\nLABELLED EXAMPLES (training months; labels are noisy)\n" + "\n".join(
        D.lean_block(D.row[d["txn_id"]], f"EX{n + 1}") + f"\nlabel: {d['label']}" for n, d in enumerate(pick))
    (HERE / "prompt_static_lean.txt").write_text(static, encoding="utf-8")
    print("lean demos", len(pick), "static chars", len(static))


def dynamic_prompt(D, ids, mode="full"):
    if mode == "lean":
        return f"Label these {len(ids)} TARGET transactions.\n" + "\n".join(D.lean_block(D.row[t], t) for t in ids)
    return (f"Label these {len(ids)} TARGET transactions: {', '.join(ids)}\n\n" +
            "\n\n".join(D.block(D.row[t], f"### TARGET {t}") for t in ids))


def build_pilot(D):
    m = D.meta.sort("txn_id")
    sep = m.filter((pl.col("month") == 9) & ~pl.col("is_test"))
    nat = sep.sample(300, seed=SEED, shuffle=True).with_columns(pl.lit("natural").alias("subset"))
    rest = sep.join(nat, on="txn_id", how="anti")
    enr = pl.concat([rest.filter(pl.col("label") == l).sample(60 if l == "legit" else 15, seed=SEED + k, shuffle=True)
                     for k, l in enumerate(LABELS)]).with_columns(pl.lit("enriched").alias("subset"))
    s = pl.concat([nat, enr]).select("txn_id", "subset").sample(fraction=1.0, seed=SEED, shuffle=True)
    assert s.height == 525 and s["txn_id"].n_unique() == 525
    s.write_csv(HERE / "pilot_sample_ids.csv")          # no labels; order shuffled
    pilot_ent = set()
    for t in s["txn_id"]:
        i = D.row[t]
        pilot_ent.update((D.snd[i], D.rcv[i], D.dev[i]))
    # demonstrations: July/August, fixed-seed random per label; a candidate is skipped only when it (or a row of
    # its neighbourhood) shares a sender, receiver or device with a pilot row. No pilot label is used here.
    early = m.filter(pl.col("month").is_in([7, 8]) & ~pl.col("is_test"))
    demos = []
    for k, l in enumerate(LABELS):
        cand = early.filter(pl.col("label") == l)["txn_id"].to_list()
        random.Random(SEED + k).shuffle(cand)
        need = N_DEMO_LEGIT if l == "legit" else N_DEMO_FRAUD
        for t in cand:
            if need == 0:
                break
            if not (D.entities(D.row[t]) & pilot_ent):
                demos.append({"txn_id": t, "label": l})
                need -= 1
        assert need == 0, l
    random.Random(SEED).shuffle(demos)
    (HERE / "demos.json").write_text(json.dumps(demos, indent=1))
    static = INSTRUCTIONS + "\nLABELLED EXAMPLES (training months; labels are noisy)\n\n" + "\n\n".join(
        D.block(D.row[d["txn_id"]], f"### EXAMPLE {n + 1}") + f"\nlabel: {d['label']}" for n, d in enumerate(demos))
    (HERE / "prompt_static.txt").write_text(static, encoding="utf-8")
    print("pilot ids", s.height, "demos", len(demos), "static chars", len(static))


# ---------------------------------------------------------------- API
def api_key():
    for ln in (ROOT / ".env.txt").read_text().splitlines():
        if ln.startswith("OPENAI_API_KEY="):
            return ln.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no OPENAI_API_KEY in .env.txt")


def cost_of(u, model=MODEL):
    p_in, p_cached, p_write, p_out = PRICES[model]
    d = u.get("input_tokens_details") or {}
    cached = d.get("cached_tokens", 0) or 0
    write = d.get("cache_write_tokens", d.get("cache_creation_tokens", 0)) or 0
    return ((u.get("input_tokens", 0) - cached - write) * p_in + cached * p_cached + write * p_write
            + u.get("output_tokens", 0) * p_out) / 1e6


def worst_cost(static, dyn, max_out, model=MODEL):
    return (len(static) + len(dyn)) / CHARS_PER_TOKEN * PRICES[model][2] / 1e6 + max_out * PRICES[model][3] / 1e6


class Bucket:
    """Client-side tokens-per-minute limiter (counts every input token, cached or not, plus expected output)."""
    def __init__(self, tpm):
        self.rate, self.cap, self.lock = tpm / 60.0, tpm / 12.0, threading.Lock()
        self.tok, self.t = self.cap, time.time()

    def take(self, n):
        while True:
            with self.lock:
                now = time.time()
                self.tok, self.t = min(self.cap, self.tok + (now - self.t) * self.rate), now
                if self.tok >= min(n, self.cap):   # a request larger than the burst size may overdraw
                    self.tok -= n
                    return
                need = (n - self.tok) / self.rate
            time.sleep(min(need, 1.0) + 0.01)

    def penalty(self, sec):                 # after a 429: nobody sends for `sec` seconds
        with self.lock:
            self.tok = min(self.tok, -self.rate * sec)


def stop_epoch(hhmm):
    if not hhmm:
        return float("inf")
    lt, (h, m) = time.localtime(), map(int, hhmm.split(":"))
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, h, m, 0, 0, 0, -1))


def post(key, static, dyn, max_out, effort, model=MODEL, timeout=900):
    # explicit cache breakpoint after the static block: only the static prefix is written to / read from the cache
    body = {"model": model, "store": False, "prompt_cache_options": {"mode": "explicit"},
            "input": [{"role": "developer", "content": [{"type": "input_text", "text": static,
                                                         "prompt_cache_breakpoint": {"mode": "explicit"}}]},
                      {"role": "user", "content": dyn}],
            "reasoning": {"effort": effort}, "max_output_tokens": max_out,
            "text": {"format": {"type": "json_schema", "name": "predictions", "strict": True, "schema": SCHEMA}}}
    return requests.post(URL, headers={"Authorization": "Bearer " + key}, json=body, timeout=(15, timeout))


def parse(resp_json, ids):
    """Whole response valid or not at all: status completed, every id exactly once, no extras, labels in vocabulary."""
    if resp_json.get("status") != "completed":
        return None, f"status={resp_json.get('status')} {resp_json.get('incomplete_details')}"
    text = "".join(c.get("text", "") for o in resp_json.get("output", []) if o.get("type") == "message"
                   for c in o.get("content", []) if c.get("type") == "output_text")
    try:
        preds = json.loads(text)["predictions"]
        got = [p["txn_id"] for p in preds]
        if sorted(got) != sorted(ids):
            return None, "id mismatch"
        if any(p["label"] not in LABELS for p in preds):
            return None, "label not in vocabulary"
        return {p["txn_id"]: p["label"] for p in preds}, None
    except (ValueError, KeyError, TypeError) as e:
        return None, f"bad json: {e!r}"


class Ledger:
    """Hard cap: spent + reserved (in-flight worst case) never exceeds cap."""
    def __init__(self, cap, spent):
        self.cap, self.spent, self.res, self.cv = cap, spent, 0.0, threading.Condition()

    def reserve(self, amt):
        with self.cv:
            while self.spent + self.res + amt > self.cap:
                if self.res <= 1e-12:
                    return False            # nothing in flight can free budget
                self.cv.wait()
            self.res += amt
            return True

    def settle(self, amt, cost):
        with self.cv:
            self.res -= amt
            self.spent += cost
            self.cv.notify_all()


def selftest():
    ids = ["a", "b"]
    ok = {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(
        {"predictions": [{"txn_id": "b", "label": "legit"}, {"txn_id": "a", "label": "qr_swap"}]})}]}]}
    assert parse(ok, ids)[0] == {"a": "qr_swap", "b": "legit"}
    assert parse(ok, ["a"])[0] is None and parse(ok, ["a", "b", "c"])[0] is None and parse(ok, ["a", "a"])[0] is None
    assert parse({**ok, "status": "incomplete"}, ids)[0] is None
    bad = json.loads(json.dumps(ok).replace("qr_swap", "qr"))
    assert parse(bad, ids)[0] is None
    L = Ledger(1.0, 0.5)
    assert L.reserve(0.4) and not Ledger(1.0, 0.5).reserve(0.6)
    L.settle(0.4, 0.1)
    assert abs(L.spent - 0.6) < 1e-9 and L.res < 1e-9 and not L.reserve(0.41)
    assert abs(cost_of({"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 400}, "output_tokens": 100})
               - (600 * 10 + 400 * 1 + 100 * 50) / 1e6) < 1e-12


DEC_URL = "https://api.openai.com/v1/decisions"
PRICES["decisions"] = (0.10, 0.01, 0.125, 0.0)      # Decisions API guide: $0.10/M uncached input, output not charged, luna cache rates
DEC_OUTPUT = ("ANSWERING\nThere is one question per TARGET transaction, named by its txn_id. For each question choose the one "
              "label, from the 12 names above, of that transaction.\n")


def post_decisions(key, static, dyn, ids, timeout):
    body = {"model": "gpt-6-luna", "prompt_cache_options": {"mode": "explicit"},
            "input": [{"role": "developer", "content": [{"type": "input_text", "text": static,
                                                         "prompt_cache_breakpoint": {"mode": "explicit"}}]},
                      {"role": "user", "content": dyn}],
            "questions": [{"type": "choice", "name": t, "instructions": f"Label of TARGET {t}",
                           "choices": [{"value": l} for l in LABELS]} for t in ids]}
    return requests.post(DEC_URL, headers={"Authorization": "Bearer " + key}, json=body, timeout=(15, timeout))


def parse_decisions(j, ids):
    """Valid only if every id is answered exactly once by a choice answer (no refusal) with a label in the vocabulary."""
    try:
        ans = j["answers"]
        if sorted(x["name"] for x in ans) != sorted(ids):
            return None, "id mismatch"
        if any(x.get("type") != "choice" or x.get("choice") not in LABELS for x in ans):
            return None, "refusal or label not in vocabulary"
        return {x["name"]: x["choice"] for x in ans}, None
    except (KeyError, TypeError) as e:
        return None, f"bad response: {e!r}"


def selftest2():
    ok = {"answers": [{"type": "choice", "name": "a", "choice": "legit"}, {"type": "choice", "name": "b", "choice": "qr_swap"}]}
    assert parse_decisions(ok, ["b", "a"])[0] == {"a": "legit", "b": "qr_swap"}
    assert parse_decisions(ok, ["a"])[0] is None and parse_decisions({}, ["a"])[0] is None
    assert parse_decisions({"answers": [{"type": "refusal", "name": "a"}]}, ["a"])[0] is None
    b = Bucket(600)
    t0 = time.time(); b.take(50); b.take(10)
    assert time.time() - t0 > 0.5                    # 600/min = 10/s, burst 50: the next 10 tokens wait about 1 s


def run(D, ids, kind, tag, a, log_prompts, tries=3):
    dec = a.api == "decisions"
    model = "gpt-6-luna" if dec else a.model
    pk = "decisions" if dec else model
    static = (HERE / f"prompt_static_{a.mode}.txt").read_text(encoding="utf-8")
    if dec:
        static = static[:static.index("OUTPUT\n")] + DEC_OUTPUT + static[static.index("\nLABELLED EXAMPLES"):]
    shash = hashlib.sha256(static.encode()).hexdigest()[:16]
    full = kind == "full"
    # one method per output file: the default full run owns full_*.jsonl; any other configuration needs --tag
    sfx = f"_{a.tag}" if a.tag else ""
    assert not full or a.tag or (a.api, a.mode) == ("decisions", "lean"), "pass --tag for a second full configuration"
    pred_f = HERE / (f"full_predictions{sfx}.jsonl" if full else f"{kind}_predictions_{tag}.jsonl")
    req_f = HERE / (f"full_requests{sfx}.jsonl" if full else f"{kind}_requests_{tag}.jsonl")
    run_log, csv_f = HERE / f"full_run{sfx}.log", HERE / f"llm_direct{sfx}.csv"
    done = {json.loads(l)["txn_id"] for l in pred_f.read_text().splitlines()} if pred_f.exists() else set()
    # cap scope: the full run counts its own log; probes and pilots share one cap over all their logs
    logs = [req_f] if full else [f for f in HERE.glob("*_requests*.jsonl") if not f.name.startswith("full_")]
    spent0 = sum(json.loads(l)["cost"] for f in logs if f.exists() for l in f.read_text(encoding="utf-8").splitlines())
    idset = set(ids)
    n0 = len(done & idset)
    print(f"{kind} {tag}: {len(ids)} targets, {n0} done, spent so far ${spent0:.4f}, cap ${a.cap}, model {model}, api {a.api}, "
          f"mode {a.mode}, static {shash}", flush=True)
    led, lock, key, stop = Ledger(a.cap, spent0), threading.Lock(), api_key(), threading.Event()
    bucket, deadline, fin = Bucket(a.tpm), stop_epoch(a.stop_at), threading.Event()
    stats = {"ok": 0, "invalid": 0, "err": 0, "429": 0, "rows": 0, "tok": 0}
    t_start = time.time()

    def status():
        el = time.time() - t_start
        rate = stats["rows"] / max(el, 1e-9)
        left = len(ids) - n0 - stats["rows"]
        return (f"{time.strftime('%H:%M:%S')} rows {n0 + stats['rows']}/{len(ids)} ({100 * (n0 + stats['rows']) / len(ids):.1f}%) "
                f"{rate:.1f} rows/s spend ${led.spent:.3f} eta {left / max(rate, 1e-9) / 60:.1f} min | requests ok {stats['ok']} "
                f"invalid {stats['invalid']} errors {stats['err']} http429 {stats['429']} tokens/min {stats['tok'] / max(el, 1e-9) * 60:,.0f}")

    def progress():
        while not fin.wait(30):
            with open(run_log, "a", encoding="utf-8") as f:
                f.write(status() + "\n")

    def log(rec):
        with lock, open(req_f, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def work(args):
        p, bi, bids = args
        dyn = dynamic_prompt(D, bids, a.mode)
        est = worst_cost(static, dyn, 0 if dec else a.max_out, pk) + (len(bids) * 400 * 0.125 / 1e6 if dec else 0)
        ntok = (len(static) + len(dyn)) / a.cpt + len(bids) * (a.qtok if dec else 18)
        assert ntok < 250_000                        # stay under the 272K long-context price step
        attempt = n429 = 0
        while attempt < tries and n429 < 40:
            if stop.is_set() or time.time() > deadline:
                return
            bucket.take(ntok)
            if time.time() > deadline:
                return
            if not led.reserve(est):
                stop.set()
                print(f"CAP: refusing request {bi} (spent ${led.spent:.4f} + worst case ${est:.4f} > cap ${a.cap})", flush=True)
                return
            rec = {"req": f"{kind}-{tag}-p{p}-{bi}-{attempt}.{n429}", "time": time.time(), "api": a.api, "model": model,
                   "effort": None if dec else a.effort, "max_output_tokens": None if dec else a.max_out, "builder": VERSION,
                   "mode": a.mode, "static_hash": shash, "target_ids": bids,
                   "prompt_hash": hashlib.sha256((static + dyn).encode()).hexdigest()[:16], "cost": 0.0}
            if log_prompts:
                rec["prompt"] = dyn                  # static part: prompt_static_<mode>.txt (+ DEC_OUTPUT swap), see static_hash
            t0, wait = time.time(), 2.0 * 2 ** attempt + random.random()
            try:
                r = post_decisions(key, static, dyn, bids, a.timeout) if dec else \
                    post(key, static, dyn, a.max_out, a.effort, model, a.timeout)
            except requests.RequestException as e:   # outcome unknown: may have been billed, keep the reservation as spent
                rec.update(error=f"{type(e).__name__}", latency=time.time() - t0, cost=est, valid=False)
                led.settle(est, est); log(rec); stats["err"] += 1; attempt += 1; time.sleep(wait); continue
            rec.update(http=r.status_code, latency=time.time() - t0,
                       ratelimit={k: v for k, v in r.headers.items() if k.lower().startswith("x-ratelimit")})
            if r.status_code != 200:                 # rejected or failed requests are not billed
                rec.update(error=r.text[:2000], valid=False)
                led.settle(est, 0.0); log(rec)
                ra = r.headers.get("retry-after")
                ra = float(ra) if ra and ra.replace(".", "").isdigit() else None
                if r.status_code == 429:             # adaptive backoff: everyone pauses, this request waits longer each time
                    stats["429"] += 1; n429 += 1
                    bucket.penalty(ra or 3.0); time.sleep((ra or min(30.0, 1.5 ** n429)) + random.random()); continue
                stats["err"] += 1
                if r.status_code not in (500, 502, 503, 504, 520, 529):
                    print("fatal http", r.status_code, r.text[:300], flush=True); return
                attempt += 1; time.sleep(wait); continue
            j = r.json()
            preds, err = parse_decisions(j, bids) if dec else parse(j, bids)
            u = j.get("usage") or {}
            c = cost_of(u, pk)
            rec.update(response=j, usage=u, cost=c, valid=preds is not None, error=err)
            led.settle(est, c); log(rec)
            stats["tok"] += u.get("input_tokens", 0) + u.get("output_tokens", 0)
            if preds is None:
                stats["invalid"] += 1; attempt += 1
                continue                             # retried as a whole; nothing is invented or defaulted
            with lock, open(pred_f, "a", encoding="utf-8") as f:
                for t in bids:
                    f.write(json.dumps({"txn_id": t, "label": preds[t], "req": rec["req"]}) + "\n")
                done.update(bids)
                stats["ok"] += 1; stats["rows"] += len(bids)
            return

    def safe(args):
        try:
            work(args)
        except Exception as e:                       # one bad batch must not stop the run; its rows stay unfinished
            stats["err"] += 1
            print("worker error", repr(e)[:300], flush=True)

    if full:
        threading.Thread(target=progress, daemon=True).start()
    for p in range(a.passes):                        # later passes retry rows left unfinished, while time remains
        todo = [t for t in ids if t not in done]
        if not todo or stop.is_set() or time.time() > deadline:
            break
        batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
        if p == 0:
            safe((p, 0, batches[0]))                 # first alone so the static prefix is cached before fan-out
        with ThreadPoolExecutor(a.conc) as ex:
            list(ex.map(safe, [(p, bi, b) for bi, b in enumerate(batches)][1 if p == 0 else 0:]))
    fin.set()
    cov = len(done & idset)
    msg = f"END {kind} {tag}: coverage {cov}/{len(ids)} valid rows, unfinished {len(ids) - cov}; " + status()
    print(msg, flush=True)
    if full:
        with open(run_log, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        if cov == len(ids) == 260582:                # complete only: submission-format file, sample order, nothing defaulted
            lab = {}
            for l in pred_f.read_text().splitlines():
                d = json.loads(l)
                lab[d["txn_id"]] = d["label"]
            assert all(lab[t] in LABELS for t in ids)
            pl.DataFrame({"id": ids, "label": [lab[t] for t in ids]}).write_csv(csv_f)
            print("wrote", csv_f.name, flush=True)


def main():
    selftest(); selftest2()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe0", "build-pilot", "build-lean", "probe", "pilot", "full"])
    ap.add_argument("--api", default="responses", choices=["responses", "decisions"])
    ap.add_argument("--model", default=MODEL, choices=[m for m in PRICES if m != "decisions"])
    ap.add_argument("--mode", default="full", choices=["full", "lean"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--cap", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--max-out", type=int, default=8000)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--tpm", type=float, default=900_000, help="client-side tokens/min budget (all input counted)")
    ap.add_argument("--cpt", type=float, default=1.85, help="chars per token used by the tokens/min limiter")
    ap.add_argument("--qtok", type=float, default=90, help="decisions: estimated tokens per question")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--stop-at", default="", help="HH:MM local (IST): no new requests after this time")
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "probe0":
        r = post(api_key(), "Reply with the JSON object requested.", "Label TX1 as legit.", 2000, a.effort, a.model)
        print(r.status_code, {k: v for k, v in r.headers.items() if k.lower().startswith("x-ratelimit")})
        j = r.json()
        with open(HERE / "probe0_requests.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"req": "probe0", "http": r.status_code, "response": j, "cost": cost_of(j.get("usage") or {}, a.model)}) + "\n")
        print(json.dumps({k: j.get(k) for k in ("status", "model", "usage", "error", "output", "reasoning", "service_tier")}, indent=1)[:3000])
        return
    D = Data()
    if a.cmd == "build-pilot":
        return build_pilot(D)
    if a.cmd == "build-lean":
        return build_lean(D)
    tag = a.tag or f"{'decisions' if a.api == 'decisions' else a.model.split('-')[-1]}_{a.mode}"
    pilot = pl.read_csv(HERE / "pilot_sample_ids.csv")["txn_id"].to_list()
    if a.cmd == "probe":
        pool = D.meta.filter((pl.col("month") == 9) & ~pl.col("is_test") & ~pl.col("txn_id").is_in(pilot)).sort("txn_id")
        ids = pool.sample(a.n, seed=a.seed)["txn_id"].to_list()
        a.cap, a.batch, a.conc, a.passes = a.cap or 2.0, a.n, 1, 1
        run(D, ids, f"probe{a.seed}", tag, a, True, tries=1)
    elif a.cmd == "pilot":
        assert a.cap > 0, "--cap required"
        run(D, pilot[:a.limit or None], "pilot", tag, a, True)
    else:
        assert a.cap > 0, "--cap required"
        from train import SAMPLE
        ids = pl.read_csv(SAMPLE)["id"].to_list()
        run(D, ids[:a.limit or None], "full", tag, a, False)


if __name__ == "__main__":
    main()
