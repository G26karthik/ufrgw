# Frozen configuration of the shipped LLM run

Timeline (file times, IST, 15 Nov 2026): lean static prompt built 08:42:47; full run started 08:44:00; Decisions pilot predictions saved 08:46:49. Agent A read no pilot label after the sample was built, opened no file in `check/` and computed no score. Agent B's `check/pilot_predictions_eval.md` (evaluation of the earlier astra pilot on the same 525 ids) has file time 08:40:21, before the lean prompt was built; Agent A did not open it, so the lean prompt was not written against it.

| item | value |
|---|---|
| endpoint | `POST https://api.openai.com/v1/decisions` |
| model | `gpt-6-luna` (the only model this endpoint accepts) |
| parameters | `prompt_cache_options: {"mode": "explicit"}`; no reasoning, temperature or output-token parameter exists on this endpoint |
| builder | `run_llm.py`, `VERSION = "v2"`, `--api decisions --mode lean` |
| static prompt (developer message, cache breakpoint at its end) | `prompt_static_decisions_lean.txt`, sha256 prefix `604a5ec7c68db6c9` (logged as `static_hash` in every request) |
| target table (user message) | `Label these N TARGET transactions.` + one line per target + its context rows, built by `Data.lean_block` |
| questions | one per target: `{"type": "choice", "name": <txn_id>, "instructions": "Label of TARGET <txn_id>", "choices": [{"value": <label>} x 12]}` in `common.LABELS` order |
| label taken | the returned `choice`, unchanged. No rule, threshold or default on top |
| targets per request | 300 |
| demonstrations | 26 rows from July/August (`demos.json`: first 2 per fraud label and first 4 legit in file order) |
| context rule | nearest 2 other rows of the sender account and nearest 2 of the receiver (either role), within 72 h before or after; no labels |

## Request body

```json
{"model": "gpt-6-luna",
 "prompt_cache_options": {"mode": "explicit"},
 "input": [
   {"role": "developer", "content": [{"type": "input_text", "text": "<prompt_static_decisions_lean.txt>",
                                      "prompt_cache_breakpoint": {"mode": "explicit"}}]},
   {"role": "user", "content": "<target table>"}],
 "questions": [{"type": "choice", "name": "TX...", "instructions": "Label of TARGET TX...",
                "choices": [{"value": "legit"}, {"value": "mule_account"}, "... 12 labels"]}]}
```

## Static prompt text

The exact text is the file `prompt_static_decisions_lean.txt` (13 kB: task, 12 label definitions, organisers' notes, format, 26 labelled examples). It equals `prompt_static_lean.txt` with the `OUTPUT` paragraph replaced by:

```
ANSWERING
There is one question per TARGET transaction, named by its txn_id. For each question choose the one label, from the 12 names above, of that transaction.
```

## What is in the prompt and what is not

- In: organisers' task text and label definitions (`upi-fraud-rings/PROBLEM_STATEMENT.md`), all 20 original fields of each target (raw values, missing = empty), the sender's account record and the receiver's account or merchant record, context rows (7 fields), 26 labelled July/August examples.
- Not in: model predictions, probabilities, submission labels, rule outputs, engineered features, labels of context rows, anything from `work/subs`, `work/oof`, `work/preds`, `scratch_gap`.
- The demonstrations share no sender, receiver or device with a pilot row (the check covers each demonstration and its full-mode neighbourhood rows).

## Other configuration (earlier pilot, not shipped)

`pilot_predictions_astra_full.jsonl`: `gpt-6-astra`, `/v1/responses`, `reasoning.effort = "high"`, `max_output_tokens = 8000`, strict `json_schema` output, 25 targets per request, static prompt `prompt_static_full.txt` (50 demonstrations, nearest 8 rows per sender, receiver and device within 72 h, all fields). Code as run: `run_llm_v1_astra_full.py.txt` (builder v1).

## Exact command of the full run (started 08:44:00 IST, cwd `work`)

```bash
PYTHONPATH=. PYTHONUTF8=1 POLARS_MAX_THREADS=4 ../.venv/Scripts/python.exe scratch_llm/run_llm.py full --api decisions --mode lean --batch 300 --conc 6 --tpm 1930000 --cpt 1.85 --qtok 164 --cap 12 --stop-at 09:42 --passes 6 --timeout 120
```

`--batch` targets per request; `--conc` worker threads; `--tpm` client-side tokens/min budget; `--cpt` characters per token and `--qtok` tokens per question for that budget; `--cap` hard dollar cap; `--stop-at` no new requests after this local time; `--passes` extra passes over unfinished rows.

## Reproduction steps

1. Put `OPENAI_API_KEY=...` in `.env.txt` at the repo root. The key needs access to `/v1/decisions`.
2. Build the data as in the repo README (`python prep.py` gives `work/data/*.parquet`).
3. From `work/`: `python scratch_llm/run_llm.py build-pilot`, then `python scratch_llm/run_llm.py build-lean`. Both are deterministic (seed 20261115). They rewrite `pilot_sample_ids.csv`, `demos.json`, `prompt_static.txt` and `prompt_static_lean.txt`. Check: the `static_hash` printed by the run must be `604a5ec7c68db6c9`.
4. Run the command above. Drop `--stop-at` on another day. Rerun the same line to resume; finished txn_ids are skipped.
5. `python scratch_llm/verify_full.py` checks coverage, vocabulary, one configuration per file and the csv order.
6. `llm_direct.csv` (`id,label`, sample-submission order) is written only when all 260,582 rows have a valid answer.

The model's answers are not guaranteed to be identical on a rerun: the endpoint has no seed or temperature parameter. The raw answers of this run, with probabilities, are in `full_requests.jsonl`.
