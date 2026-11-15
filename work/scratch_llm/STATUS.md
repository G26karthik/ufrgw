# scratch_llm status (Agent A)

Updated 08:50 IST, 15 Nov 2026.

## State

| item | state |
|---|---|
| Full run, 260,582 October rows, Decisions API + `gpt-6-luna`, lean prompt | RUNNING since 08:44:00 IST. Progress: `full_run.log` (one line per 30 s; last line starts with `END` when the process stops). No new requests after 09:42 IST |
| Blinded pilot, shipped configuration | DONE: `pilot_predictions_decisions.jsonl`, 525 rows, 0 invalid |
| Blinded pilot, `gpt-6-astra` full context | DONE: `pilot_predictions_astra_full.jsonl` (copy: `pilot_predictions.jsonl`), 525 rows, 0 invalid |
| Responses-API lean run on `gpt-6-luna` | NOT RUN (replaced by the Decisions path; code path exists, never probed) |
| `llm_direct.csv` | written by the runner only if coverage is exactly 260,582 valid rows |

## Files

| file | content |
|---|---|
| `run_llm.py` | the runner (builder `v2`). `run_llm_v1_astra_full.py.txt` = the code that made the astra pilot |
| `FROZEN.md` | frozen model, parameters and prompt of the shipped run |
| `FEASIBILITY.md` | measured tokens, cost, speed, limits and projections |
| `prompt_static_decisions_lean.txt` | exact static prompt of the shipped run (static_hash `604a5ec7c68db6c9`) |
| `prompt_static_lean.txt` | lean static prompt for the Responses API (same text, `OUTPUT` paragraph instead of `ANSWERING`) |
| `prompt_static_full.txt` (= `prompt_static.txt`) | full-context static prompt of the astra pilot |
| `demos.json` | the 50 July/August demonstration rows (txn_id, label); lean mode uses the first 2 per fraud label and first 4 legit |
| `pilot_sample_ids.csv` | `txn_id,subset` (natural 300 / enriched 225), shuffled, NO labels |
| `pilot_predictions_<tag>.jsonl` | one line per row: `{"txn_id", "label", "req"}` |
| `pilot_requests_<tag>.jsonl` | one line per request attempt: `req`, `time`, `api`, `model`, `effort`, `max_output_tokens`, `builder`, `mode`, `static_hash`, `target_ids`, `prompt_hash`, `prompt` (the dynamic part: target table), `http`, `latency`, `ratelimit` (x-ratelimit headers only), `response` (raw JSON), `usage`, `cost`, `valid`, `error` |
| `full_predictions.jsonl` | append-only, same format as pilot predictions; partial until `full_run.log` ends with `END ... coverage 260582/260582` |
| `full_requests.jsonl` | as pilot requests, without `prompt` (rebuild it from `builder` + `target_ids`) |
| `probe*_requests*.jsonl`, `probe*_predictions*.jsonl` | probe requests on non-pilot September rows |

Decisions response format (`response` in the request log): `{"model", "answers": [{"type": "choice", "name": <txn_id>, "choice": <label>, "probabilities": [{"value", "probability"} x 12], "confidence"}], "usage"}`. The predictions file holds `choice`; the probabilities are only in the request log.

## How to run (cwd `work`)

```bash
export PYTHONPATH=. PYTHONUTF8=1 POLARS_MAX_THREADS=4
# full run (resumes: finished txn_ids are skipped, the cap counts every logged attempt of full_requests.jsonl)
../.venv/Scripts/python.exe scratch_llm/run_llm.py full --api decisions --mode lean --batch 300 --conc 6 --tpm 1930000 --cpt 1.85 --qtok 164 --cap 12 --stop-at 09:42 --passes 6 --timeout 120
# pilots
../.venv/Scripts/python.exe scratch_llm/run_llm.py pilot --api decisions --mode lean --tag decisions --batch 300 --conc 2 --tpm 300000 --cpt 1.85 --qtok 164 --cap 25
../.venv/Scripts/python.exe scratch_llm/run_llm_v1_astra_full.py.txt pilot --cap 25 --batch 25 --conc 12 --max-out 8000 --effort high   # as run at 08:37
# rebuild sample ids, demonstrations and static prompts (deterministic, seed 20261115)
../.venv/Scripts/python.exe scratch_llm/run_llm.py build-pilot && ../.venv/Scripts/python.exe scratch_llm/run_llm.py build-lean
```

## Rules the runner enforces

- Cap in dollars: before each send it reserves a worst-case cost (all input priced as cache write + maximum output); it refuses when spent + reserved + this request exceeds the cap. Probes and pilots share one cap over all their logs; the full run counts its own log. A request with an unknown outcome (timeout) keeps its reservation as spent.
- A response is accepted whole or not at all: every requested id exactly once, no extra id, `choice` answer (no refusal), label in the 12-name vocabulary. An invalid batch is retried (3 tries, then again in up to 6 passes while time remains). Nothing is defaulted.
- Tokens/min limiter on the client (all input tokens counted) and a pause for every sender after an HTTP 429.
- The API key is read from `.env.txt` inside the process; it is never printed or logged. Only `x-ratelimit-*` response headers are logged.

## Known limits

- The bucket fix in `Bucket.take` (a request larger than the burst size may overdraw) was made at 08:46, after the full run had started; the running process has the older code, which is equivalent at its settings.
- Agent A did not read `check/` results and did not compute scores.
