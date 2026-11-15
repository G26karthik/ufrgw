# Direct LLM classification of the test transactions (15 Nov 2026)

Question: can a general LLM, given only original fields and label-free context, label UPI transactions well enough to
stand as an independent submission, or to correct the ranked pipeline?
Answer: no. Both setups score below the pipeline on a sealed September pilot, and no rule that lets the LLM override
the pipeline gains. The LLM-only file exists as a complete experiment; it is not a candidate for the ranked score.
`subs/dd_final.csv` is unchanged and does not use any LLM output.

Rule note: the problem statement (line 115) allows "publicly available pretrained weights". The models used here are
reached through a paid API, not public weights. The LLM file was therefore never meant as the ranked upload.

## Setup (`scratch_llm/`, frozen configuration in `scratch_llm/FROZEN.md`)
- Two agents with separate duties. A: context builder, runner, probes, pilot predictions (`run_llm.py`). B: isolation
  checks and scoring (`check/`). A never opened labels of the pilot rows or B's score files.
- The prompt holds: the organisers' task text and label definitions, the 20 original fields of each target, the
  sender's account record, the receiver's account or merchant record, label-free neighbouring rows, and labelled
  examples from July / August only.
- The prompt never holds: model predictions, probabilities, submission labels, rule outputs, engineered features,
  labels of context rows. Context selection does not depend on the target's label. B's `check_isolation.py` passed on
  both pilots and on the full run's requests.
- The API key is read from a git-ignored local file; it is not printed, logged or committed.
- Pilot: 525 sealed September rows (300 natural, 225 enriched for fraud), ids fixed before any call
  (`pilot_sample_ids.csv`). Scores are reweighted per true class to the September distribution.

## Pilot results (September, reused development evidence; baseline = pipeline label on the same rows)
| setup | reweighted macro-F1 | baseline | LLM right, baseline wrong | baseline right, LLM wrong | cost |
|---|---|---|---|---|---|
| `gpt-6-astra`, `/v1/responses`, reasoning high, full context, 25 targets per request | .437 (pooled .426) | .514 (.506) | 0 | 13 | $4.57 |
| `gpt-6-luna`, `/v1/decisions`, lean context, 300 targets per request | .250 | .506 | 1 | 39 | under $0.01 |

- Natural sample, Decisions setup: legit rows called fraud 13 of 283 (4.6%); fraud-versus-legit precision .350 against
  .889 for the pipeline. At October volume that rate alone adds about 11,000 false fraud labels.
- `gpt-6-astra` equals the pipeline on fraud-versus-legit (precision .889, recall .471) and loses on the fraud type.
- Tables: `check/pilot_predictions_astra_full_eval.md`, `check/pilot_predictions_decisions_eval.md`.
- Status: both setups measurably worse than the pipeline on the pilot. The pilot is 525 rows; per-class intervals are wide.

## Mixing the LLM with the pipeline (`check/hybrid_eval.py`, `check/hybrid_eval.md`)
Three override families on top of the pipeline label, each at confidence thresholds 0, .5, .7, .9, .95, .99:
D = LLM adds a fraud label where the pipeline says legit; T = LLM changes the fraud type; V = LLM vetoes a fraud label.
- Decisions setup, pooled 525 rows: D corrects 0 and damages 14 (threshold 0) down to 0 and 1 (.95); T corrects 1 and
  damages 13; V corrects 0 and damages 12. Every threshold below .99 lowers macro-F1; at .99 nothing is overridden.
- 6 of the wrong overrides carry confidence >= .9, so confidence does not separate the useful row.
- `gpt-6-astra`: 0 rows where it is right and the pipeline wrong.
- Status: no override rule is supported. No hybrid file was built.

## Full test run
- Decisions API (`gpt-6-luna`), lean context, started 08:44:00 IST, ended 09:30:45 IST.
- Coverage: 260,582 of 260,582 rows with a valid LLM label; 869 valid requests, 0 invalid answers, 0 HTTP 429,
  1 HTTP 503 (not billed, retried by the runner). Cost $8.42.
- File: `scratch_llm/llm_direct.csv` (`id,label`, sample-submission order, no missing labels), SHA-256
  `08557cb8a39c7e6b259cbade65566c2de060544bd2ee9ebff513dddfb194b615`. Not in the repository, not uploaded.
- Against `subs/dd_final.csv`: same label on 246,868 rows (94.7%). The LLM file calls 14,906 rows fraud, the pipeline
  3,984; same fraud label on 2,096 rows.
- Expected public score from the pilot: about .25. The file is an LLM-only experiment, not a candidate.
- Not run at the end (session budget): `check/validate_final.py` and the late isolation re-check. Coverage, order and
  label checks above were made directly on the csv.

## Reproduce (cwd = `work`, key in `.env.txt` at the repo root as `OPENAI_API_KEY`)
```
python scratch_llm/run_llm.py build-pilot ; python scratch_llm/run_llm.py build-lean
python scratch_llm/run_llm.py full --api decisions --mode lean --batch 300 --conc 6 --tpm 1930000 --cpt 1.85 --qtok 164 --cap 12 --passes 6 --timeout 120
python scratch_llm/verify_full.py ; python scratch_llm/check/validate_final.py
```
The endpoint has no seed or temperature parameter, so a rerun need not return identical labels. Prompt files and
request logs contain raw data rows and are not in the repository; the two build commands regenerate the prompts
(static prompt hash `604a5ec7c68db6c9`).

## Limits
- One prompt design per setup, one pilot sample, no prompt tuning after the pilot was scored.
- A full `gpt-6-astra` run (estimated $1,800 to $2,300) was not made; its pilot gave 0 corrected rows of 525.
- Fine-tuning and retrieval of labelled neighbours were not tried.
