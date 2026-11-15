# Shared context for the three forensic agents (written 15 Nov 2026, 07:38 IST)

## Goal of this round
Find observable information that the current representation or pipeline loses, especially among fraud-labelled rows
that the final pipeline predicts as legit. Neither a dataset ceiling nor the existence of recoverable signal is
established. Required outcome: executable findings and measured comparisons, not ideas. "We reached the dataset limit"
is not an acceptable conclusion; report exactly what was examined and what the tests found.

## Project
- Competition: 12-class macro-F1 on UPI transactions (class 0 = legit, 11 fraud classes). Train = July, August, September
  (labelled); test = October (no labels). Current file `work/subs/dd_final.csv`, public .5505.
- Two exploratory uploads scored .5484 (cmb80) and .5494 (sq35) on the public part. Aggregate observations only: do not
  infer any individual test label from them and do not use them for any choice.
- Repo root `C:\Users\saita\Downloads\upi-fraud-rings-gradient-wars`; all code runs with cwd = `work`.
- Python: `cd /c/Users/saita/Downloads/upi-fraud-rings-gradient-wars/work && PYTHONPATH=. PYTHONUTF8=1 POLARS_MAX_THREADS=4 ../.venv/Scripts/python.exe script.py`
  (Git Bash). Polars, numpy, lightgbm, torch are installed. Plain `date` prints IST. Bash cwd can drift: `cd` to the absolute path first.
- Raw CSVs: `data/` under the repo root (see `work/common.py`: `DATA`, `load_all()` returns every transaction with `is_test`).
  Original fields must be read from the CSVs / `load_all()`, not only from cached float matrices.
- Feature modules: `work/feats/*.py` (flow, pairs, takeover, ring, scam, burst, core2, ts_event, ts_bait, ts_victim, mnorm, adapt)
  and module notes `work/notes/{flow,pairs,takeover,ring,hop,embed,art_acct,scam,burst}.md` (hop, embed, art_acct were tested and
  not kept in production; read their notes before proposing anything similar). `work/train.py` = trainer (`T.LGB`, `T.K`,
  `T.LABELS`, `T.FOLDS`, `tune_bias`, `decode`, `macro_f1`), `postproc.py` + `ts_judge_post.py` = decoder.
- Earlier tests and their status: `work/notes/final_comparison.md`, `assumption_tests.md`, `fresh_columns.md`, `fresh_errors.md`,
  `confusion.md`, `audit_generalization.md`, `ledger.md`. A proposal must say how it differs from these.

## Benchmarks (held-out predictions already saved in `work/scratch_gap/`)
- N = labels of the held month hidden, features at full density. D = production density-matched matrix; D is a
  label-informed density stress test, not independent ground truth. Report N and D separately.
- Fold 0: fit July, held August. Fold 1: fit July + August, held September. Tables `scratch_gap/{N,D}_f{0,1}.parquet`
  (190 feature columns + `txn_id, month, is_test, y_fit`); builders `scratch_gap/nat_data.py`, `d_data.py`.
  Held rows and true labels: `scratch_gap/{N,D}_{aug,sep}_rows.parquet` (`y_true`; prediction files are in this row order).
- Reference probabilities: N `PN_C_{aug,sep}.npy` (LightGBM, seeds 0,1); D `PD_h6_{aug,sep}.npy` (6 seeds), `PD_A_C_{aug,sep}.npy` (seeds 0,1).
- Decoded September labels after the full decoder: `scratch_gap/dec_N_wt.npz` (`y`, `C_pre`, `C_q`), `scratch_gap/dec_D_sq6.npz`
  (`y`, `h6_pre`, `h6_q`); `_pre` = after bias only, `_q` = after forcing, zones and bait rule. A missed fraud row = `y > 0` and `_q == 0`.
- Tools: `scratch_gap/aug_dec.py N|D VAR,..` (August score after the decoder, event-grouped halves),
  `scratch_gap/bench_eval.py N|D REF,VAR,.. _suffix` (September, paired event bootstrap, writes `dec_<B>_suffix.npz`),
  `scratch_gap/dec_diff.py dec_file REF VARS` (rows changed / corrected / newly wrong / missed fraud recovered / new false
  positives), `scratch_wt/post.py table dec_file REF VARS` (per-class precision / recall / F1).
- Matched-baseline harness to copy: `scratch_wt/wt.py` (arm `wC` reproduces `PN_C` / `PD_A_C` bit for bit: same keep mask,
  seeds 0,1, `T.LGB`, 150 rounds, learning rate .05) and `scratch_gap/numrep.py` (arms that add or change columns).
  One fit costs about 85 s (fold 0) and 110 s (fold 1) per seed at 4 threads.
- Event ids for fraud rows: `clean_labels.fraud_events(tx, g=48 * 3600)` (used by `aug_dec.py`). Count independent events, not rows.

## Rules (all agents)
- Write ONLY inside your own directory (given in your brief). New prediction files go there too, not into `scratch_gap/`.
  Never write to `work/subs`, `work/oof`, `work/preds`, `work/feats_cache`. Do not edit any existing file outside your directory.
- No git commands that change state, no uploads, no network requests, no publishing. Do not stop processes you did not start.
- Feature construction must be label-free. No hand-labelling of test rows. September is reused development evidence
  (inspected more than 40 times), not a fresh holdout; freeze choices on July / August first.
- Shared machine: 16 logical cores, about 17 GB free RAM, other jobs are running. Use `POLARS_MAX_THREADS=4`, `num_threads=4`.
  Check free RAM before a large load. Agents A and B: no model fits beyond small diagnostics. Agent C owns the scored fits
  (at most two LightGBM fits in parallel).
- Status words for every result: supported / promising but unconfirmed / no detected difference / measurably worse /
  not evaluated adequately. A narrow test that fails does not show that its whole family fails. No score ceiling claims.
- Keep a running `FINDINGS.md` in your directory, updated as you go (agent C reads A's and B's files while they work).
  Every finding: exact predicate or feature (code), event counts, rate among matched legit controls, per-month numbers,
  the script that reproduces it, and how it differs from earlier tests.
- Clock: write the real time from `date`; never a time ahead of the clock.
