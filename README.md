# UPI Fraud Rings (ML Gradient Wars 2026): solution

This repo labels every October transaction with one of 12 classes (legit plus 11 fraud types). The metric is macro-F1.
One command rebuilds every upload from the raw CSVs:

| upload | file | public LB | what it is |
|---|---|---|---|
| #1 | `work/subs/h6_ppq.csv` | .5081 | h6 model (6 seeds) + post-rules |
| #3 | `work/subs/ts_judge_h6.csv` | **.5443** | h6_ppq + device-id zone override + bait rule |
| #4 | `work/subs/final1.csv` | pending | c2s6 model (h6 + core2 features) + zones/bait + velocity receiver propagation + mule/velocity zone DP |

## Setup

Windows with PowerShell, Python 3.11 and [uv](https://docs.astral.sh/uv/).

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -r requirements.txt   # versions pinned from the venv that made h6
```

Put the competition files in `upi-fraud-rings/`: `train.csv`, `test.csv`, `accounts.csv`, `merchants.csv` and `sample_submission.csv`.

## Run

```powershell
.\run_all.ps1            # full chain -> work/subs/repro_ppq.csv, ts_judge_repro.csv, repro_final1.csv
.\run_all.ps1 -H6Only    # stop after repro_ppq.csv (steps 1-5)
# options: -Tag NAME (default repro) -Threads N (default 10) -SkipPrep
```

Outputs are named after `-Tag`, so a run never overwrites the reference uploads. At the end of steps 5, 6 and 11 the script
compares each output byte-for-byte with `subs/h6_ppq.csv`, `subs/ts_judge_h6.csv` and `subs/final1.csv` and prints
`byte-identical` or `DIFFERS`.

The script runs these steps from `work/`, with `PYTHONUTF8=1` and `POLARS_MAX_THREADS=8`:

| step | command | output | time* |
|---|---|---|---|
| 1 | `python prep.py` | raw CSV -> `data/*.parquet` | <1 min |
| 2-3 | `python make_matrix.py feats_cache/matrix_repro.parquet` | module caches `feats_cache/<module>[__dm\|__dm2].parquet`, then the 1,824,642 x 194 matrix | ~50 s |
| 4a | `python train.py --modules flow,burst,takeover,scam,ring,pairs --matrix feats_cache/matrix_repro.parquet --relabel C --rounds 150 --seeds 6 --threads 10 --tag repro --mode val` | `oof/repro*` (OOF probs, biases) | ~6 min |
| 4b | the same command with `--mode full` | `preds/repro_test.npy`, `subs/repro.csv` | ~4 min |
| 5 | `python postproc.py repro --qr-null` | `subs/repro_ppq.csv` | ~1 min |

\*Measured on a 12-thread desktop CPU with 32 GB RAM; peak RAM is under 10 GB. No GPU is used.

Step 4b reads the biases and round count that step 4a saved under the same `--tag`, so run 4a first.
Set `$env:UPI_CACHE` to a folder to build the module caches there instead of in `work/feats_cache`.
`make_matrix.py --force` rebuilds the caches even when they are up to date.
The `scratch_*` folders hold the exploration scripts. The pipeline does not use them.

### Expected numbers

Validation: fit Jul, score Aug; fit Jul+Aug, score Sep. "Pooled" covers both folds.
The noisy labels are the given ones. The cleaned labels come from `clean_labels.clean`, an event-majority proxy.

| | pooled | Sep | cleaned pooled | cleaned Sep |
|---|---|---|---|---|
| train val, argmax | .4735 | .4800 | | |
| train val, bias-decoded | .4791 | .4851 | .4914 | .4976 |
| + postproc (printed `post:` line) | **.4793** | **.4855** | **.4916** | **.4979** |

The out-of-sample decode (`python oos.py TAG`) tunes the bias on one fold and scores it on the other.
It gives Aug->Sep .4837, Sep->Aug .4724 and mean .4780.
The submission has 260,582 rows: 256,843 legit and 3,739 fraud.
Verified on 13 Nov: `run_all.ps1` with fresh caches (`UPI_CACHE`, `--force`) rebuilt a matrix equal to the original in all 194 columns. It also gave OOF probs, test probs, `repro.csv` and `repro_ppq.csv` byte-identical to `h6`.
A second check ran `.\run_all.ps1` with no options in a clean copy that holds only the files a git clone would have, plus the raw CSVs.
Every output was again byte-identical to `h6`, and `oos.py repro` gave the same numbers.
The result does not depend on the thread count: 1-seed validation runs with `--threads 4` and `--threads 10` gave bit-identical OOF probs.

## Approach

**Data facts that shaped the design.**
- October (test) has about half the daily legit rows of Jul-Sep, but fraud events stay complete.
- `ip_risk_score` and `app_version` drift in October, so they are dropped. Account/merchant ages and weekday are dropped too.
- About 12% of fraud labels are flipped to legit and about 4% have the wrong type.
- About 63% of fraud rows are "camouflage": they cannot be told apart from legit rows (AUC 0.50).

**Features.** `quick_eval.base_features` gives 27 row features. Six modules in `work/feats/` add 163 graph and window features (190 in total). No module reads labels.

| module | catches | signature it encodes |
|---|---|---|
| `flow` | mule_account, layering_chain | P2P fan-in bursts of new-payee senders followed by onward transfers; hop chains A->B->C with each hop at 0.96-0.995x the previous |
| `burst` | velocity_bot, refund_abuse | about 40 integer payments from one device in about 8 min; one sender with 3-8 REFUND rows over days |
| `takeover` | sim_swap, account_takeover | sender drain sessions of 2-4 P2P rows, each about 0.7x the balance; flags for the SIM archetype (new SIM, home, night) and the device archetype (new device, away, PIN retries) |
| `scam` | phishing_collect, fake_cashback, qr_swap | collects just under round limits; Rs 1-10 bait followed by a large payment back; QR scans whose payee is a P2P account |
| `ring` | loan_app_extortion, merchant_collusion | evening payments from one victim to several new lenders; bursts of round amounts from a few accounts at small merchants |
| `pairs` | qr_swap / refund_abuse / merchant_collusion | per-merchant segments of round-amount rows; their size separates the three classes (about 3, 5 and 17 rows) |

**Density matching (`density.py`).** The window and graph features need train to look as sparse as October.
- Train rows are split into two halves the way October was sampled. Legit rows go to a random half; each fraud event goes whole into one half.
- Each module is built once per half, plus all test rows, and each row takes its features from its own half.
- Test rows take their features from half 0.
- burst, scam and ring also keep fraud bursts at small merchants whole (rows on one merchant within 24 h are linked: `mlink`, cache `__dm2`).
- `pairs` is built at full density, because halving splits the merchant bursts it measures.
- On Sep at October density this recovers most of the 0.02 macro-F1 lost by full-density features.

**Model (`train.py`).**
- LightGBM multiclass with lr 0.05, num_leaves 31, min_data_in_leaf 200, feature_fraction 0.5, bagging 0.8, lambda_l2 10.
- Fixed 150 rounds in validation; the full fit on all train rows uses 165 rounds. Six seeds are averaged.
- Legit rows are downsampled to 20% with weight 5.
- Strong L2 regularisation is the parameter that matters under label noise (+0.009 over l2=1).
- **Relabel C** changes the training target only. sim_swap and account_takeover labels are about 30% swapped within each drain archetype, so each takeover row is relabelled to its archetype's type (+0.006).

**Decoding for macro-F1.** Per-class log-probability biases are tuned on the pooled OOF by coordinate ascent: grid -4..6, step 0.1, with the score curve smoothed so the bias sits on a plateau. Test predictions are `argmax(log p + bias)`.

**Post-rules (`postproc.py`).** The biases are re-tuned on the OOF after the rules.
- Rows with the takeover archetype flags are forced to sim_swap (SIM archetype) or account_takeover (device archetype).
- The 46 November-dated test rows are never legit: they are spill-over from October fraud events.
- `--qr-null`: in October the QR fraudsters blank `qr_vpa_mismatch`, which is null instead of 1.
  - A QR payment to a P2P account with a null mismatch flag is set to qr_swap.
  - No train row shows this, so the rule cannot be validated.
  - On Sep, P2P-QR rows with mismatch=1 are 84% qr_swap.

**core2 features (c2s6 model, upload #4).** `density.halves` links fraud rows into events with a 48 h gap, so it splits
multi-day events in train: refund episodes (54% of refund rows see a smaller `burst__rs_n`), loan victims (28%) and
layering chains broken at a label-flipped hop (11%). October events are whole. `feats/core2.py` rebuilds those 28 event-size
columns of burst/ring/flow at full density (their legit background is tiny, so full density matches October) and
`make_matrix.py --core2` swaps them in for the halved copies (still 190 features). Gain is small: pooled .4801 vs .4793,
OOS Aug->Sep .4848 vs .4841. The decoded test labels differ from h6 on 15 rows.

**Device-id zone override (`ts_judge_post.py`, upload #3 and #4). This exploits a data-generator artefact, not fraud behaviour.**
- The generator gives each fraud event a fresh phone from a per-class sequential counter, in a fixed class order
  (mule, velocity, layering, sim, ATO, phishing, qr, refund, fake_cashback, loan, collusion). Fresh legit phones come after.
  Velocity bots use a fixed pool (DV0420000-0422999) that is reused every month.
- So the device number range tells the fraud type: train zones are DV0423000-0428454 and October's are DV0473632-0474551.
  A row on a zone device is the zone's type in 85% of Sep rows; most of the rest is legit label noise.
- `feats/ts_event.py` holds the train zone edges. They were fit once on Jul+Aug labels.
- The zone edges actually used are found without labels. A DP (`ts_event_post.dp_edges`) splits the id block into 11
  contiguous zones in class order.
  - It uses a length prior (Gaussian approximation of a Poisson count) scaled from the train layout.
  - Each zone device's decoded model predictions are weak evidence. Legit predictions give no evidence, and sim/ATO
    predictions are pooled.
  - A multi-row device is only allowed in the takeover zones.
  - It runs on the OOF predictions for train ids and on the test predictions for October ids.
  - On Sep it recovers the label-fit edges within 1-2 ids.
- Every row on a zone device then takes the zone's type. On test, zones plus the bait rule change 299 rows: about 200 legit -> zone type and
  73 ATO -> sim. Gain: Sep .4855 -> .5096 (cleaned .4979 -> .5224, OOS .4841 -> .5082). Public LB: .5081 -> .5443.
- `--mv` (art_zonemax) also pools mule and velocity predictions as evidence. h6 never decodes velocity-zone rows as
  velocity, which otherwise drags the mule|velocity edge. This changes 1 test row.

**Bait rule (`feats/ts_bait.py`, applied in `ts_judge_post.py`).**
- A bait event is a sender/receiver pair with a tiny integer P2P payment (Rs 1-10), plus every payment back.
- If the bait is never answered, the event is phishing (99% in train).
- If it is answered and there are 2 baits, or the payback comes after 3 h, or the payback is not to a new payee, it is
  fake_cashback (~100%).
- The rule only touches rows outside the zones that the model already decodes as phishing, fake_cashback or loan.
- It changes 2 test rows (+.0002).

**Velocity receiver propagation (`art_acct_post.py`, upload #4).**
- A row gets velocity_bot when all of these hold:
  - it is not on a zone device and it is decoded fraud;
  - its P2P receiver account also received velocity-zone rows (pool or zone) within +-24 h;
  - no other zone type is in that window.
- The rule was picked on Aug from a 180-config grid. Its precision on changed rows is 73/74 on Aug and 22/23 on Sep.
- Sep .5096 -> .5104, OOS .5082 -> .5090. It changes 13 test rows (mule/layering -> velocity).

**Assembling final1 (`combine.py`).** art_acct and art_zonemax were built and gated on top of ts_judge_h6. `combine.py`
copies the rows each rule changed (vs ts_judge_h6) onto ts_judge_c2s6, and asserts that the two rules touch disjoint rows.
final1 differs from ts_judge_c2s6 on 14 rows and from ts_judge_h6 on 26 rows.

**What did not work.**
- *Camouflage ceiling:* 63% of fraud rows cannot be found with this data. Multi-hop graph features, node embeddings, sender sequences and balance continuity all score AUC of about 0.50 on them. Even with every other row classified perfectly, cleaned Sep macro-F1 tops out near 0.57.
- *Stacking:* a second stage on entity-aggregated stage-1 probabilities gave no gain.
- *Pseudo-labelling on October:* no gain.
- *Event smoothing:* averaging probabilities over device, sender or pair segments gave at most +0.0007. It was dropped.
- *Other attempts:*
  - Relabel variants A, B, D and E were neutral or worse.
  - XGBoost, and blends of XGBoost and LightGBM on the same features, gave no out-of-sample gain.
  - Prior shift (EM) could not be validated.
  - Fixing features that encode month length was neutral.
