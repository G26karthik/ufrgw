# Coverage of every original field in the 190-column vector and the decoder

Sources read: `common.py:25-44` (joins), `quick_eval.py:28-46` (27 base columns), `feats/{flow,burst,takeover,scam,ring,pairs}.py`
(163 module columns; assembled by `make_matrix.py:22-38`), decoder `postproc.py:26-41`, `ts_judge_post.py:28-41`,
`feats/ts_event.py:15-41`, `feats/ts_bait.py:14-33`. Modules `adapt`, `mnorm`, `core2`, `ts_victim` are NOT among the 190
columns of `scratch_gap/{N,D}_f*.parquet` (column list printed by `00_peek.py`); `ts_bait` and `ts_event` act in the decoder only.
Status words: preserved = raw value is a column; transformed = row-level function; aggregated = only window / group
statistics; dropped = no path to the vector or decoder.

## transactions (train.parquet / test.parquet)
| field | status | where (file:line) | what survives / what is lost |
|---|---|---|---|
| txn_id | dropped (join key) | common.py:54 | id number never used as a feature; `notes/art_order.md` tested order artefacts |
| timestamp | transformed + aggregated | quick_eval.py:33-34 (hour, minute_of_day); `ts` = epoch seconds common.py:40; ring.py:33 (hour >= 20), takeover.py:39 (hour <= 5) | lost: date, weekday (dropped on purpose, drift.md:110), second of minute, sub-second part (`dt.epoch("s")` truncates; `01_missed.py` columns `sec`, `usec` measure whether anything is there) |
| sender_id | aggregated | flow.py:33-36,73-77; burst.py:72-80,107-111; takeover.py:20-36; ring.py:42-57; scam.py:43,83,112-117; pairs.py:27-30 | identity lost; windows: 60 s-30 min, 2 h session, +-2/4/10 d, +-3/15 d. No whole-history sender profile (own typical amount, hour, payee set) |
| receiver_id | aggregated | flow.py:46-71; burst.py:77-101; ring.py:59-95; scam.py:72-123; pairs.py:25; takeover.py:43 | identity lost; P2P and merchant ids share one code space (flow.py:33). No receiver whole-history profile except ring.py:77-83 (`rcv_n`, `rcv_nsnd`, `rcv_np_int_eve`) |
| receiver_type | preserved | quick_eval.py:43,45 (`is_p2p`, code) | - |
| device_id | aggregated + decoder | burst.py:71,75,99-107 (60 s-30 min counts, 600 s sessions); takeover.py:28-35,66 (`dev_excl`, `dev_frac`, `n_other_dev`, `sess_ndev`); decoder zones ts_judge_post.py:28-34, ts_event.py:15-41 | device number enters only through the decoder zones (>= 423000 and pool 420000-422999). Lost in the vector: number of senders per device over the whole data, device's own city history |
| amount | preserved + transformed | quick_eval.py:35-38,41 (amount, log, cents, round100, round1000, amt_share_bal); flow.py:40 (multiple of 10); ring.py:30-32; scam.py:36-48; takeover.py:21-26 | Float64 -> Float32 (earlier audit: no material loss) |
| txn_type | preserved | quick_eval.py:45 | ordered code |
| merchant_category | preserved (code) + flags | quick_eval.py:45; burst.py:105,110; ring.py:34 | null -> "NA" code. P2P rows with a category (`adapt__qr_p2p_cat`, adapt.py:22) are not in the 190 columns as a flag, but null vs code keeps it recoverable with is_p2p |
| device_city | transformed | quick_eval.py:42,44 (`away_from_home`, `mer_same_city`); takeover.py:24,30-36 (`city_excl`, `sess_away`); scam.py:133 (`qrp2p_payee_local`) | city value lost; city change against the sender's previous row not represented (predicate `city_chg_vs_prev_row`) |
| sim_age_hours | preserved + aggregated | quick_eval.py:39; takeover.py:37,56-57 | - |
| device_age_days | preserved + aggregated | quick_eval.py:39; takeover.py:38,58-59 | - |
| sender_account_age_days | dropped | no reference in quick_eval.py / feats (grep) | removed for drift by design, drift.md:31,110 (cost -.001 Sep) |
| new_payee | preserved + aggregated | quick_eval.py:39; flow.py:39 (`fill_null(0)`; column has 0 nulls, so no missing/zero merge); takeover.py:22; ring.py:38; scam.py:71-75 | - |
| pin_retries | preserved + aggregated | quick_eval.py:39; takeover.py:64 | - |
| login_to_txn_sec | preserved + aggregated | quick_eval.py:40; takeover.py:65 | - |
| balance_before | preserved + transformed | quick_eval.py:40-41; takeover.py:21,40,53-55,72; ring.py:32 (`b11`) | balance continuity across the sender's rows only inside 2 h sessions (takeover.py:40,72); across sessions not represented (`bal_jump_rel` in `01_missed.py`) |
| app | preserved (code) | quick_eval.py:45 | app change against the sender's previous row not represented |
| app_version | dropped | quick_eval.py:30 (comment) | on purpose: Oct is 100 % 5.x (drift.md:20, fresh_columns.md:48) |
| ip_risk_score | dropped | quick_eval.py:30 (comment) | on purpose: Oct drift (drift.md:15); within-month rank judged unstable there |
| qr_vpa_mismatch | preserved (NaN / 0 / 1) + aggregated + decoder | quick_eval.py:40; scam.py:56,106-112; pairs.py:23 (`fill_null(0) != 1` only as a row filter); postproc.py:36 | null stays NaN in the vector (not merged with 0). `adapt__qr_mm` not in the 190 columns |
| label | target only | common.py:42; clean_labels.py (training target transform) | never a feature |

## accounts (joined twice: sender `snd_*`, P2P receiver `rcv_*`; common.py:30-35)
Join coverage: `rcv_*` null on 1,159,577 rows = merchant receivers; `mer_*` null on 665,065 rows = P2P receivers
(`00_peek.py`). `merchant_category` is null on 662,633 rows: 2,432 P2P rows carry a category (the QR-to-P2P pattern).
| field | status | where | note |
|---|---|---|---|
| account_id | join key only | common.py:30-35 | joined to sender_id and receiver_id; number equals the home device number (`DV0` + number) |
| snd_home_city | transformed | quick_eval.py:42; takeover.py:24 | only `device_city != home` |
| snd_bank_code | preserved (code) | quick_eval.py:45 | ordered code |
| snd_kyc_level | preserved (code) | quick_eval.py:45 | - |
| snd_opened_on | dropped | no reference | art_acct.md:11-13: identical across classes |
| snd_os | preserved (code) | quick_eval.py:45 | - |
| rcv_home_city | transformed | scam.py:133 (`qrp2p_payee_local`, QR-P2P rows only) | elsewhere dropped; scam.md:43 |
| rcv_bank_code | dropped | no reference | same-bank flag tested in camo.md:35 |
| rcv_kyc_level | preserved (code) | quick_eval.py:45 | "NA" = merchant receiver |
| rcv_opened_on | dropped | no reference | receiver account age: drift.md:30-31 (no signal, drifts) |
| rcv_os | dropped | no reference | camo.md:35 |

## merchants (joined on receiver_id; common.py:37-38)
| field | status | where | note |
|---|---|---|---|
| merchant_id | join key only | common.py:37-38 | joined to receiver_id |
| mer_category | preserved (code) | quick_eval.py:45 | mismatch with the row's `merchant_category` tested in fresh_columns.md:31 |
| mer_city | transformed | quick_eval.py:44 (`mer_same_city`) | NaN on P2P rows |
| mer_onboarded_on | dropped | no reference | art_merch.md:7 |

## Aggregations that merge distinct counterparties or sequences (candidates for `02_trace.py`)
- `takeover` session = sender rows with gaps <= 2 h (takeover.py:10,20); all receivers and devices of the session pooled.
- `burst` sessions = 600 s gap per device / per sender (burst.py:106-107); refund episodes 4 d gap (burst.py:111).
- `ring` receiver segments = 12 h silence over ALL rows of the receiver (ring.py:14,66-67): background rows of a busy
  receiver merge with burst rows; `pairs` re-segments only merchant rows with amount in {1000,2000,2500,5000,10000}
  and 24 h gap (pairs.py:17,22-25); every other row is NaN there.
- `flow` hop edges need ratio in [0.9, 1) and `ts2 > ts` within 6 h and middle account `rin_n3h <= 3` (flow.py:82-85).
- `scam` pair features are keyed on (sender, receiver) codes, 15 d or 3 d windows (scam.py:72-123).
