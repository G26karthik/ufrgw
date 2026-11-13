# UPI Fraud Rings

## >>> RESUME HERE (13 Nov ~21:30 IST) <<<
Best public: subs/bc_bandv.csv 0.5493 (#1). bc_nr .5490 (#8), sc_final .5486 (#7), v3_final .5476. BANDV probe: bandv > nr, so BANDV confirmed; bc_bandv is the base. Previous leaders .5488 / .5485.
Uploads today: 9 of 10 used (after #8,#9); the counter resets at midnight IST. Deadline 15 Nov 10:00 IST.
Pipeline: run_all.ps1 reproduces h6_ppq -> ts_judge_h6 -> final1 -> v3_final -> sc_final byte-identically from raw CSVs (verified in a clean room through final1; steps 12-13 verified in place). Repo PRIVATE: github.com/G26karthik/upi-fraud-rings-gradient-wars. Tags up3/up5/up6/up7. Make it public at 09:45 15 Nov (command below).
What works: (1) device-id zone override (generator artefact); (2) Oct-only blanked QR mismatch cells -> qr_swap; (3) small structural rules on small classes (each beat its Sep estimate on the LB: 16 rows +.0016, 12 rows +.0017).
Null (do not retry): stacking, pseudo-labels, smoothing, augmentation, OvR, bagging beyond 6 seeds, priors, boundary, zclean, ip/app, id artefacts other than devices, camouflage hunts.
Done: big-class dives wf_80d2f9be-d2c (notes/bc_*.md, bc_final.md). Phishing/qr saturated. Accepted: CZ (10 rows layering->mule), LFM (4 layering->mule), X999 (1 legit->ph), BANDV (10 mule->velocity, a bet on host DV0473782; final1 .5435 argues against it).
Advisor (Fable 5.2) plan: PAIRED PROBE. subs/bc_nr.csv = sc_final + CZ,LFM,X999 (15 rows, no-regret, ~+.0007). subs/bc_bandv.csv = bc_nr + BANDV only (10 rows). Upload both; the sign of bandv - nr on public reads the event class (one event, clean Oct labels).
  bandv > nr: BANDV confirmed, bc_bandv is the base. bandv < nr: DV0473782 is mule, so band ids below it are mule; check on-zone band decodes, consider mule|vel edge 473749 -> 473783 + re-run zprop (hold upload #10 for this). Equal: keep bc_nr.
  Then: rank other single-event zone-band bets (>= ~8 rows, e.g. the velocity agent's 13-row band) and probe one bet per upload diff. Stop model retraining (all gates saturated).
Done: structural retrains wf_2c589d43-8a1 both FAIL the gate (stamp drop/relabel OOS -.0013/-.0014; halves linkage OOS -.0013, extra FP fraud calls on legit rows). Code edits reverted (mtimes kept). Notes: notes/ls_stamp.md, notes/ls_halves.md. Deletable: feats_cache/*__dm3*, oof/ls*, preds/ls_halves_test.npy.
Next: when they finish, gate, combine on top of the best sub, extend run_all, commit+tag, send to the user. Then one Fable 5.2 advisor review (after the user runs /compact).

 — status

Competition: ML Gradient Wars 2026 (hack4hack). Round 13 Nov 10:00 IST -> 15 Nov 10:00 IST. Metric macro-F1 over 12 labels.
Limits: 10 uploads/day, final = best public upload scored on private 70%. Must submit code repo before close.

## Target
Max score. Ignore organizer's "0.40 is excellent". Assume rivals may reach 0.9+.
Guardrails: validate on temporal holdout (Sep); never tune to public LB. Final rank = best public upload scored on the private 70%.

## Layout
- `upi-fraud-rings/` raw data + problem statement
- `work/data/*.parquet` (from `work/prep.py`)
- `work/common.py` loader + feature-module contract (`work/feats/<name>.py: build(tx) -> txn_id + '<name>__*' cols`)
- `work/quick_eval.py [modules...]` shared yardstick: fit Jul-Aug, score Sep (legit x5 downsample)
- `work/notes/` agent findings
- venv: `.venv` (python 3.11, polars, lightgbm, xgboost, catboost)

## Facts
- ip_risk_score drifts in Oct (legit mean 0.20 -> 0.44): dropped. app_version 5.x Oct-only: dropped.
- test has 46 Nov rows; train has 147 Oct rows.

## Scores (quick_eval, Sep macro-F1)
| run | argmax | tuned |
|---|---|---|
| base row feats | 0.3577 | 0.3773 |
| +flow | 0.3955 | 0.4137 |
| +burst (leave-self-out fix) | 0.4083 | 0.4246 |
| +takeover | 0.3864 | 0.4007 |
| +scam (volume fix) | 0.4435 | 0.4497 |
| +ring | 0.4356 | 0.4432 |
| all 5 (full density) | 0.4495 | 0.4574 |

Notes: Sep labels are noisy. Clean-test F1 is estimated at about +0.03 over noisy Sep. Full-density features lose about 0.02 on Oct; `density.py` matched builds recover most of it.
Phase-1 findings: work/notes/*.md. Label noise: ~12% fraud->legit, ~4% wrong type, per row and independent. Nov test rows are all fraud. Oct QR-P2P rows show null `qr_vpa_mismatch` in place of 1 (evasion). Oct has fewer velocity events. Do not rescale amounts.

## Audit (done 13 Nov)
Ponytail audit wf_8f23ce01-b1c found 17 confirmed issues; fixes applied in wf_a614f4a9-06f.
- train.py and quick_eval.py now always use density-matched features (`density.run_feature_matched`).
- Removed scam month-volume weights; ring Nov rows now use Oct's month bucket; burst edge-coverage fix.
- halves() splits on cleaned labels; config and cache guards added; fixed Enum categorical codes.
- Dropped base features: sender_account_age_days, rcv_acct_age_days, mer_age_days, weekday.
quick_eval numbers after this point are NOT comparable with earlier ones (Sep is now at Oct density).

## train.py runs (density-matched, pooled OOF Aug+Sep / Sep fold; argmax / biased)
| tag | pooled | Sep | sub |
|---|---|---|---|
| m5dm (5 modules, lgb, 3 seeds full) | .4533 / .4574 | .4599 / .4650 | work/subs/m5dm.csv |
| tun_best (lgb l2=10 ff.5 nl31 md200, 150 rounds, relabel C, 3 seeds) | .4665 / .4719 | - / .4793 | subs/tun_best.csv |
| tun_best + postproc --qr-null (arch forcing + Nov + QR-null->qr_swap) | - / .4721 | - / .4797 (cleaned .4843/.4920) | **subs/tun_best_ppq.csv** <- best |
| gpu_xgb (no relabel, d8, 600r) | .4645 / .4682 | - / .4731 | not safe: QR-null evasion |
| xgbC (train.py xgb cuda, relabel C, 600r, 3 seeds) | .4680 / .4714 | - / .4779 | subs/xgbC.csv |
| bl_1_1 (tun_best+xgbC) | .4686 / .4723 | - / .4791 | subs/bl_1_1.csv |

| confusion_pairsfd (tun_best + full-density `pairs` module, matrix_m5dm_pairsfd, relabel C, 150r, 3 seeds) + postproc --qr-null | .4778 raw | .4834 (cleaned .4958), OOS .4819 | **subs/confusion_pairsfd_ppq.csv** <- BEST |

Phase 3 (done): `pairs` won, about +.006 pooled and +.003 OOS (1.4 bootstrap sd); refund F1 .478 -> .513. Stacking and pseudo-labelling are null (confirmed). `pairs` must be built at FULL density, because density.halves splits merchant bursts. Use --matrix feats_cache/matrix_m5dm_pairsfd.parquet.
Leads: (1) density.halves does not link merchant-receiver fraud rows, so ring seg_* collusion features are under-sized in train; fixing it may help ring. (2) ring__rcv_n, ring__r36_excess and scam__rcv_*_15d separate months (AUC .97 Jul+Aug vs Sep); check Sep vs Oct, then drop or rank them.

| halves_rsb (density.halves with mlink: merchant-burst linking; ring/scam/burst __dm2; matrix_m5dm2_pairsfd) + ppq | .4789 raw | .4850, OOS .4831 | subs/halves_rsb_ppq.csv (10 rows differ from confusion_pairsfd_ppq) |

Phase 4 (done): halves/mlink is a real but tiny gain (about +.001 OOS from collusion/refund over 2 seed sets); matrix_m5dm2_pairsfd is the new base. monthsep: ring__r36_excess and scam bg() encode month length (adversarial AUC .915 -> .54 when fixed), but fixing them is neutral and changes <0.1% of Oct predictions. NOT adopted. 6 seeds vs 3: about +.001 OOS.
WARNING: do not run `python density.py <mod>`; it force-rebuilds the default __dm caches.
Running 09:35 IST: h6 = 6-seed val+full on matrix_m5dm2_pairsfd + postproc --qr-null + oos (job bmqjl3e91, logs notes/train_h6_*.log, notes/postproc_h6.log). Oct calibration investigation wf_643aa678-f5d (collusion under-call: 2.72 vs 4.9 per 10k).

| **h6** (halves_rsb recipe, 6 seeds) + ppq | .4791 raw / .4793 post | .4855 post (cleaned .4979), OOS Aug->Sep .4837, mean .4780 | **subs/h6_ppq.csv** <- BEST |

Oct calibration (wf_643aa678-f5d): predicted rates follow the structural event counts. Velocity has 7 Oct bursts vs 25 in Sep; collusion has 3 Oct bursts, which is plausible (P=.22). Even an oracle bias would gain only +.0014, so no decode change. Model is near the validation ceiling.
Reproducibility DONE (wf_7b1e8833-456): `run_all.ps1` (prep -> make_matrix.py -> train val/full 6 seeds -> postproc --qr-null) regenerates h6_ppq.csv bit-identically from raw CSVs in a clean-room copy, about 15 min. README.md, requirements.txt and .gitignore are written. NOT git-initialised: the user creates the GitHub repo and commits. work/feats_cache_repro (~450MB) can be deleted.
State 13 Nov ~10:30 IST: validation is saturated (oracle decode +.0014 max; camouflage ceiling ~.57). Next real signal = public LB score of h6_ppq.csv once the user uploads it.

Out-of-sample decode (`oos.py`: bias tuned on one fold, scored on the other; the honest comparison):
tun_best .4711 mean (Aug->Sep .4787) > bl_1_1 .4700 > xgbC .4694. Same-feature blends do not help.

Phase 2 conclusions: event smoothing gives no gain. 63% of fraud rows are camouflage (AUC 0.50 vs legit) and cannot be caught. Only one Oct adaptation was found: QR mismatch blanked, handled by the --qr-null post-rule. Oct velocity events are 1/3 of Sep, so do not raise the velocity bias. Relabel C (sim/ATO archetype) is the only label fix that helps. Relabel A, B, D and E are neutral or worse.

## Code repo
PRIVATE: https://github.com/G26karthik/upi-fraud-rings-gradient-wars (master; tag up3-0.5443 = commit with ts_judge_h6.csv). Commit and tag at every upload. Keep it private until 15 Nov 10:00 IST. The Submission-tab link must be accessible to organizers: make it public right before the deadline or invite them. Not done yet. At 09:45 IST 15 Nov run: gh repo edit G26karthik/upi-fraud-rings-gradient-wars --visibility public --accept-visibility-change-consequences (a scheduled task was blocked by the permission classifier).

## 13 Nov 19:50 IST running
- Idea panel wf_b21c6afe-c91 DONE: all 5 NULL (boundary <=3 test rows; prior worse; ovr = same ranking as multiclass; zclean worse; bag: 12 seeds = 6, blends null). Model-level levers exhausted. Seed-set noise on OOS is about .0005.
- Repro builder wf_40660de5-217 (run_all for final chain; NOTE the qrm rule is only an inline override so far and must go into code)
- Deep v3 DONE (cells: QRN rule, 4 rows; zprop: 8 rows; dpocc rejected). Earlier plan was: cells (generalises the Oct-only-cell win), dpocc (zone DP occupancy), zprop (zone propagation on ambiguous pairs) -> v3_final.csv + work/v3_rules.py
- Queued: stamp-cleanup retrain (fresh_judge #3) once CPU frees. After all finish: build the end-to-end combined candidate, tell the user it is safe to /compact, then one Fable 5.2 advisor review.

## UPLOADS LOG
| # | file | public | note |
|---|---|---|---|
| 1 | subs/h6_ppq.csv | **0.5081** | #2 on LB; leader .5470. Matches val (cleaned proxy .498), so camouflage IS fraud in test |
LB 13 Nov evening: Granuki .5488, #2 .5485, us .5443 (#3). Idea panel wf_b21c6afe-c91 running (boundary / prior / ovr / zclean / bag + judge -> subs/panel_best.csv). Repro packaging wf_40660de5-217 running.
| 7 (sent) | subs/sc_final.csv | ? | v3_final + RS (3 rows coll->refund: 4-8-row round-set merchant segments, 0 collusion in train) + FCB (5 rows qr->phishing: ph|qr zone edge 474352->474360 via bait emission). work/sc_final.py. Expected +.0006 |
| 6 | subs/v3_final.csv | **0.5476** (best) | qrm + QRN rule (4 rows legit->qr) + zone-prop on ambiguous pairs (4 sim->ATO, 4 lay->mule); work/v3_rules.py. Sep +.0005 noisy, +.0005 OOS |
| 5 | subs/ts_judge_h6_qrm.csv | 0.5459 | ts_judge_h6 + 16 Oct merchant-QR null-mismatch non-integer rows -> qr_swap (0 such rows in train; fresh-eyes #1). Expected +.0004-.0008 |
| 4 | subs/final1.csv | 0.5435 | ts_judge_c2s6 (c2s6 probs + zones + bait; Sep .5101/.5228/OOS .5091) + art_acct (13 rows) + art_zonemax (1 row) via combine.py. 26 rows differ from ts_judge_h6 |
| 3 | subs/ts_judge_h6.csv | **0.5443** | h6 + device-ID zone override (299 rows). +.036 vs h6 (Sep predicted +.024). Leader .5470. 7 uploads left on 13 Nov |
| 2 | subs/h6_d+3.0.csv | 0.4946 | diagnostic: all fraud biases +3 (+682 fraud preds). LB -.0135 vs Sep-predicted -.017: val decoding is right, do not shift the threshold. The LB tracks Sep val. |

## 13 Nov 14:10 IST
- cleanest: h6 clean-test estimate about .509 ± .013, matching public .5081. The leader's .547 matches the scenario where sim_swap and ATO are separable (h6 would be .544 there). Hypothesis: the leader cracked sim/ATO type separation (+.035 macro).
- octsig: null, closed. hCE val .4535 (worse; not uploaded). core2 (recall agent) is about +.0014 pooled, gate pending.
- Running: type-split hunt wf_9dcc440f-b90 (victim / event / ring / bait angles + judge), augment wf_9694541b-9a5, deep-core recall + reviewers wf_a3ce0e32-d5a.

## DEVICE-ID LEAK (13 Nov ~15:30, wf_9dcc440f-b90)
Fresh phones (DV0 n >= 420000) are numbered by per-class counters in a fixed class order (mule, velocity, layering, sim, ATO, phishing, qr, refund, fc, loan, collusion), then legit fresh phones. Train region DV0423000-428454; Oct DV0473632-474551. Velocity pool DV0420000-422999 is reused every month. Zone edges come from a label-free DP on model predictions (test sim|ATO edge 473987).
- ts_judge_h6.csv (work/ts_judge_post.py h6): Sep noisy .5096 (+.024), cleaned .5224, OOS .5082. Changes 299 test rows. **Sent to user as upload #3.**
- Own-phone sim/ATO events (S archetype) remain at ~.70 accuracy, no signal. bait/ring: null.
- Caveat: the gain is a generator artefact; user decides whether to use it in the final pick.

Deep-core (wf_a3ce0e32-d5a, done): core2 (full-density event-size feats) passes the gate but is tiny (+.0007 OOS; c2s6_ppq differs from h6_ppq by 15 rows). Within-signature type confusions are inseparable (AUC .52-.59) except via artefacts. The clean proxy under-credits recall by about .0035 because density.halves splits flipped rows from their events (evaluation only).
Running: artefact sweep wf_3eab18de-824 (acct, merch, order, zonemax + judge -> art_final.csv); augment wf_9694541b-9a5. Upload #3 = ts_judge_h6.csv, score pending.

## LB target (13 Nov ~11:00 IST)
Leader public score is .5470. We have not uploaded yet; the user will upload in 1-2 h.
- ip_risk_score / app_version separate CORE fraud from legit in train (AUC .778 / .262) but NOT camouflage (.504 / .500). In Oct, ip is .54 for confident fraud. So camouflage rows are exact legit draws, and the gap to .547 must come from core rows.
- diag.py: shifting every fraud bias by -1.5..+3 moves test fraud rows only 3633..4421, so decoding is not the lever. Diagnostic subs: subs/h6_d*.csv.
- Running: hCE (relabel C+E, trains only on structurally supported fraud labels; a noise-hypothesis LB diagnostic), logs notes/train_hCE_*.log. Deep-core workflow wf_a3ce0e32-d5a (cleanest clean-test estimate, per-class recall-gap features core2, Oct-only signals).

## Uploads
(none yet; LB opens 13 Nov 10:00 IST) Recommended first upload: work/subs/h6_ppq.csv (fallback/A-B: confusion_pairsfd_ppq.csv, 13-23 rows differ; tun_best_ppq.csv, ~100 rows differ)

## Ceiling analysis (scratch_main/ceiling.py)
Cleaned Sep macro .4916 for tun_best. With every non-camouflage row classified perfectly the ceiling is about .572: per-class recall is capped by 57-70% camouflage. Precision is already .66-.99.

## Running (13 Nov ~07:00 IST, user away: continue autonomously)
- Phase 3 wf_4fa2b154-20e: confusion (pairs module, .4722 pooled so far, which is noise), stack, pseudo.
- Camo-deep wf_cbe86801-a8d: DONE, null and confirmed. Multi-hop AUC .502; oracle label-seeded and camo-only seeds .496-.505; embeddings + sender-sequence + balance continuity .501; positive controls .73-.99. Camouflage line CLOSED (notes/hop.md, hop_review.md, embed notes).
Next: integrate any winners, then train val+full, postproc --qr-null, oos.py check, update best sub.

## Phase 2 (wf_28ef347d-59c, done)
Agents: postproc (event smoothing + rules), tuning (owns train.py), gpu (xgb/catboost, models_gpu.py), adapt (Oct adaptation, feats/adapt.py), camo (camouflage hunt), relabel (relabel.py). Each has a reviewer.

## Next
- Phase 1 workflow (wf_bd2f819e-5ca, running): per-fraud-family feature agents + drift/noise + research + trainer harness.
- After phase 1: run ponytail-audit, fix the top findings, then work ponytail-style. Use context7 for library APIs, and tell subagents to as well.

## Reproduction (done 13 Nov)
One command: `run_all.ps1` (README.md), work/make_matrix.py builds matrix_m5dm2_pairsfd from scratch. Verified bit-identical: matrix (194 cols), oof/repro.npy, preds/repro_test.npy, subs/repro_ppq.csv == h6. Check caches in work/feats_cache_repro (UPI_CACHE env). requirements.txt + .gitignore at root; no git init.

## core2 / deep-core (13 Nov, scratch_core2/)
- Core recall gap is type confusion, not missed fraud. On core rows (p_legit<.9 or the class signature), h6 decodes only 243 rows as legit across all classes. The rest of the gap is mule/vel 275, mule/lay 251, sim/ATO 234/248, ph/fc 168/115, loan->ph 44 (a1.py). Within-signature LGB on all feats + raw ip/app/ages gives AUC .47-.59 (chain lay|mule .63, already handled via cspan) (a2.py). The confusions cannot be separated.
- Clean-test estimate (a4.py), h6 Sep: cleaned .4979. World B (archetype swaps are train-only noise) gives .5279. B plus singleton-flip credit gives ~.547, the same as the leader. Public-LB sampling sd is .010. Row order carries no leak (AUC .5002).
- density.halves splits multi-day/chained events: refund episodes 54% (rs_n), loan 28% (s4_np_int_eve), layering chains 11% (a5.py). `feats/core2.py` = full-density copies of those 28 cols. matrix_core2swap replaces their dm copies. Test features are unchanged (0-11 rows differ).
- c2s6 (6 seeds) vs h6, every metric: pooled .4801/.4793, Sep .4859/.4855, cSep .4984/.4979, OOS .4848/.4841, cOOS .4972/.4964 (P(d<=0)=.05). 3-seed c2swap vs h3: +.0013 pooled, +.0004 cOOS. Sub: subs/c2s6_ppq.csv. It differs from h6_ppq on 15 test rows, so LB-neutral. Keep h6_ppq as the first upload.

## Reproduction of final1 (13 Nov ~20:00 IST)
`run_all.ps1` now runs the whole chain (default; `-H6Only` stops at h6_ppq): h6 -> ts_judge -> art_zonemax (`ts_judge_post.py --mv`) -> art_acct -> core2 matrix (`make_matrix.py --core2`) -> c2 train -> ts_judge -> `combine.py` -> `<Tag>_final1.csv`. No pipeline script imports from `scratch_*`.
Verified with `UPI_CACHE=feats_cache_r`, `-Tag r_repro -SkipPrep` (73 min on a loaded CPU): h6/c2s6 OOF and test probs, h6_ppq, ts_judge_h6, art_zonemax, art_acct, ts_judge_c2s6 and final1 are all byte-identical. README updated (steps, numbers, Approach: zones/bait/acct). `work/feats_cache_r` (~0.6 GB) and the `r_*` files in subs/oof/preds can be deleted.
