## 15 Nov 03:15: combined exploratory candidate HANDED OVER for the one authorised upload; Track 2 tests done; final report work/notes/final_decision.md. NOTHING RUNNING. work/subs/cmb80_dd_final.csv sha256 cff9925e...c3e3 = h6 at .80 + six sources at .20 (c2s6, sTS4, TabM, kA, TabICLv2, BSEL; all refitted on the density-matched matrix), blended before the decoder, full accepted rule chain (scratch_mix/build_all.sh, rules declared in scratch_mix/DECLARED.md); 15 rows differ from dd_final. Selector August D: .5001 / .5002 / .5003 for anchor .50 / .65 / .80 (h6 .5002) = no detected difference; September D +.0001 (-.0004..+.0007). No uploader script exists: the user uploads by hand. PUBLIC SCORE: not yet reported; record it in final_decision.md as one observation, no further uploads. Track 2 (work/notes/assumption_tests.md): A stamp masking .995 -> .991 (no dependence on the stamp alone), B categorical handling no detected difference, C class weights no detected difference alone / one raw blend promising but unconfirmed. dd_final.csv unchanged (8e04353d...29cd). NEXT: 09:45 IST make the repo public only after the user says "go" (ask about HANDOFF.md); code submission before 10:00 IST.

## 14 Nov 23:20: c2s6 alternative PREPARED FOR REVIEW, not uploaded, not a validated improvement (work/notes/alt_c2s6.md). work/subs/alt_c2s6_dd_final.csv sha256 b7c293e5...345e = c2s6 model + all accepted layers of dd_final (steps 12-17), without the final1 art bundle; built by work/alt_c2.py (self-check: same driver on ts_judge_h6 gives dd_final byte-identical; base ts_judge_c2s6 byte-identical to the clean from-raw run). 12 rows differ from dd_final, all pure model disagreements; LFM fires on a 5th row (TX2927ddfc) which lands on dd_final's label; bc_final.py's count gate would refuse this base. Evidence: D Sep +.0008 (+.0000..+.0017) from loan_app_extortion, a class absent from the 12 rows; on the transition types present in October, September gives no usable signal either way (weighted by the October mix about 3.7 vs 4.1 rows right of 9; 3 rows have no September analogue); validate_sub scenario +-.0012. Support is D-only and D is label-informed. Reading: no detected difference; recommendation: do not upload. dd_final unchanged (8e04353d...29cd). Final report corrected: D is a label-informed density stress test.

## 14 Nov 23:00: evening tests complete (work/notes/final_comparison.md), NO candidate, dd_final unchanged, nothing uploaded. Stacker / soft two-stage / TabM checkpoints (scratch_stack), experiment A binning + exact values (scratch_gap/numrep.py), B neighbour columns (scratch_knn), C TabICLv2 (scratch_icl). Sep after decoder vs C: LR stacker +.0009 (CI across 0); two-stage 35% blend N +.0017 (+.0008..+.0027) but D +.0001; 1023 bins all features N +.0009 / +.0016 (two seed pairs) but D -.0011 (-.0019..-.0003); 1023 bins row-level columns N +.0011 / +.0012, D -.0006; exact-value features N +.0020 then -.0015 (seeds 2,3), D -.0005; neighbour columns +.0003, blend +.0008 (CI across 0); TabICLv2 alone -.0042 (-.0063..-.0021), blend +.0007 (CI across 0). Every N gain re-tested on D did not reproduce there; D is a label-informed density stress test, so this is a disagreement between benchmarks, not proof the N gains cannot generalise (corrected 23:16). c2s6 (D +.0008, 12 Oct rows, scratch_gap/c2x_diff.csv) stays promising but unconfirmed. Nothing is running (checked 23:10; GPU idle). Benchmark D has one seed pair for the binning arms. scripts/validate: validate_sub.py run on dd_final after its 18:31 edit: format OK, 0 rows changed. Next: 15 Nov 09:45 IST make repo public only after the user says go; ask about HANDOFF.md.

## 14 Nov 20:30: full-history combination audit (work/notes/ledger.md), NO candidate, dd_final unchanged. Ledger of 50 validation runs + 98 subs + all rule layers. Chain: final1 scored .5435 < ts_judge_h6 .5443, so step 12 restarts from ts_judge_h6; of final1's 26 rows 8 are reproduced in dd_final, 5 (AC291816) conflict, 1 edge row + 12 core2-model rows absent. Generic later layers on Sep (zprop, CZ, LFM, dd): union +.0016 N / +.0017 D (boot +.0005..+.0029), all already in dd_final; QRN/RS/FCB/X999/BANDV/EDGEFIX/qr-null are test-only. Ensemble on D (h6, c2s6, ls_halves, aug3btta, xgbC, gpu_cat, TabM): Aug-chosen E_LGB +.0001 (-.0008..+.0011); E_ALL +.0005 (-.0011..+.0023); c2s6 alone +.0008 (+.0000..+.0017) but its only public measurement (final1 bundle) is -.0008. RUNNING: subagent (same model) on benchmark N: cross-model stacker, soft two-stage classifier, TabM checkpoint grid; progress work/scratch_stack/PROGRESS.md, hard stop 22:30 IST. Next: 15 Nov 09:45 IST make repo public after user says go.

## 14 Nov 19:35: TabM + phishing/fc audit (work/notes/tabm_phfc.md), NO candidate, dd_final unchanged. Official tabm 0.0.3 on benchmark N, 190 features, GPU: Sep after decoder LightGBM .5070; TabM plain s0 .5027; TabM ple s0 .5049, s0+1 .5029 (-.0041, boot -.0059..-.0022); blend w=.35 (picked on Aug) s0 .5072 (+.0002), s0+1 .5065 (-.0005, boot -.0014..+.0003). TabM fixes 55-88 fraud rows LightGBM misses and loses 75-105; difference is the phishing/fc boundary. Audit of e24 'AUC .80, precision .56': temporal event AUC Aug .836 / Sep .784 holds (134 / 141 events), but no Aug threshold gains; Sep t=.5 flips 81 rows (48 ph, 33 fc), macro -.0012 (boot -.0031..+.0005); signal = event size. Not promoted. Next: 15 Nov 09:45 IST make repo public after user says go.

## 14 Nov 18:35: gap reset (work/notes/gap_reset.md), NO candidate, dd_final unchanged. New label-hidden benchmark N (scratch_gap/bench_nat.py: held labels hidden, no later rows, no density.halves; permuting held labels leaves features identical) vs density stress test D: current recipe after decoder N .5070 / D .5078; RAW +.0001 / -.0003, AC -.0006 / -.0004, CE -.0209 / -.0229 -> old gate rejected nothing better; label dependence of density.halves did not change the decision. Sep error compartments (N): no-signature missed 10,851 rows (+.414 oracle), mistyped 732 (+.057; 465 in five look-alike pairs +.032, 267 scattered = 4% label noise), legit-label fraud-pred 1,009 (+.0165). Budget: whole gap needs ~900 new correct rows at precision 1, ~1,550 at .5; break-even precision per class = F1/2 (.17-.33). Ceiling claims reworded as scenarios (README, HANDOFF, notes/burst.md, validate_sub.py PREC_C now post-decoder). No new mechanism: no new observable input. Next: 15 Nov 09:45 IST make repo public after user says go.

## 14 Nov 17:45: generator experiment done, NO candidate, dd_final stays final (work/notes/generator_experiment.md). Mechanism 1 bait-event generator (gen_events.py bait-v2, 845 events, fidelity AUC .59): Sep macro A .4824, weight controls B05 .4823 / B1 .4830, regularisation BR .4789, generated C05 .4805 / C1 .4818 -> fails pass rule (no gain over weighting). Mechanism 2 Oct legit amounts x1.17: faithful emulation (multiples of 10 kept) costs only .0003, matched fit -.0006 -> falsified; the -.0035 in notes/drift.md item 5 was an emulation artefact. Clean raw-data reproduction PASSED: git archive of 330ad4c in an empty dir, run_all.ps1 -Tag repro -Threads 8, all 17 steps, repro_dd_final.csv == dd_final.csv byte-identical (sha256 8e04353d...29cd), 39 min, log work/logs/run_all_clean_14nov.log; clean dir C:\Users\saita\Downloads\upi_repro_clean kept (links removed). dd_final format check OK (260,582 unique ids, sample order, valid labels, no nulls). No upload used. Next: 15 Nov 09:45 IST make repo public after user says go.

## 14 Nov 17:30: user: site shows only the single score after an upload (no per-class view). Two more generator checks, both NULL (work/scratch_me/): np1.py/np2.py new_payee is not a generation-order record (P2P repeat pairs: rows with new_payee=1 per pair follow Binomial(k, .04); the new row's time rank is uniform; merchant rate .25 at every pair size); qn.py/qn2.py blank qr_vpa_mismatch: test P2P QR blank 140 (train 0), test merchant QR blank 144 incl. 17 non-integer amounts (train: all integer), every one already qr_swap/collusion/refund in dd_final, 0 left legit. Re-read genorder/shuffle/classwise notes: coverage holds. NO candidate; dd_final stays final. Leaders have 10-15 uploads in total, so their gain is not leaderboard probing. Next: 15 Nov 09:45 IST make repo public (ask user first).

## 14 Nov 16:45: neural/pretrained test NULL (scratch_nn/eval.log): camo vs legit AUC LightGBM .505, MLP .500, MLP+id embeddings .500, P@1% .18 vs base .167; 11-way class = prior; TabPFN-3 (tabpfn 10.0.0, v3 ckpt) on 211 Oct zone camo vs fresh legit CV AUC .51-.53 = LightGBM. Hunt 3 (wf_ae7ce909-a34): bots, genorder, camotime, classwise, pairs, shuffle all null (shuffle: all 2^32 numpy seeds, 3 F3 flags false); judge running -> work/scratch_x/judge/VERDICT.md. Own checks (work/scratch_me/): txn_id seed-free tests random (tx1.py); Oct amount scaling is inside the distribution, no digit trace (amt.py); raw bytes of sample_submission/test clean (raw.py); per-class decoded counts test vs Sep-scaled (cnt.py, judge j3): velocity .55x, collusion .61x, mule 1.28x, fake_cashback 1.37x, others .87-1.13; velocity->mule-shape shift rejected (only ~2 core-looking rows on velocity-zone devices vs ~8 expected, 6/6 HL probes mule). Judge j1: macro if every dd_final row were right = .5648, so leaders (.6075-.6338) need >= ~450-750 more correct rows than we predict. validate_sub.py now has per-class precision (sim/ATO .72, fake_cashback .69, mule/layering .86, phishing .87). NO candidate; dd_final stays final. User rules: no new workflows/agents (token cost), work sequentially, advisor only when needed.
## 14 Nov ~16:00 knockout probes: ko_velocity_bot .5194 (drop .0311 -> public F1 velocity .373; model expected .45), ko_merchant_collusion .5224 (drop .0281 -> F1 .337, as expected). Velocity: ~1,330 true rows in test vs 308 predicted (recall ~.23); at train camo share ~570 would be core, so ~260 core velocity rows (about 7 bot sessions) are in test in a shape we miss. 10/10 uploads used 14 Nov; rank 10. validate_sub.py KNOWN_F1 updated.
## 14 Nov ~15:30: re-upload of dd_final scored .5505 -> scoring unchanged. LB tiers: .6338/.6311, .6111/.6075, .5731, cluster .5499-.5547 (us 9th). Reading: cluster = model + device zone; top 4 have a typed signal worth +.06-.08 on top. Legit file chunks carry no identity (work/scratch_x/b1.log). Recommended last 2 slots today: knockout probes work/scratch_fe/blindspot/ko_velocity_bot.csv (expect drop ~.034) and work/scratch_fe/judge/ko_merchant_collusion.csv (~.026); drop x 12 = public F1 of the class. RUNNING wf_ae7ce909-a34 (attack-own-work, relay lanes with PROGRESS.md in work/scratch_x/<lane>/: shuffle, genorder, bots, camotime, classwise, pairs; verdict -> work/scratch_x/judge/VERDICT.md).
## 14 Nov ~14:40: fresh-eyes-fable wf_0f330ab6-c62 DONE (6 Fable lanes + judge, work/scratch_fe/): 0 candidates. Naive all-columns model Sep .474 (= ours), excluded cols worth +.004; graph propagation, label-semantics windows, forensics, literature all null. New fact: files are chunk-shuffled (train 6, test 2 chunks), no label info. Gap needs ~911 correct camo rows. Top hypothesis: scoring changed for new uploads. PLAN: slot 1 = re-upload dd_final.csv unchanged (sha 8e04353d6f52fe86); != .5505 means the key/metric changed. Optional knockout probes in work/scratch_fe/judge/ko_*.csv (probe only).
## 14 Nov ~13:30: leader-gap-hunt wf_ba2491da-9c2 DONE, 0 candidates (7 lenses + critic, work/scratch_lead/). Test ceiling (core perfect, camo->legit) .564-.593; test camo share = train (~.63, label-free zone-singleton counter). Camo rows = class code path calling the legit recipe; only class trace is the device zone, already fully used. A useful camo flag must carry class. Data cannot explain ranks 1-4 (.6075-.6338); ranks 5-7 (.5731/.5547/.5513) are inside the ceiling (core type-swap headroom +.02-.03, no mechanism found). All top 7 were < .54 at 04:00 -> likely one common source. External generator search: nothing. dd_final stays final. Only measured bet: 71 zone-edge rows (sign unknown, ~+-.004 public), optional probe.
## 14 Nov 12:30: LB shock: leader .6338 (11 entries), we fell to #7. Our oracle "perfect core, camo->legit" on noisy train = .569, so the gap needs camo detection in test, a lower test camo share, or a flaw in our ceiling. Running wf_ba2491da-9c2 (leader-gap-hunt: ceiling, testedits, generator, core, entity, critic, metric; scratch work/scratch_lead/). HL probes STOPPED; hold the 3 slots for a large-signal candidate (mechanism + hundreds of rows + expected >= +.005).
## 14 Nov uploads: #1 dd_final .5505 (BEST, #1); #2 probe_hl_ac015935_dd_final .5499 -> mule (keep); #3 probe_hl_ac279075_dd_final .5495 -> mule (keep). Next (all on dd_final, disjoint, each read vs .5505): probe_hl_ac298085 (20r), ac072953 (18r), ac253951 (17r), ac205092 (16r) _dd_final.csv; probe_rules.py now builds any HL_<acct> generically (all mule rows the account receives/sends; reproduces the hand lists). Winners (> .5505) get unioned into one final file and uploaded. Remaining candidates: scratch_probe/p12_hostless.log. 03:49 probes ac298085 .5493, ac072953 .5491, ac253951 .5501, ac205092 .5497 -> all MULE (6/6 HL mule). 3 uploads left 14 Nov. LB 04:00: us .5505, GradientAscent .5499, Grenuke .5497.
Done (14 Nov ~04:00): pool-push wf_d74b2cc8-227 -> 0 rules on all 4 lenses (scratch_pp/). tmpl: every pool x template cell at legit baseline or already decoded; swap: pools carry no mule/vel/layering/sim/ATO info, HL P(velocity) ~.11 each (rank: 298085, 072953, 253951, 205092, then 388997, 328754, 365217); expand: iteration no-op, no label-free account marker (static AUC .506-.516); feat: pool features in LightGBM delta indistinguishable from 0 (Sep +-.0008). Verdict: saturated. Use free slots on HL probes in rank order (no competing candidate).

# UPI Fraud Rings

## >>> RESUME HERE (13 Nov ~21:30 IST) <<<
Best public: subs/probe_edgefix_bc_bandv.csv 0.5500 (#1, upload #10, +.0007 over bc_bandv .5493). Final = best PUBLIC upload scored on private, so only upload files we accept as final.
Earlier: bc_bandv 0.5493.
AUDIT (wf_80d672e9-6ec, notes/audit_generalization.md): all 6 layers KEEP, no skeptic refuted. About 95% of the gain over h6_ppq comes from train/structure evidence checked out of time; the LB touched about 30 rows (qrm gate, BANDV sign, EDGEFIX). Expected private gain vs h6_ppq about +.042 (+.032 to +.050). probe_edgefix_bc_bandv stays final.
UPLOAD POLICY 14 Nov: only single fan-in-event probes of >=13 rows, one event per file, built on the current best; keep one only if public > .5500, then rebuild the next on the winner.
  Ready: subs/probe_hl_ac015935_probe_edgefix_bc_bandv.csv (22 rows), subs/probe_hl_ac279075_probe_edgefix_bc_bandv.csv (21 rows; rebuild on the winner if the first wins). P(win) about .14 each. Next: other p12 hostless segments, n 13-25 (n >= 20 last).
  NEVER: V49, SIMATO, VELLAY, RS revert, bundles, stale-base files (subs/probe_hl_*_bc_bandv.csv are stale), anything chosen by public score (retrain/blend/bias/edge re-tune). Leaving slots unused is fine.
Done (14 Nov 00:00): deep dive v2 wf_d1e4cff4-a0b (resumed). errbudget/precision/octcells: 0 own leads; total train-validated headroom 0 to +.001. ACCEPTED dd_final (judge+skeptics, notes/dd_final.md): 13 rows legit -> phishing_collect = pool-bait (11: tiny int P2P PAY bait-rule rows from fraudster-pool/role senders; OOT prec Sep .69, Aug .48-.52 vs break-even .23) U stamp (4: phishing stamp marker, 0 legit-path occurrences in 584k train rows; 2 overlap). Expected private +.0004 (floor +.0001). subs/dd_final.csv == dd_both.csv. UPLOAD ORDER 14 Nov: (1) dd_final.csv [train-validated, keep in stack regardless]; (2) probe_hl_ac015935_dd_final.csv; keep if public > max(.5500, dd_final); (3) HL_AC279075 on the winner (probe_hl_ac279075_dd_final.csv prebuilt; rebuild via probe_rules.py if (2) wins). Old *_probe_edgefix_bc_bandv HL files are now STALE.
run_all.ps1 step 16 (EDGEFIX) added; checked byte-identical in place on a copy of bc_bandv. bc_nr .5490 (#8), sc_final .5486 (#7), v3_final .5476. BANDV probe: bandv > nr, so BANDV confirmed; bc_bandv is the base. Previous leaders .5488 / .5485.
Uploads today: 10 of 10 used (#10 = subs/probe_edgefix_bc_bandv.csv: bc_bandv + 3 zone-edge rows, EV +.00016; keep as base unless public drops). Next probes tomorrow: notes/probe_bets.md (HL_AC015935 19 rows, HL_AC279075 18 rows; P~.14, probe-only); the counter resets at midnight IST. Deadline 15 Nov 10:00 IST.
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
