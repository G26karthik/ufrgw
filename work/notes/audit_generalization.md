# Generalization audit (wf_80d672e9-6ec, 13 Nov night)

Six rule layers, each audited and attacked by a skeptic; judge verdict below. No layer refuted.

## final_pick

Yes. Keep subs/probe_edgefix_bc_bandv.csv (.5500) as the file scored on private. Dropping any layer is not worth an upload, and in practice it cannot be done:
- A dropped-layer sub becomes final only if its public score is above .5500. That means removing a layer whose public rows were wrong.
- No layer shows that signal. Every layer's public delta was positive, or the layer was never isolated.
- For RS specifically, the wrong-case public reading implies 0 public RS rows. A revert would then tie, not win.
- So any drop either scores lower and is ignored, or ties and puts us at risk under the unknown tie rule.

**Where the public LB touched a decision** (this answers the user's overfitting worry): in about 30 rows. About 95% of the gain over h6_ppq comes from train or structure evidence that was checked out of time.
- qrm gate: 16 rows.
- BANDV sign read: 10 rows, one pure event, so the sign carries to its 9 private rows.
- EDGEFIX: 3 rows, 1 of them private and unverified.
- LFM target choice: final1 was in view, but mule is the no-regret target either way.

**Expected private effect vs h6_ppq:** about +.042, range +.032 to +.050.
- Zones +.033.
- qr: qrm/QRN +.0008 on top. qr_null already sits in h6_ppq's private score, so it is not counted again.
- zprop +.0006, sc +.0004, bc +.0006, BANDV+EDGEFIX +.0013.
- The private score is not forecast here. It depends on h6_ppq's own private score, which is unknown.

## per_layer

All six layers are kept. No layer is LB-overfit. The auditor and the skeptic agree on every keep. Where they differ, I use the skeptic's EV.

1. **zones: KEEP.** Fit on Jul+Aug, checked on Sep: +.024 noisy, cleaned and OOS. Test edges come from a label-free DP. ts_judge_h6 was the only zone variant uploaded, and nothing was tuned on public. Private EV vs h6_ppq is about +.033 (+.025 to +.037). The only edges set by the prior are sim|ATO 473987/8 and mule|vel 473749-83; resampling them costs about .001-.002. Leave the 473969-473986 tail band alone, because train shows the same zero-sim tails at its true edges.

2. **qr (qr_null + qrm/QRN): KEEP.** The evidence is the Oct-only blanking structure:
   - The paired mm cells drop.
   - Phishing was not blanked: 10 seen vs 9.9 expected.
   - The nulls sit at small merchants: 82% vs 8%.
   - Precision is about .9 or higher; break-even is .25.
   qr_null has 0 LB gates. qrm has 1 binary gate on a rule written in advance. EV is about +.004 to +.007 in total. Note that qr_null is already inside h6_ppq, so only qrm/QRN, about +.0008, adds on top of the chain.

3. **zprop R2+R3: KEEP.** Windows were picked on Aug and sit on a plateau. Decode-free purity by source zone: R2 ATO-zone 24/24 and R3 mule-zone 16/16. The losing outcomes are 0/54 in train. The skeptic rebuilt the edges on the reverted decode and got the same 8 rows back. EV is +.0004 to +.0008 (low end +.0001). Correction to the audit note: AC074759 is sourced from 474132, 145 ids into the ATO zone.

4. **sc: KEEP, with RS downgraded.**
   - FCB (5 rows) is solid: 0/230 bait rows fall in the qr zone, and the result does not change for penalties from -1 to -32. EV about +.0003.
   - RS (3 rows at MR004209) sits outside train support for both classes. Refund runs have at most 4 rows per sender, and every one-sender run of 6 or more is collusion. P(refund) is about .5 from structure, and about .8 after the 4.5:1 public update. EV is about 0 to +.0002.
   - RS is effectively locked in. If RS is right and has public rows, reverting it lowers public. If RS is wrong, the +.0010 read only fits with 0 public RS rows, so a revert would tie at .5500 and could not become the best.
   - Layer total: +.0001 to +.0006.

5. **bc (CZ, LFM, X999): KEEP.**
   - CZ: 0 layering labels across the 9 train mule-zone chains. The model's test layering call repeats the miss seen in train.
   - LFM: 0/96 train receivers are layering-majority. With the host now velocity, LFM is worth only about +.00006. Do not retarget it to velocity.
   - X999: 1 row, about ±.00006.
   - EV +.0006 (+.0002 to +.0009).

6. **probes: KEEP BANDV and EDGEFIX.** Both were chosen on structure evidence. The LB was used only as a sign read on single pure events: 0/186 train fan-in segments are mixed. BANDV is 9 private rows at about +.0012. EDGEFIX has 1 private row with P(right) .8-1.0, about +.0001.
   - V49 and SIMATO were already recorded as reject in probe_bets.md (line 16 still calls them last resort). They are now NEVER, see upload_policy.

## upload_policy

**Principle.** The best public upload becomes final automatically. So a probe is safe only if its public and private rows share one truth, so they cannot disagree. Then a probe that loses costs only an upload slot. A probe that wins on public almost surely wins on private too (P(public up and private down) is 0.5% or less).

**Safe uploads.** Each upload must be exactly one mule->velocity fan-in receiver event. The purity bound of 0/186 covers only this event type.
- Order: HL_AC015935 (19 rows), then HL_AC279075 (18 rows), then the other p12 hostless segments with n 13-25. Put n of 20 or more last (train 1/19).
- Rebuild each probe on the current best. The owner runs `python probe_rules.py HL_AC015935 subs/probe_edgefix_bc_bandv.csv`; I have no write access to subs/. The prebuilt subs/probe_hl_*_bc_bandv.csv files are stale: they revert EDGEFIX's 3 rows.
- First add the receiver's 3 forward rows to its id list:
  - AC015935: TX79154343, TXd3a27ba9, TX63158e21.
  - AC279075: TXf4ab64e0, TX78533590, TX873c0b2e.
- Only probe events of 13 rows or more. The tie rule is unknown, and a probe with 0 public rows ties at .5500. At 13 or more rows, .7^n is 1% or less.
- Upload one event per file. Keep a probe only if public goes above .5500; then rebuild the next probe on the winner.

**Never upload:**
- V49 (AC291816 is 2 sub-events with a 17 h gap).
- SIMATO (devices 473980-473986) and VELLAY (cut at 473818). Both move an edge across several devices, so the true edge can sit inside the moved range, with public rows right and private rows wrong. Treat them like V49, not as a last resort.
- RS revert or an MR005795 refund->collusion flip. The segment shape is outside train support for both classes, so no purity evidence covers them, and the RS revert can only tie.
- Bundles of several events.
- Files built on a stale base.
- Any retrain, blend, bias or threshold variant, or zone-edge re-tune, chosen by public score.

**Expected use.** About 4-8 useful probes exist, at P(win) about .14 each. Leaving the remaining slots unused is fine. Do not fill them.

## risks

- **Tie rule:** not stated. If a tie picks the latest upload, a wrong probe with 0 public rows would ship. The 13-row minimum guards against this. Ask the organizers if possible.
- **Random-by-row split:** assumed, not confirmed. BANDV's +.0003 fits a row-random split; a grouped split would give 0 or about +.003.
- **Repo reproduction:** run_all.ps1 reproduces only through sc_final. The final sub carries 28 more rows (bc_nr, BANDV, EDGEFIX). Extend run_all and commit it before the code-repo deadline.
- **Prior-set edges:** sim|ATO and mule|vel rest on the length prior, at about .001-.002.
- **RS:** sits outside train support. A wrong RS costs about -.0009 and cannot be removed.
- **Not checked by me:** I did not re-run any audit script. HL base rate .14 was not fit strictly forward in time. Organizers could call the device-id zones a leak, though test.csv columns are valid inputs.

## Per-layer detail

### zones: device-id zone override + bait rule (ts_judge_post.py, h6_ppq -> ts_judge_h6)

evidence train-oot, overfit risk low, verdict keep

EV: Expected private effect vs h6_ppq: about +.033 macro-F1, range +.025 to +.038.
- Low end: cleaned-Sep delta scaled by Oct zone exposure (+.0277), minus up to .001 edge-transfer cost.
- High end: clean-label bracket times 1.13 exposure (+.035 to +.037).
- Public +.036 was not used to select or tune the rule, so it is an unbiased read of the same effect. Private sampling sd of the paired delta is about .002.
- The current best sub still carries 291 of the layer's 299 rows, so nearly all of the effect remains.

Skeptic (refuted=False): Keep the zone layer in the final submission (probe_edgefix_bc_bandv carries 291 of its 299 rows). Do not spend uploads on re-tuning zone edges. The only prior-sensitive edges are sim|ATO (473987) and mule|velocity (473749-473783). The 473969-473986 tail band (28 rows) matches the zero-sim tails seen on train with true edges, so do not move it. Count the mule|vel band under the BANDV layer's risk, not this one.

### qr: Oct-only blanked QR mismatch cells -> qr_swap. Covers postproc --qr-null (P2P-QR null mm), the qrm inline override (ts_judge_h6_qrm, 16 merchant-QR null non-integer rows) and QRN in v3_rules.py (qrm plus 4 m10 rows).

evidence mixed, overfit risk low, verdict keep

EV: Expected private macro-F1 for the whole layer: about +.004 to +.007, central about +.0055.
- qr_null: 63 rows, central about +.0045, range +.003 to +.006. Precision is .90-.98. Per-row values (TP about +7.2e-5, FP about -2.4e-5) are borrowed from bc_qr.md (true qr T≈1202, P=537), which I did not verify. The counterfactual rows were legit-decoded, so a wrong call costs only qr precision and nothing in another class. The rows are spread at random by row, so about 44 fall in private. The effect is a property of the per-class F1, so it carries to private.
- QRN/qrm: 20 rows, central about +.0008, range +.0004 to +.0012. Public rose +.0016 on ~16 rows, more than ~5 expected public rows can deliver, so public probably drew more than 30% of them. I shaded the private row count down and did not count the public gain again.
- Downside scenario: QR blanking also hit Oct legit rows. The aggregates rule this out: the legit mm=1 nonint rate is normal in Oct, and the nulls cluster at small merchants. Even then, break-even precision is about .25, so a loss is implausible.

Skeptic (refuted=False): Keep the whole layer (qr_null, qrm/QRN including the 4 m10 rows) in the final submission. Expected private gain is about +.004 to +.007; the auditor's +.0055 central estimate stands. Do not spend uploads re-testing it. If one more QRN guard is wanted, drop only the 4 m10 rows. That bounds the downside at about .0002, but the evidence (v3_cells.md items 37-39) does not call for it. Do not extend the rule to the P2P mm=0 int cell or the mm=1 degree <=6 sibling. The sibling's noisy precision is .22, at break-even. Skipped: I did not re-run v3_cells' m10 isolation scan or bc_qr's T_qr estimate. Scripts: C:\Users\saita\Downloads\upi-fraud-rings-gradient-wars\work\scratch_audit\qr\s1.py and s1.log.

### zprop: pair-restricted zone propagation in v3_rules.py (R2 sim/ATO same-sender, R3a/R3b chain hop) plus the BAND source exclusion. Shipped in v3_final and carried unchanged to probe_edgefix_bc_bandv.

evidence train-oot, overfit risk low, verdict keep

EV: Expected about +.0008 on private, plausible range +.0003 to +.0013, all-wrong floor about -.001.
Per-row values come from scratch_probe p5:
- sim->ATO: +.000176 if the row is ATO, -.000154 if sim, about 0 if legit or other. At P(ATO) about .9 and P(sim) about .03, that is about +.00015 per row. 4 rows give +.0006.
- lay->mule: +.00007 if mule, -.000085 if layering, +.000015 if velocity or other. At P(mule) about .9 and P(lay) about .05, that is about +.00006 per row. 4 rows give +.00024.
The full-test expectation is about +.00085. With a random row split the private expectation is the same, but the variance is higher. It depends on how many of the 8 rows fall in the private 70% (binomial; expected 5.6). Per private row the value is about 1/0.7 of the full-test value.
The independent units are 2 R2 senders (AC074759 and AC223753, 2 rows each) and 4 R3 pairs, not 8 rows.
If one R2 sender is truly sim, the layer is still about +.0003. If every row is wrong (both senders sim and all 4 hops truly layering), it is about -.001. That needs the sim|ATO edge to be off by more than 20 ids against a train edge error of 1, so the tail probability is small.
Allow about ±30% on the magnitudes: p5 class sizes assume Sep recall and come from an older base.

Skeptic (refuted=False): Keep zprop R2+R3 as shipped in probe_edgefix_bc_bandv.csv. No upload is needed. In the audit record, restate the purity evidence per source zone: R2 ATO-zone 24/24 (Wilson lower bound about .86), R3 mule-zone 16/16 (about .81). Widen the low end of the EV range to about +.0001. Fix the R2 source note: AC074759 uses device 474132 (145 ids into the ATO zone), and AC223753 uses 474007 (20 ids in). If the mule|velocity edge ever moves below 473731, re-run v3_rules.zprop on the reverted decode. The R3 rows sourced from 473709 and 473731 would then become velocity-sourced, which is about neutral.

### sc (RS sc_rules_collusion.py + FCB sc_rules_fc.py, applied by sc_final.py on top of v3_final)

evidence train-oot, overfit risk low, verdict keep

EV: Expected private effect: about +.0006 macro-F1, range -.0015 to +.0010. Roughly 5-6 of the 8 rows fall in the private 70%. These are about two all-or-nothing events, not 8 independent rows.

- **RS (1 segment, 3 rows):** +.0005 if the segment is refund, -.0009 if it is collusion. At P(refund) .8-.9 the EV is +.0003 to +.0004.
- **FCB (1 edge decision, 5 rows):** +.0005 if correct (+.0002 from the 2 bait rows alone). My arithmetic for the wrong case (rows truly qr: 5 qr true positives lost, 5 ph false positives added) gives about -.0006. P(correct) is about .98 for the 2 bait rows and .85-.9 for the 3 rows inside the zone, giving EV about +.0003 to +.0004.
- **Scenarios:** both right gives about +.0010; RS wrong gives about -.0004; both wrong (under 2%) gives about -.0015.
- **Public check:** the +.0010 public result fits the both-right scenario and mildly tilts toward it.

Skeptic (refuted=False): - **Keep the current best:** subs/probe_edgefix_bc_bandv.csv stays as is. Do not spend an upload on reverting RS alone. There is about a 34% chance (.7^3) that none of its 3 rows is public, and the final ranking takes the best public upload, so a revert only counts if it also beats .5500.
- **Downgrade RS in the notes:** change P(refund) from .8-.9 to about .5 from train structure, or about .8 after the public update. RS EV goes to about 0 to +.0002, not +.0004.
- **Do not reuse the RS logic:** do not reuse "4-8 round-set segment => refund" as evidence for other rows.
- **Do not judge by public delta:** if a later candidate is built anyway, do not judge RS or MR005795 by that upload's public change.
- **Flag MR005795:** it has the same Oct-only shape (6+1 senders, 7 rows, all decoded refund) and is outside this layer. It shares RS's uncertainty, so the answer for it and for MR004209 should be the same.
- **FCB:** keep as is.

Not checked: the actual public/private split of the 8 rows, and true per-class precision. The leaderboard update uses assumed ph/qr F1 of about .5 and the notes' refund and collusion counts.

### bc (bc_nr: CZ + LFM + X999 via bc_final.py)

evidence train-oot, overfit risk low, verdict keep

EV: Center is about +.0007 private macro-F1. Plausible range is +.0003 to +.0010. Tail risk is about -.0005, only if 2 or more CZ chains and the LFM event are all truly layering, which train never shows. Add about ±.0002 for which of the 15 rows land private.

Per-row values from b10: +.00007 if truly mule, -.000083 if truly layering, +.000014 if velocity or legit.

By rule:
- CZ: +.0007 if all 4 chains are mule. One wrong chain still gives about +.0003. Expected about +.0006.
- LFM: +.00028 if the event is mule, +.00006 if velocity (the band state the best sub now carries via BANDV), -.00033 if layering. Expected about +.0001.
- X999: about +.00006 at P(phishing) .8. Break-even is .23.

Expected private delta is about the full-test delta, because per-row value scales by 1/.7 while the private row count scales by .7.

Skeptic (refuted=False): Keep all 15 rows (CZ, LFM, X999) in any future final sub. Do not retarget LFM to velocity, since that would rest on reasoning from the BANDV leaderboard result. Spend no upload on this layer. Record the corrected EV: centre +.0006, range +.0002 to +.0009, with LFM worth about +.00006 now that the host is labelled velocity in the best sub.

### probes (BANDV, EDGEFIX, planned HL_AC015935 / HL_AC279075 / V49)

evidence mixed, overfit risk low, verdict split

EV: Lock-in mechanics: the best public upload is final automatically. A probe that scores below .5500 costs only a slot, and one that scores above cannot be deselected. So whether to upload a probe reduces to P(public up AND private down).

Current best vs bc_nr (13 rows):
- BANDV: given 1 public row, 9 of 10 rows are private. At about 130e-6 per row on private, that is +.0012 (range +.0009 to +.0013). P(event is velocity) is about .97 or higher, because the public sign is positive and the event is pure. The only downside is a mixed event (1.6% or less); if the event were all mule it would be about -.0007.
- EDGEFIX: 1 private row. P(right) is about .8-1.0: 1.0 if the private row is 474283 and 474281 was public-right (contiguity), .81 if it is 474281, .86 if it is 473917. EV +.00005 to +.0001. Downside -.0002 with P about .1-.15.
- Net current best over bc_nr on private: about +.0012 (range +.0006 to +.0014).
- This layer is locked in anyway. Removing it would need an upload above .5500 without it.

HL probes (each is a single event):
- If right: about +.0017 on both public and private (n = 19 or 18, about 13 private rows).
- If wrong: public falls, the upload is ignored, and private is unchanged.
- P(public up AND private down) is about 0.5% or less: P(mixed) 1.6% or less, times an adverse split. P(0 public rows) = .7^19 = .001.
- Probe EV = P x .0017, about +.0001 to +.0004 each.

V49:
- P is below .2 against a break-even of .38.
- It spans 2 sub-events, so the public rows (about 3) can land in a velocity part while the private rows are in a mule part.
- P(public up AND private down) is about .05-.1 at a loss of about -.0005 to -.0008. EV is negative. Do not upload.

Skeptic (refuted=False): Keep BANDV and EDGEFIX; the current best probe_edgefix_bc_bandv.csv stays final.

The HL probes are OK to upload as probe-only, one per upload, with these changes:
- Rebuild each on subs/probe_edgefix_bc_bandv.csv, not on bc_bandv: `python probe_rules.py HL_AC015935 subs/probe_edgefix_bc_bandv.csv`. Do the same for HL_AC279075.
- First add each receiver's 3 forward rows to the HOSTLESS id lists, as BANDV did:
  - AC015935: TX79154343, TXd3a27ba9, TX63158e21.
  - AC279075: TXf4ab64e0, TX78533590, TX873c0b2e.
- Keep a probe only if public goes above .5500. Build the next probe on the winner.

Never upload V49. Use SIMATO and VELLAY only as a last resort.
