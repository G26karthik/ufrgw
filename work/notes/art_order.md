# art_order: row order and id/value microstructure (13 Nov)

Scratch: `work/scratch_art_order/` (a1-a10 .py and .log). Verdict: **null**. No candidate sub was written. ts_judge_h6 stays the best.

| probe | result | verdict |
|---|---|---|
| Raw CSV row order (train and test, before any sort) | AUC(row idx, fraud) .5002; per month .4996-.5008. P(next fraud given fraud) .0367 vs base .0369. corr(row idx, ts) .0001. Test: AUC(idx, ts_judge fraud pred) .503. sample_submission order == test.csv order. | perfect shuffle, null |
| txn_id hex | AUC .5004, corr with ts -.0004, first/last 1-2 hex chars at null. 0 collisions in 1.82M ids (388 expected if iid 32-bit), so the ids were drawn without replacement or deduped. Not a bijective counter hash: low/high-bit histograms are multinomial (var ratio ~1.0, an affine map gives ~0), and inverting fmix32, lowbias32 and 9 multiplicative hashes gives uniform preimages. | null |
| Timestamp sec/min (and %2, %10) | type Cramer V .030-.035 vs null .032 | null |
| sim_age / device_age / ip decimals, login parity | V at null (.011-.017 vs .013) | null |
| amount / balance cents | Differ for model-caught fraud (round amounts: .00 in 89% vs 26% legit). Camouflage fraud (Sep, h6 decodes legit) matches legit: .00 share .263 vs .260, cents TV .031 vs null .030. | behaviour, already in the model; null for camouflage |
| app x app_version, city | Known behaviour signals (V .09 / .008) | already used |
| Account / merchant id numbers (sender, receiver, merchant) | Per-class deciles are uniform and match legit. opened_on is uncorrelated with account n. | no counter |
| Base-range foreign devices (device != sender's own, 75k rows) | fraud share .0336 = own-phone share .0343 | null |
| Legit fresh single-use phones (train 428455-473631, test >= 474552) | 21 fraud labels in 29.9k OOF rows. h6/ts_judge already predicts 100% legit there (Sep and test). Forcing legit changes nothing. | already exploited by the model |
| Within-zone device number vs first-use time | Spearman -0.18..+0.09 train: the counter does not tick in time order | no event-time info |
| Zone gap rate (unused ids/width) | Class signature that is stable train->test: sim .70/.70, ATO .26/.26, velocity .35/.33. Others are noisier (mule .53/.68, qr .34/.42). | could corroborate small-zone edges; only a few rows at stake |
| Split boundary (train has 147 Oct rows = Sep events spilling over, 12% legit = noise; test has 46 Nov rows) | Oct test rows linked to Nov rows within 48 h: sender 9 (all already same type), device 1 legit-pred row, receiver 4 legit-pred rows (shared refund merchants, not specific) | <=5 rows, not worth a sub |

Test impact: none. Every lever is either at the null or already used by h6/ts_judge.
