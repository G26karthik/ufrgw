# art_merch (13 Nov): merchant-side generator artefacts — NULL

Scratch: `work/scratch_art_merch/` (m1-m9 structure checks, p1/p2 zone-anchor propagation; logs `*.log`). No candidate sub written.

## Merchant table (merchants.csv, MR000000-MR029999, dense ids)
- Ids are not class-allocated. Per-1k-id share of fraud rows / legit rows is flat (0.5-1.9, noise) for every class, Jul-Sep and test (m2).
- onboarded_on (2012-10 .. 2026-06): uncorrelated with id (r=.005); year mix per class equals legit; nothing onboarded after
  Jun 2026, no class has new merchants (onb<120d .3-1% for all, m2/m3).
- Txn merchant_category == table category on 100% of merchant rows; device_city/home_city == merchant city ~1.5% for all
  classes (m3).
- Popularity is Zipf over a random permutation of ids (Spearman(legit count, id) = .001, within category -.004; m7). So id
  carries no popularity rank either.

## Merchant choice by class (behaviour, already modelled)
- Core velocity / refund / loan / collusion / qr rows pick merchants ~uniformly (median other-month merchant rows 16-129 vs 1206
  legit), camouflage rows pick by popularity like legit (m5). Captured by ring__rcv_n / burst / scam receiver counts.
- Category lifts: loan->lending x25, refund->electronics x3 / ecommerce x1.9, velocity->gaming/ecommerce x2.4 (m5). Known.

## Out-of-time merchant purity (fit Jul+Aug labels, score Sep)
- P(Sep row is class k | merchant had class-k rows in Jul+Aug) = base rate or BELOW for all 11 classes (m4, m6), e.g. loan on
  lending 0.030 vs 0.039 without history, refund 0.002 vs 0.014. Fraud merchants are redrawn per event; history only marks
  popular merchants (anti-predictive).
- P2P receiver history: x8-15 lift for mule/layering/sim/ATO/phishing/qr but precision <= .08 (mule pools; ring features).

## Sequence checks (m8)
- Consecutive fraud events (time order) have merchant-id gaps equal to shuffled (~8.5k). Zone-device number vs time
  rho ~0 (-.30..+.07) and vs merchant id ~0: the per-class device counter runs in generation order, not time, and does not
  index merchants.
- Sender account ids / opened_on by class: flat (m9). Receiver account ids by class: flat (m2).

## Zone-anchor propagation (p1 receiver, p2 sender), label-free edges from h6 OOF DP
- Non-zone Sep rows on the same merchant within 1h/6h/24h of a zone-device row: 98% legit for every anchor class (zone rows
  pay popular merchants = camouflage), so overriding loses ~all rows (W=1h: 5.6k linked, gain 19 rows / loss 5.4k).
- P2P receiver links (mule/layering) are pure (.90-.99 cleaned) but h6 already predicts them (h6=zone .95-1.00): 0 gain.
- Same-sender links: 58-334 rows, purity <.9 except sim P2P (h6 already right). Null.

## Verdict
No merchant-side artefact. Gate not reached (no candidate beats ts_judge_h6 Sep .5096 / cleaned .5224 / OOS .5082).
Test impact: 0 rows. The device counter is the only id artefact; merchants and accounts are drawn from fixed pools.
