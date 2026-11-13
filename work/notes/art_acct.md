# art_acct: account identifiers and accounts.csv (13 Nov)

Scratch: `work/scratch_art_acct/` (a1-a14 analyses, `base.py` rebuilds ts_judge_h6 decoded arrays and asserts they equal
`subs/ts_judge_h6.csv`; cache `base_h6.npz`, grid `a14_grid.csv`). Deliverable: `work/art_acct_post.py` -> `subs/art_acct.csv`.

## Verdict: accounts carry no generator artefact. One small account-link rule passes the gate.

## Nulls (all checked on train labels; Sep where it matters)
- Account numbers: sender and receiver id quantiles are uniform 0..419999 for every class and role. There are no
  sequential ranges and no gaps. |sender - receiver| < 1000 at the chance rate (.4-.7%) for every class.
- accounts.csv: opened_on (q05/median/q95/max), kyc, os and bank_code shares are identical across all 12 classes, for both
  roles. All ids in tx exist in accounts/merchants. No self-transfers. No account outside 0..419999 in test.
- sender_account_age_days = round(timestamp - opened_on), a 12 h boundary. Fraud and legit match per hour. Class
  differences are only the hour mix (sim at night).
- Other-base devices (dn < 420000, dn != sender, ~5% of rows): same share, owner-uses rate (.76), pair-repeat rate (.95)
  and owner-is-fraud-account rate in every class. dn == receiver: never.
- Activity: never-send, never-receive, single-row and first-seen-this-month rates are flat across classes. A mule
  receiver has more inbound rows (median 6 vs 3). The model already captures this.
- Cross-month account label history (Jul+Aug labels -> Sep): it gives no lift over ts_judge_h6.
  - Among pred-legit rows, the fraud rate is .023-.025 for every history group.
  - Among pred-fraud rows where the history class differs from the model's class, history is right in 1-2% of rows and
    the model in 78-88%.
  - Accounts are redrawn at random for each event.
- Zone internals: inside every class zone, device number has no order by time or by sender (|Spearman| < .1, except fc and
  collusion with n < 80). The legit fresh-phone region (428455-473631 train, 474552+ Oct) is 99.93% legit. h6 already
  predicts legit on all 14,861 Sep rows and all 7,462 test rows there.
- Raw CSV row order is shuffled (sorted fraction .50). Tie order inside a timestamp is at chance.

## Candidate: velocity propagation through the receiver account (`art_acct_post.py`)
Rule: a row that is not on a zone device and is decoded fraud gets velocity_bot when two things hold:
- its P2P receiver also received velocity-zone rows (the pool or the velocity zone) within +-24 h;
- no other zone type is in that window.
How the rule was chosen and how it behaves:
- Grid in `a14.py`: links sn/rn/rid x W 10 min-7 d x cond x class, 180 configs. The rule ranked first on Aug.
- Precision on the changed rows: Aug 73/74, Sep 22/23. The model had called these rows mule or layering.
- Sender-link and 24 h same-sender propagation hurt: the victim's other rows are legit.

| | Sep noisy | Sep cleaned | OOS Aug->Sep |
|---|---|---|---|
| ts_judge_h6 | .5096 | .5224 | .5082 |
| art_acct | .5104 | .5232 | .5090 |

- Test: 13 rows change vs `subs/ts_judge_h6.csv` (10 mule->velocity, 3 layering->velocity). Sep at full density
  changes 23, so 13 fits Oct's half density.
- Expected test impact: about +.0007 macro, all of it on velocity F1. This is within noise for an LB read.
- Risk: low. The rule is label-free at prediction time, using test zones from ts_judge_h6's DP. It touches only fraud->fraud
  type. It could also be added as a model feature, but at this size it is not worth a retrain.
