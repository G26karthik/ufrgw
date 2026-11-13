# dd_final: judge verdict on the dd candidates (pool, stamp, both)

Base: `subs/probe_edgefix_bc_bandv.csv`. Output: `subs/dd_final.csv`, built by `work/dd_final.py`.

## Verdict

- **Accept pool and stamp, as their union.** That gives 13 rows, all changed from legit to phishing_collect.
- **dd_both is not a third rule.** It is exactly this union, and `dd_final.py` asserts that `diff(dd_both) == union`.
- **Upload dd_final.csv** (content-identical to dd_both.csv).
- **Do not stack** dd_pool, dd_stamp and dd_both, and do not add their gains.

## Overlap

- pool contributes 11 rows. stamp contributes 4 rows.
- The 2 overlapping rows are TX4a2a8f0c and TXbeb95064. Both rules set them to phishing_collect, so they agree.
- The union is 13 rows.

## Why accept every row

The break-even precision is 0.233: a correct row is worth +79e-6 and a wrong row costs -24e-6. Every subcell's train out-of-time point precision is above that break-even:

| Subcell | Rows | Train precision |
|---|---|---|
| P only, in B | 5 | .375 |
| P only, role-only | 4 | .632 |
| P and S (stamp A) | 2 | .571 |
| S only (stamp B) | 2 | .667, cell purity .83 |

A skeptic re-ran all three gates and found no leakage, no Oct-only quirk, and no overlap with the existing chain.

Expected gain is about **+.00039**. The floor is about +.0001 and the ceiling about +.0005.

## Weak rows, kept on purpose

**P-only in_B (5 rows).** Wilson90 is about .16, which is below break-even. The point estimate of .375 is above it, at about +14e-6 per row. Dropping these 5 rows gives up positive expected value.

**TX81feab4f.** This row reaches F only through an Oct pool-device row, and that path has n=0 in train.
- Its worst case is -24e-6.
- It follows the same pool-sender mechanism as the in_B cell.
- It is kept. To drop it, remove it from `union` in `dd_final.py`, which gives 12 rows at about +.00038.

## Not done

- No upload was made.
- No public-LB scores were used.
- The private split is untouched: the reserved accounts AC015935 and AC279075 are not in any changed row.
