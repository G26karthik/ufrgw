# Review of camo (adversarial reviewer, 2026-11-13)

Scripts: `work/scratch_camorev/r1.py` (numbers, alignment, bias optimism) and `r2.py` (time clustering). Outputs: `r1.txt`, `r2.txt`.

## Verdict
The null result holds: there is no usable signal in the generic rows, and m5dm stays the best. One framing is wrong, though: scenario B, and the "public LB ≥0.75 means noise" diagnostic, contradict the problem statement.

## Reproduced (m5dm OOF, stored meta bias)
- Alignment is OK: `m5dm_ids` y equals the cached matrix y on every row. txn_id is unique. fold 0 = Aug (526,852 rows), fold 1 = Sep (509,996 rows).
- Noisy labels:
  - argmax: pooled .4533, Sep .4599.
  - Biased: pooled .4574, Sep .4650.
- Cleaned proxy A:
  - argmax: pooled .4657, Sep .4721.
  - Biased: pooled .4700, Sep .4772.
  - All match camo's numbers.
- Generic rows: 24,212 of 38,501 clean-fraud rows (.629). Per class the share is uniform, .58-.70.
- clean() only relabels legit to fraud: 495 rows, of roughly 5k hidden ones. So proxy A still holds about 4.5k fraud rows labelled legit.

## Issues
1. **Scenario B is invalid.** PROBLEM_STATEMENT says training noise is only "fraud never reported (labelled legit)" and "filed under the wrong type". There are no legit-to-fraud flips. So fraud-labelled generic rows are real fraud, and they will be present in the clean test ("Some fraud looks completely ordinary").
   - Expect public LB around 0.47-0.50 (A world, before Oct drift).
   - A public score near 0.5 does not "diagnose" anything.
   - The B columns (about .80) should be dropped from the decision.
2. **"Noisy bias is optimal in both worlds" is overstated.** tune_bias smooths its curve (smooth=5), so it is not the in-sample argmax. The A-tuned and B-tuned rows differ from it by 0.0001-0.0006, which is optimizer noise.
   - Correct conclusion: the bias choice does not matter between noisy and A.
3. **In-sample bias optimism (applies to the baseline, not to camo).** Bias tuned on Aug only and scored on Sep gives noisy .4627 and A .4749, against .4650 and .4772 in-sample. So the honest bias gain is about +.003, not +.005.
4. Minor: the G-vs-L LightGBM in a4 early-stops on the same Sep rows it reports AUC on. This is optimistic, but the result is still AUC .50, so the null stands.

## Extra lead checked (not in camo): time / segment clustering of G
G/(G+L) rate per day: dispersion vs binomial 0.64 (1.0 means random), correlation with the core-fraud rate -.02.
- Per day x sender home city: dispersion 1.00, correlation .01.
- Per day x bank: dispersion 1.02, correlation .04.
So generic rows are spread binomially and uniformly at random. There are no campaign days, and nothing to exploit through core-fraud intensity.
