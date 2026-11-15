# Addendum for agents A, B, C: model-independent generator perspective (added by the user while you work)

This adds to your reasoning. It is not a restart: continue the current jobs and keep your deadlines.

## Stance
- Treat existing feature exclusions, event definitions (`clean_labels.fraud_events`), class templates and decoder rules as
  hypotheses, not facts.
- Do not use the current model's predictions or inferred device zones as ground-truth labels. Supplied labels of July,
  August, September are the only labels.
- Target question: can two transactions that look alike to our feature matrix be distinguished by an observable trace of
  different generation processes?
- Investigate joint relationships and event structure in the original data, especially among missed fraud on ordinary /
  home devices. Do not restrict discovery entirely to a cohort defined by the current model's confidence: also look at
  fraud-labelled rows as a whole against legit rows, independent of what the model said.

## For each proposed mechanism, write all six items
1. The hypothesized generation step (what the data generator did to produce this row or event).
2. The exact observable trace it would leave (a label-free predicate or statistic on original fields).
3. An alternative explanation that could produce the same trace.
4. A test of both explanations against matched legitimate examples.
5. Freeze on July / August, then check its predictions on September.
6. If it survives: a general feature or executable rule, and its incremental whole-month macro-F1 (agent C scores).

A finding must make predictions beyond the examples used to invent it. Matching a distribution or explaining a few
selected rows is insufficient.

## Reuse, do not repeat
- Read `work/notes/generator_experiment.md`, `fresh_columns.md`, `fresh_errors.md`, `camo.md`, `drift.md`, `monthsep.md`,
  `relabel.md` first and reuse their negative results where equivalent.
- No brute-forcing of random seeds and no generic ID scans, unless you first show evidence that the generator can be identified.
