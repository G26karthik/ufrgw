# bc_layering: layering_chain class dive (13 Nov)

Scratch: `work/scratch_bc_layering/` (cwd=work, `PYTHONPATH=.`).
- `prep.py` builds `frame.parquet`: all rows plus flow feats, sc_final decode, judge OOF/test arrays from `scratch_fresh_errors/e1_*`.
- `a1`-`a8` are the analyses, `gate.py` is the validation, `ev.py` is the gain arithmetic. Each has a `.log`.

Rule: `work/bc_rules_layering.py` (`apply(sub_df, tx)`).
Candidate: `subs/bc_layering.csv` = `subs/sc_final.csv` + 10 rows, all layering_chain -> mule_account.

## (a) Test rows decoded layering in sc_final: 570
| source | rows | structure |
|---|---|---|
| model, P2P chain (hop component size >= 3) | 409 | chains |
| model, P2P non-chain | 121 | small fan-ins (see below) |
| layering zone override | 40 | 15 merchant, 25 P2P (12 in chains) |

- Oct has 95 chains of size >= 3 (431 rows), against 186-205 per train month. That is .51x, the same as the volume ratio.
- Every Oct chain row is decoded fraud.
- Sources of the 8 mule-decoded chain rows: zone override, plus v3 R3.

## (b) Oct structure: nothing new, one propagation gap
- **Chain completeness.** 0 Oct chain rows are decoded legit. Neighbour rows of chain accounts within 24 h (a6) are about 85% legit-labelled in every month. Chains have no missing hops or ends.
- **Size-2 hop pairs.** Oct has 32, against 56-68 per train month. About 88% are legit-labelled in train (a2). Null.
- **Fan-ins (a5).** Two generator kinds:
  - Layering fan-ins are small: about 4.3-5.5 rows, new-payee r10 amounts, and they never forward.
  - Mule fan-ins are large and do forward (a 1+ forward share of 183/188 at n >= 5).
  - The model separates them: Aug/Sep event decode lay 143/151 and mule 1526/1530.
  - Oct n >= 5: 42 mule, 12 lay. That is .5x of train. No Oct shift.
- **Non-chain layering labels** are r10 fan-in rows plus about 140 non-integer camouflage rows per month. The decoded-legit ones are camouflage singletons.
- **Mule forwards** (sender is a fan-in receiver, br about 1) are labelled mule and decoded mule. Fine.
- **No Oct-only or Oct-depleted layering cell.** Chain hop ratio and gaps are unchanged (flow.md).
- **Gap: chain-wide zone propagation.**
  - The generator labels a chain as one event: label purity .99 per chain.
  - Train chains with a zone-device row carry the zone type (a7, gate.py label-only, off-zone rows of zone-linked chains):

    | month | mule-zone chains: y == mule | layering-zone chains: y == lay |
    |---|---|---|
    | Jul | 6/8 | 63/71 |
    | Aug | 14/16 | 52/59 |
    | Sep | 4/6 | 71/77 |

    The misses are legit (noise), except 1 Sep other-G1 row. Mule labels never appear in layering-zone chains, and layering labels never appear in mule-zone chains.
  - Test has 4 chains with a mule-zone device: 473637, 473660, 473709 and 473731. All sit inside the mule zone [473632, 473749), below the disputed BAND.
  - 10 rows of those 4 chains are still decoded layering, because v3 R3 only reaches hops within 1 h of a direct account link.

## (c) Rule CZ (label-free)
Take hop components of size >= 3 (flow.py edge definition, whole data). If the zone-device rows, excluding BAND devices and with zone type in {mule, layering}, carry exactly one type, every off-zone chain row decoded in {mule, vel, lay} takes that type.
- Velocity is excluded as a source. The only train velocity-zone chain (Sep, TR_EDGES) is device 424155, 1 id below the lay edge, and it is labelled layering, so it is an edge error.

**Judge decode, Aug/Sep OOF (gate.py):**
- Not reachable by R3: Aug 8 rows, 7 correct; Sep 1 row, 1 correct.
- Reachable by R3: Aug 7/7, Sep 1/1.
- The increment over v3 is therefore 8/9 correct, and the miss is legit.

**Gate:**

| | Sep noisy | Sep cleaned | OOS |
|---|---|---|---|
| judge | .5096 | .5224 | .5082 |
| judge + CZ | .5097 | .5225 | .5083 |

The gate moves only 2 Sep rows. The evidence is mainly label purity.

**Expected test gain (ev.py):** T_test = Sep cleaned T x .511; P comes from sc_final (mule 794, lay 570); precision is Sep judge precision.
- Mule F1 is about .539 and layering F1 about .559.
- 10 rows changed: macro +.00075 at precision 1.0, +.00068 at .89 (train), +.00043 at .5. The rule stays positive at .5, because the layering-side FPs are removed either way.
- About 3 rows are expected to be public, so the LB will not resolve this.

## Verdict
- CZ: ACCEPT. Expected gain about +.0007.
- Everything else for layering is saturated. The +.004 target cannot be reached from this class.
- Disclosure: a8 printed row-level ids of the 4 test chains, but only to check that the source devices sit below BAND. The rule shape came from the a7 aggregates, and no condition was fit to those chains.
- Not acted on: a 3-row test fan-in whose velocity-zone device sits in the disputed mule|vel band. That is the R1 bet v3 declined.
