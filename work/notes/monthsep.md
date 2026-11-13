# monthsep: per-receiver month drift / hidden month tags (monthsep agent, 13 Nov)

Scratch: `work/scratch_monthsep/` (`adv.py` adversarial month classifier, `lattice.py`, `mk_matrix.py`, `boot.py`, logs `adv*.log`, `ev_*.txt`, `boot.txt`).

## Verdict: nothing passes the GATE. confusion_pairsfd stays best. No full run was made.

## Root cause of adversarial AUC .97-.99: a hidden month tag
- Adversarial AUC on matrix_m5dm_pairsfd: Sep (train) vs Oct+Nov (test) is .991 with a random split and .915 with a receiver-grouped split. Jul+Aug vs Sep: .983 / .920.
- `ring__r36_excess` alone carries it (56% of gain). Dropping it moves the grouped AUC from .915 to .592 (Sep vs Oct).
- Mechanism: r36_excess = r36_n / (1 + 3 * cnt_month / days_in_month). Quiet receivers have integer counts, so the values sit on a lattice that depends on month length.
  - 31.3% of Sep rows are exactly 1/(1+3/30) = .90909.
  - 30.6% of Jul, Aug and Oct rows are exactly 1/(1+3/31) = .91176.
  - It is a deterministic Sep-vs-other-months tag, not fraud signal.
- Second tag: `scam__rcv_in_15d` (and `rcv_newpayee_in_3d`). The bg() edge correction makes the values non-integer only in early Jul (32% of rows) and late Oct (32%). Integer everywhere else.
- After both are fixed (or dropped together with `scam__rcv_qr_mm1_frac_15d`), the AUC is .537-.539. The rest is QR-mismatch evasion, a real shift.
- Monthly medians do not move: univariate AUCs are all within .45-.55.

## Identity proxies
- `ring__rcv_n` and `ring__rcv_nsnd` are whole-window per-receiver totals, so they act as receiver identity. Grouped vs random split shows memorisation of about .07 AUC.
- Receivers seen in only one month: 11% of rows in Jul, Aug and Sep, against 8.9% in Oct test. That is no new-receiver blow-up in Oct.
- They are not month tags: grouped-split gain is .03.

## Experiments (train.py val, relabel C, 150 rounds, threads 3; ev.py + oos.py)
| tag | change | seeds | pooled | Sep | cPooled | cSep | OOS Aug->Sep | cOOS | oos mean |
|---|---|---|---|---|---|---|---|---|---|
| confusion_pairsfd | baseline | 3 | .4778 | .4834 | .4901 | .4958 | .4815 | .4938 | .4755 |
| ms_drop | --drop ring__r36_excess | 3 | .4779 | .4830 | .4902 | .4952 | .4818 | .4941 | .4756 |
| ms_mn | r36_excess -> mnorm__r36x (constant 30.4375-day divisor) | 3 | .4778 | .4836 | .4902 | .4960 | .4826 | .4950 | .4761 |
| ms_mnr | ms_mn + rounded bg() counts | 3 | .4782 | .4838 | .4905 | .4963 | .4825 | .4948 | .4762 |
| ms_base6 | baseline | 6 | .4778 | .4835 | .4902 | .4960 | .4828 | .4951 | .4766 |
| ms_mnr6 | ms_mnr | 6 | .4779 | .4835 | .4903 | .4959 | .4825 | .4949 | .4767 |
(raw rows; post rows move all tags by the same +.0002-.0004.)

- At 3 seeds, ms_mnr looked like +.001 OOS. At 6 seeds this vanishes: -.0003 OOS Aug->Sep and +.0001 pooled. The 3-seed gain was seed noise.
- The fix is neutral on Sep, as expected: Sep is the one 30-day month, and the tag barely changes tree splits.
- The tag cannot hurt Oct directly, because Oct sits on the same 31-day lattice as Jul and Aug. It could only make Oct follow Jul/Aug-specific patterns. That is not measurable here.
- Optional robustness swap (no measured gain): `--matrix feats_cache/matrix_m5dm_pairsfd_mnr.parquet --modules ...,mnorm`.

## Files (new only)
- `feats/mnorm.py`: dm build gives `feats_cache/mnorm__dm.parquet`. A self-check reproduces ring__r36_excess from its parts (max err < 1e-5).
- `feats_cache/matrix_m5dm_pairsfd_mn.parquet`, `feats_cache/matrix_m5dm_pairsfd_mnr.parquet`.
- oof: `ms_drop`, `ms_mn`, `ms_mnr`, `ms_base6`, `ms_mnr6`.
