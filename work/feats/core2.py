"""core2: FULL-density copies of the event-size features of multi-day / chained fraud events (analysis: scratch_core2/a5.py, a6.py).

Why full density: density.halves links fraud rows into events with a 48 h gap on cleaned labels, so
  - refund episodes (3-8 REFUNDs over 4-9 days) are split: 54% of refund rows with an episode see a smaller burst__rs_n in the
    dm build than at full density;
  - loan victims (2-6 payments over 1-7 days) are split: 28% of loan rows see a smaller ring__s4_np_int_eve;
  - layering chains break at hops whose label was flipped to legit (clean() does not restore layering): 11% smaller chain_size.
Oct events are whole, so the dm build under-sizes these events in train/val only. The features count rows whose legit
background is tiny (REFUND rows of one sender, new-payee whole >=1000 rows of one sender, 0.9-1x P2P hop edges), so the
full-density build matches Oct (scratch_core2/a6.py). Same reasoning as `pairs` (built full density).
Label-free: it only calls the burst / ring / flow builders. Use with common.run_feature("core2"), NOT run_feature_matched.
"""
import polars as pl
from feats import burst, ring, flow

COLS = {
    burst: ["snd_ref3d", "snd_ref7d", "snd_ref14d", "dev_ref7d", "pair_ref14d", "rs_n_device", "rs_n", "rs_nrcv", "rs_dur_min",
            "rs_int_frac", "rs_small_frac", "rs_cat_frac"],
    ring: ["s4_np_int", "s4_np_int_eve", "s4_lend", "s4_np_int_nrcv", "s4_b11", "s10_np_int_eve", "s10_lend"],
    flow: ["hop_nr", "hop_ng", "hop_pr", "hop_pg", "chain_cspan", "chain_csin", "chain_crin", "chain_cpos", "chain_size"],
}


def build(tx: pl.DataFrame) -> pl.DataFrame:
    out = tx.select("txn_id")
    for mod, cols in COLS.items():
        name = mod.__name__.split(".")[-1]
        f = mod.build(tx).select("txn_id", *[pl.col(f"{name}__{c}").alias(f"core2__{name}_{c}") for c in cols])
        out = out.join(f, on="txn_id", how="left", maintain_order="left")
    return out
