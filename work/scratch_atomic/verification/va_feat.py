"""Build the new columns of one mechanism on every frame the harness needs, and prove they are label-free.
MODULE.py must define build(tx) -> DataFrame(txn_id + new columns); tx has the load_all() columns. `y` / `label` must not matter.
Frames (information convention):
  N_f0 / N_f1 : scratch_gap/bench_nat.world(tx, fit months, held month) = the convention of scratch_gap/nat_data.py: only the rows
                of the fit months and the held month exist (no later month, no October test row); held rows are visible but
                unlabelled; one build on that world at full density. fold 0 = Jul + Aug, fold 1 = Jul + Aug + Sep (+147 Oct-dated
                train rows).
  ALL         : load_all() = every train and test row (Jul..Oct). This is the production convention of scratch_gap/d_data.py
                (benchmark D = production matrix built once on all rows) and is also the all-rows deliverable (October test rows
                included). Limit: the production matrix builds window features on density-matched halves (density.py); a new
                column built here is at full density, so benchmark D does not density-stress the new column.
  DM          : the same all rows, built the production way of density.run_feature_matched without its cache: train rows are
                split into the two halves of density.halves(tx, mlink=86400) (label-informed split: fraud events whole, legit
                rows at random), the module is built once per half (that half + all test rows), every train row takes the
                columns of its own half and test rows those of half 0. The split uses labels (as production does); the column
                values do not: in the permutation check the halves stay frozen and the labels inside each half are permuted.
                This is the frame for window / count columns on benchmark D and for a production candidate.
Permutation check, every frame: the true labels (`y`, `label`) are restored on all rows of the frame and then permuted across
ALL rows of the frame (row set fixed); the rebuilt columns must equal the first build exactly (DataFrame.equals, and
bit-identical numpy buffers with NaN == NaN). The script aborts on any difference and writes no file for that frame.
Output: scratch_atomic/verification/feat/<NAME>_N_f0.parquet, <NAME>_N_f1.parquet, <NAME>_ALL.parquet, <NAME>_DM.parquet
usage (cwd=work, PYTHONPATH=.): python scratch_atomic/verification/va_feat.py path/to/MODULE.py NAME [N_f0,N_f1,ALL,DM]"""
import importlib.util, os, sys, time
import numpy as np
import polars as pl
import train as T
from common import load_all

sys.path.insert(0, "scratch_gap")
from bench_nat import world  # noqa: E402

c = pl.col; HERE = "scratch_atomic/verification"


def permuted(w, truth):
    """Frame w with the true labels of its rows shuffled across all its rows."""
    t = w.select("txn_id").join(truth, on="txn_id", how="left", maintain_order="left")
    sh = np.random.default_rng(1).permutation(w.height)
    return w.with_columns(t["y"].gather(sh).alias("y"), t["label"].gather(sh).alias("label"))


def same(a, b):
    if a.columns != b.columns or a.height != b.height or not a.equals(b): return False
    return all(np.array_equal(a[k].to_numpy(), b[k].to_numpy(), equal_nan=a[k].dtype.is_float()) for k in a.columns if k != "txn_id")


if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("mech", sys.argv[1]); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    name = sys.argv[2]; frames = (sys.argv[3] if len(sys.argv) > 3 else "N_f0,N_f1,ALL").split(",")
    tx = load_all(); truth = tx.select("txn_id", "y", "label"); os.makedirs(f"{HERE}/feat", exist_ok=True)
    for fr in frames:
        t = time.time(); w = tx if fr in ("ALL", "DM") else world(tx, *T.FOLDS[int(fr[-1])]); bld = mod.build
        if fr == "DM":  # production density convention: one build per label-informed half (+ all test rows), halves frozen
            from density import halves
            h = halves(tx, mlink=86400)  # the split of the burst / scam / ring modules in matrix_m5dm2 (make_matrix.py)

            def bld(fw):
                parts = []
                for k in (0, 1):
                    sub = fw.filter((h == k).fill_null(False) | fw["is_test"])
                    parts.append(mod.build(sub).join(sub.filter(~pl.col("is_test") | pl.lit(k == 0)).select("txn_id"), on="txn_id", how="semi"))
                return pl.concat(parts)
        F = bld(w); new = [k for k in F.columns if k != "txn_id"]
        assert F.height == w.height and F["txn_id"].n_unique() == w.height, "one row per transaction required"
        F = w.select("txn_id").join(F, on="txn_id", how="left", maintain_order="left")
        wp = permuted(w, truth); assert not wp["y"].equals(w["y"])
        F2 = wp.select("txn_id").join(bld(wp), on="txn_id", how="left", maintain_order="left")
        assert same(F, F2), f"{name} {fr}: columns changed when the labels were permuted -> NOT label-free"
        F.write_parquet(f"{HERE}/feat/{name}_{fr}.parquet")
        print(f"{name} {fr}: rows {w.height} (months {sorted(w['month'].unique().to_list())}, test rows {int(tx['is_test'].sum()) if fr in ('ALL', 'DM') else 0}),"
              f" columns {new}, label permutation over all {w.height} rows: bit-identical | "
              + ", ".join(f"{k} non-null {F[k].drop_nulls().len()} mean {F[k].cast(pl.Float64).fill_nan(None).mean():.5f}" for k in new) + f" ({time.time() - t:.0f}s)", flush=True)
