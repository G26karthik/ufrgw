import polars as pl, numpy as np
from common import load_all
tx = load_all()
print(tx.schema)
print(tx.height, tx.group_by("month","is_test").len().sort("month"))
print(tx.null_count().transpose(include_header=True).filter(pl.col("column_0")>0))
N = pl.read_parquet("scratch_gap/N_f1.parquet", n_rows=5)
print(N.columns)
for f in ("N_sep_rows","D_sep_rows","N_aug_rows","D_aug_rows"):
    d = pl.read_parquet(f"scratch_gap/{f}.parquet"); print(f, d.shape, d.columns[:8], d.columns[-4:])
for f in ("dec_N_wt","dec_D_sq6"):
    z = np.load(f"scratch_gap/{f}.npz"); print(f, {k: z[k].shape for k in z.files})
for f in ("PN_C_sep","PN_C_aug","PD_h6_sep","PD_h6_aug"):
    print(f, np.load(f"scratch_gap/{f}.npy").shape)
