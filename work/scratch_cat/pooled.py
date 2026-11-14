"""Which stored levels are absent from LightGBM's category list in the cLGBcat boosters (rare levels are dropped by its bin
construction), with their fit-row / held-row counts. Reads no label except y_fit < 0 (dropped rows). Also the mean split / gain
summary over the eight boosters of each arm."""
import glob, json
import numpy as np
import polars as pl
from quick_eval import CAT_COLS

c = pl.col
for BM in ("N", "D"):
    for fi, vm in ((0, 8), (1, 9)):
        D = pl.read_parquet(f"scratch_gap/{BM}_f{fi}.parquet", columns=["is_test", "month", "y_fit"] + CAT_COLS)
        fit = D.filter(~c("is_test") & (c("y_fit") >= 0)); V = D.filter(c("is_test") & (c("month") == vm))
        for s in (0, 1):
            f = f"scratch_cat/models/{BM}_cLGBcat_f{fi}_s{s}_inspect.json"
            try: J = json.load(open(f))
            except FileNotFoundError: continue
            for k in CAT_COLS:
                kept = {v for v in J["cols"][k]["feature_info"]["values"] if v >= 0}
                for lv in sorted(set(fit[k].unique().to_list()) | set(V[k].unique().to_list())):
                    if int(lv) not in kept:
                        print(f"{BM} fold{fi} seed{s} {k}: level {int(lv)} not in the category list | fit rows (before legit downsample) {int((fit[k] == lv).sum())} of {fit.height}, held rows {int((V[k] == lv).sum())} of {V.height}")
for arm in ("cLGBnum", "cLGBcat"):
    for BM in ("N", "D"):
        J = [json.load(open(f)) for f in sorted(glob.glob(f"scratch_cat/models/{BM}_{arm}_f*_inspect.json"))]
        if J: print(f"{BM} {arm}: boosters {len(J)}, nine-column splits mean {np.mean([j['nine_cols_splits'] for j in J]):.0f} of {J[0]['splits_total']}, gain share mean "
                    f"{np.mean([j['nine_cols_gain_share'] for j in J]):.4f} (min {min(j['nine_cols_gain_share'] for j in J):.4f}, max {max(j['nine_cols_gain_share'] for j in J):.4f}), "
                    f"decision types on the nine columns {sorted({t for j in J for v in j['cols'].values() for t in v['decision_types']})}, pandas_categorical {[j['pandas_categorical'] for j in J][0]}")
