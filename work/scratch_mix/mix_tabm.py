"""TabM `ple` at fixed checkpoints on the D tables (production density-matched matrix). Training loop, preprocessing,
model and optimiser as scratch_stack/n14/tabm_ck.py (Prep and predict imported from scratch_gap/tabm_run.py).
fold 0 -> aug, 1 -> sep, 2 (all train rows -> real test rows) -> oct. No label of a held row is read.
usage (cwd=work, PYTHONPATH=.): python scratch_mix/mix_tabm.py SEED FOLD EP,EP,... -> scratch_gap/PD_xTck<SEED>e<EP>_<kind>.npy"""
import os, sys, time
import numpy as np
import polars as pl
import torch
import torch.nn.functional as Fn

SEED, FOLD, GRID = int(sys.argv[1]), int(sys.argv[2]), [int(e) for e in sys.argv[3].split(",")]
sys.argv = [sys.argv[0], "ple", str(SEED), "D"]  # tabm_run reads its configuration from argv at import
from scratch_gap import tabm_run as R  # noqa: E402
from tabm import TabM  # noqa: E402
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins  # noqa: E402
from quick_eval import CAT_COLS, cat_dtypes  # noqa: E402

c = pl.col; dev = R.dev; K, LF, BS = R.K, R.LF, R.BS; kind = ("aug", "sep", "oct")[FOLD]
for e in GRID: assert not os.path.exists(f"scratch_gap/PD_xTck{SEED}e{e}_{kind}.npy"), e
D = pl.read_parquet(f"scratch_gap/D_f{FOLD}.parquet"); feats = [k for k in D.columns if k not in R.META]
num = [k for k in feats if k not in CAT_COLS]; cards = [len(cat_dtypes()[k].categories) for k in CAT_COLS]
rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
fit = D.filter(~c("is_test")); y = fit["y_fit"].to_numpy().astype(np.int64)
keep = ((y != 0) | (rng.random(len(y)) < LF)) & (y >= 0)
fit = fit.filter(pl.Series(keep)); y = y[keep]; w = np.where(y == 0, 1 / LF, 1.0).astype(np.float32)
Xn = fit.select(num).to_numpy().astype(np.float32); P = R.Prep(Xn, rng); Xn = P(Xn)
Xc = fit.select(CAT_COLS).to_numpy().astype(np.int64)
held = D.filter(c("is_test") & (c("month") == 8 + FOLD)) if FOLD < 2 else D.filter(c("is_test"))
if FOLD < 2: assert held["txn_id"].equals(pl.read_parquet(f"scratch_gap/D_{kind}_rows.parquet", columns=["txn_id"])["txn_id"])
else: held.select("txn_id").write_parquet(f"scratch_mix/oct_ids_tabm{SEED}.parquet")
Hn, Hc = P(held.select(num).to_numpy().astype(np.float32)), held.select(CAT_COLS).to_numpy().astype(np.int64)
del D
emb = PiecewiseLinearEmbeddings(compute_bins(torch.as_tensor(Xn), n_bins=48), d_embedding=16, activation=False, version="B")
model = TabM.make(n_num_features=Xn.shape[1], cat_cardinalities=cards, d_out=K, num_embeddings=emb, n_blocks=3, d_block=512, dropout=0.1).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=R.LR, weight_decay=R.WD); scaler = torch.amp.GradScaler("cuda")
tn, tc, ty, tw = (torch.as_tensor(a, device=dev) for a in (Xn, Xc, y, w)); n = len(y)
print(f"xTck{SEED} fold{FOLD}: fit rows {n}, held rows {len(Hn)}, numeric {Xn.shape[1]}, checkpoints {GRID}", flush=True)
for ep in range(1, max(GRID) + 1):
    model.train(); t = time.time(); perm = torch.randperm(n, device=dev); tot = 0.0
    for i in range(0, n, BS):
        b = perm[i:i + BS]
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(tn[b], tc[b])
        kk = lg.shape[1]
        loss = (Fn.cross_entropy(lg.float().flatten(0, 1), ty[b].repeat_interleave(kk), reduction="none") * tw[b].repeat_interleave(kk)).sum() / (tw[b].sum() * kk)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); tot += loss.item() * len(b)
    msg = f"  epoch {ep} train {tot / n:.4f} ({time.time() - t:.0f}s)"
    if ep in GRID:
        t = time.time(); p = R.predict(model, Hn, Hc); np.save(f"scratch_gap/PD_xTck{SEED}e{ep}_{kind}.npy", p); msg += f" saved ({time.time() - t:.0f}s predict)"
    print(msg, flush=True)
print("done", flush=True)
