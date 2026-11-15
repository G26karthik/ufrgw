"""Learned combination of the reference model H and the weighted model W (scratch_wt/DECLARED_learn.md), saved predictions only.
  GW<w>    global weight w/100, no fit
  CW<a|b|c> class-specific weights, penalty lambda = 1e-4 | 1e-3 | 1e-2 on the spread of the weights
  RG<a|b|c> row gate on 10 label-free columns, penalty lambda = 1e-4 | 1e-3 | 1e-2 on the gate coefficients
August files are cross-fitted over the two event-grouped halves of aug_dec.py; September files come from a fit on all August rows.
Writes scratch_gap/P<B>_<name>_{aug,sep}.npy (never overwrites) and scratch_wt/learn_<B>.json (fitted selectors, all-August fit).
usage (cwd=work, PYTHONPATH=.): python scratch_wt/learn.py D|N"""
import json, os, sys
import numpy as np
import polars as pl
import torch
import train as T
from common import load_all
from clean_labels import fraud_events

G = "scratch_gap"; c = pl.col; B = sys.argv[1]; K = T.K; LAM = {"a": 1e-4, "b": 1e-3, "c": 1e-2}; dev = "cuda" if torch.cuda.is_available() else "cpu"
ld = lambda n: np.load(f"{G}/{n}.npy").astype(np.float64)
CTX = ["log_amount", "device_age_days", "sim_age_hours", "login_to_txn_sec", "qr_vpa_mismatch"]


def save(name, kind, P):
    f = f"{G}/P{B}_{name}_{kind}.npy"
    if os.path.exists(f) and name[:2] in ("GW", "C4"): return  # no-fit files of the first run, identical on a rerun
    assert not os.path.exists(f), f
    assert np.isfinite(P).all() and abs(P.sum(1) - 1).max() < 1e-3; np.save(f, P.astype(np.float32))


def feats(H, W, kind, fi):
    """10 label-free gate columns for the held rows of one month, in the row order of the prediction files."""
    rows = pl.read_parquet(f"{G}/{B}_{kind}_rows.parquet", columns=["txn_id"])
    F = pl.read_parquet(f"{G}/{B}_f{fi}.parquet", columns=["txn_id", "is_test", "month"] + CTX).filter(c("is_test") & (c("month") == 8 + fi))
    j = rows.join(F, on="txn_id", how="left", maintain_order="left"); assert j.height == rows.height == len(H) and j["month"].null_count() == 0
    X = j.select(CTX).to_numpy().astype(np.float64); qr = np.isfinite(X[:, 4]).astype(np.float64)
    ctx = np.column_stack([X[:, 0], np.log1p(np.clip(X[:, 1:4], 0, None)), qr])
    fo = lambda P: np.log((1 - P[:, 0] + 1e-9) / (P[:, 0] + 1e-9)); ts = lambda P: P[:, 1:].max(1) / (1 - P[:, 0] + 1e-9)
    return np.column_stack([fo(H), fo(W), ts(H), ts(W), (H[:, 1:].argmax(1) == W[:, 1:].argmax(1)).astype(np.float64), ctx])


def fit(mode, lam, H, W, Z, y):
    """Returns predict(H, W, Z) -> renormalised mixture, and the fitted parameters."""
    mu, sd = np.nanmean(Z, 0), np.nanstd(Z, 0) + 1e-9; std = lambda Z: torch.tensor(np.nan_to_num((Z - mu) / sd), device=dev)
    rk = np.bincount(y, minlength=K).astype(np.float64) ** -0.5; r = rk[y] / rk[y].mean()
    Ht, Wt, Zt, yt, rt = torch.tensor(H, device=dev), torch.tensor(W, device=dev), std(Z), torch.tensor(y, device=dev), torch.tensor(r, device=dev)
    a = torch.full((K if mode == "cls" else 1,), float(np.log(.35 / .65)), dtype=torch.float64, device=dev, requires_grad=True)
    th = torch.zeros(Z.shape[1], dtype=torch.float64, device=dev, requires_grad=mode == "gate")
    b = torch.zeros(K - 1, dtype=torch.float64, device=dev, requires_grad=True); z0 = torch.zeros(1, dtype=torch.float64, device=dev)
    mix = lambda Ht, Wt, Zt: (lambda w: (1 - w) * Ht + w * Wt)(torch.sigmoid(a)[None, :] if mode == "cls" else torch.sigmoid(a + Zt @ th)[:, None])

    def loss():
        nll = torch.nn.functional.cross_entropy(torch.log(mix(Ht, Wt, Zt) + 1e-12) + torch.cat([z0, b]), yt, reduction="none")
        return (rt * nll).sum() / rt.sum() + lam * (((a - a.mean()) ** 2).sum() if mode == "cls" else (th ** 2).sum())
    opt = torch.optim.LBFGS([p for p in (a, th, b) if p.requires_grad], max_iter=500, line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-14)
    l0 = float(loss())

    def closure():
        opt.zero_grad(); l = loss(); l.backward(); return l
    opt.step(closure); l1 = float(loss()); assert np.isfinite(l1) and l1 <= l0 + 1e-12, (mode, lam, l0, l1)

    def predict(H, W, Z):
        # exported with the fitted offset and without the loss's class weight: natural-prior probabilities on one scale for
        # every fit (equal to H at weight 0), so a bias tuned on one fit's rows applies to another fit's rows
        with torch.no_grad(): m = (mix(torch.tensor(H, device=dev), torch.tensor(W, device=dev), std(Z)) * torch.exp(torch.cat([z0, b]))).cpu().numpy() / rk
        return m / m.sum(1, keepdims=True)
    info = dict(loss_start=l0, loss_end=l1, weight=torch.sigmoid(a).tolist(), theta=th.tolist(), offset=b.tolist())
    return predict, info


if B == "D": Ha, Hs, Wa, Ws = ld("PD_h6_aug"), ld("PD_h6_sep"), ld("PD_wSQ6_aug"), ld("PD_wSQ6_sep")
else:
    Ha, Hs = ((ld(f"PN_C_{k}") + ld(f"PN_A_Cb2_{k}")) / 2 for k in ("aug", "sep")); Wa, Ws = ((ld(f"PN_wSQ_{k}") + ld(f"PN_wSQb2_{k}")) / 2 for k in ("aug", "sep"))
    save("C4", "aug", Ha); save("C4", "sep", Hs)
Da = pl.read_parquet(f"{G}/{B}_aug_rows.parquet"); ya = Da["y_true"].to_numpy().astype(np.int64); n = len(ya); assert Ha.shape == Wa.shape == (n, K)
tx = load_all().filter(~c("is_test")); _, FE = fraud_events(tx, g=48 * 3600)  # the event-grouped halves of aug_dec.py, same generator
ev = Da.select("txn_id").join(FE.select("txn_id", "ev"), on="txn_id", how="left", maintain_order="left")["ev"].to_numpy().astype(float)
rng = np.random.default_rng(2026); evh = rng.integers(0, 2, int(np.nanmax(ev)) + 1)
half = np.where(np.isnan(ev), rng.integers(0, 2, n), evh[np.nan_to_num(ev).astype(np.int64)])
Za, Zs = feats(Ha, Wa, "aug", 0), feats(Hs, Ws, "sep", 1)
print(f"benchmark {B}: August rows {n}, halves {int((half == 0).sum())} / {int((half == 1).sum())}, September rows {len(Hs)}, device {dev}", flush=True)

for w in (10, 20, 35, 50, 65, 80, 100):
    save(f"GW{w}", "aug", (1 - w / 100) * Ha + w / 100 * Wa); save(f"GW{w}", "sep", (1 - w / 100) * Hs + w / 100 * Ws)
if B == "D": assert np.allclose(np.load(f"{G}/PD_GW35_aug.npy"), np.load(f"{G}/PD_SQ6b35_aug.npy"), atol=1e-6)  # the fixed blend is one grid point
out = {}
print("gate column means (August):", np.round(np.nanmean(Za, 0), 3).tolist(), "sd", np.round(np.nanstd(Za, 0), 3).tolist(), flush=True)
for mode, tag in (("cls", "CW"), ("gate", "RG")):
    for k, lam in LAM.items():
        P = np.zeros_like(Ha)
        for h in (0, 1):  # fit on one half, predict the other
            tr, te = half == h, half != h; pred, i2 = fit(mode, lam, Ha[tr], Wa[tr], Za[tr], ya[tr]); P[te] = pred(Ha[te], Wa[te], Za[te])
            print(f"  {tag + k} half {h} fit: weight {np.round(i2['weight'], 3).tolist()} theta {np.round(i2['theta'], 2).tolist()}", flush=True)
        pred, info = fit(mode, lam, Ha, Wa, Za, ya); save(tag + k, "aug", P); save(tag + k, "sep", pred(Hs, Ws, Zs)); out[tag + k] = dict(lam=lam, **info)
        print(f"{tag + k} lambda {lam:g}: loss {info['loss_start']:.6f} -> {info['loss_end']:.6f} | weight {np.round(info['weight'], 3).tolist()} | theta {np.round(info['theta'], 3).tolist()}", flush=True)
json.dump(out, open(f"scratch_wt/learn_{B}.json", "w"), indent=1)
