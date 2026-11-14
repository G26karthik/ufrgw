"""Declared mixtures of the combined submission (scratch_mix/DECLARED.md): anchor weight a, the rest shared equally by the
alternative sources, probabilities blended before the decoder.
  assess  : write PD_MIX<a>_{aug,sep}.npy (D-trained sources, the shipped recipe) and PN_MIXN<a>_{aug,sep}.npy (N-trained
            versions, side reading; no c2s6 there). Score them with aug_dec.py / bench_eval.py.
  build A : write oof/cmb<A>.npy (+ _ids.parquet, rows in oof/h6_ids order) and preds/cmb<A>_test.npy (sample order) for
            postproc.py / ts_judge_post.py. Never overwrites.
A source with a missing file is dropped from every mixture (printed). usage (cwd=work, PYTHONPATH=.):
  python scratch_mix/mix_blend.py assess | python scratch_mix/mix_blend.py build 65"""
import os, re, shutil, sys
import numpy as np
import polars as pl
import train as T

G = "scratch_gap"; c = pl.col; WEIGHTS = (50, 65, 80)
EP = {int(m.group(1)): int(m.group(2)) for m in (re.search(r"seed (\d) epoch (\d+)", l) for l in open("scratch_mix/tabm_pick.log")) if m} if os.path.exists("scratch_mix/tabm_pick.log") else {}
D_SRC = {"c2s6": ["PD_c2s6"], "sTS4": ["PD_sTS", "PD_sTSb2"], "TabM": [f"PD_xTck{s}e{EP.get(s, 0)}" for s in (0, 1)], "kA": ["PD_xKA"], "ICL": ["PD_xICL"], "BSEL": ["PD_A_BSEL"]}
D_OCT = {"c2s6": None, "sTS4": ["PD_xSTS"], "TabM": D_SRC["TabM"], "kA": ["PD_xKA"], "ICL": ["PD_xICL"], "BSEL": ["PD_xBSEL"]}
N_SRC = {"sTS4": ["PN_sTS", "PN_sTSb2"], "TabM": ["PN_sTck0e30", "PN_sTck1e12"], "kA": ["PN_kA"], "ICL": ["PN_iICLs0"], "BSEL": ["PN_A_BSEL", "PN_A_BSELb2"]}


def mean(files, kind):
    return np.mean([np.load(f"{G}/{f}_{kind}.npy").astype(np.float64) for f in files], axis=0)


def have(files, kinds):
    return all(os.path.exists(f"{G}/{f}_{k}.npy") for f in files for k in kinds)


def mix(anchor, alts, a):
    P = a * anchor + (1 - a) / len(alts) * sum(alts)
    assert P.shape == anchor.shape and abs(P.sum(1) - 1).max() < 1e-3 and np.isfinite(P).all()
    return P.astype(np.float32)


def d_sources():
    ok = [k for k, f in D_SRC.items() if have(f, ("aug", "sep")) and (D_OCT[k] is None or have(D_OCT[k], ("oct",)))]
    print("D sources in the pool:", ok, "| dropped:", [k for k in D_SRC if k not in ok], "| TabM epochs", EP, flush=True)
    return ok


def oct_pred(name, files):
    """October predictions of one source in sample_submission order."""
    tid = pl.read_csv(T.SAMPLE).select(c("id").alias("txn_id"))
    if files is None: return np.load(f"preds/{name}_test.npy").astype(np.float64)
    ids = pl.read_parquet(f"{G}/D_f2.parquet", columns=["txn_id", "is_test"]).filter(c("is_test")).select("txn_id").with_row_index("i")
    j = tid.join(ids, on="txn_id", how="left", maintain_order="left"); assert j["i"].null_count() == 0 and j["i"].n_unique() == tid.height
    return mean(files, "oct")[j["i"].to_numpy()]


if sys.argv[1] == "assess":
    ok = d_sources()
    for kind in ("aug", "sep"):
        A = np.load(f"{G}/PD_h6_{kind}.npy").astype(np.float64); alts = [mean(D_SRC[k], kind) for k in ok]
        n = pl.read_parquet(f"{G}/D_{kind}_rows.parquet", columns=["txn_id"]).height; assert all(x.shape == (n, T.K) for x in [A] + alts)
        for w in WEIGHTS: np.save(f"{G}/PD_MIX{w}_{kind}.npy", mix(A, alts, w / 100))
        okn = [k for k in N_SRC if k in ok and have(N_SRC[k], (kind,))]; A = np.load(f"{G}/PN_C_{kind}.npy").astype(np.float64); alts = [mean(N_SRC[k], kind) for k in okn]
        for w in WEIGHTS: np.save(f"{G}/PN_MIXN{w}_{kind}.npy", mix(A, alts, w / 100))
        print(kind, "D mixtures:", len(ok), "alternatives | N mixtures:", okn, flush=True)
else:
    w = int(sys.argv[2]); tag = sys.argv[3] if len(sys.argv) > 3 else f"cmb{w}"; ok = d_sources()  # optional tag: dry runs
    for f in (f"oof/{tag}.npy", f"oof/{tag}_ids.parquet", f"preds/{tag}_test.npy"): assert not os.path.exists(f), f"{f} exists; never overwrite"
    ids = pl.read_parquet("oof/h6_ids.parquet"); V = np.zeros((ids.height, T.K), np.float32); seen = np.zeros(ids.height, bool)
    for kind, fold in (("aug", 0), ("sep", 1)):
        rows = pl.read_parquet(f"{G}/D_{kind}_rows.parquet", columns=["txn_id"]).with_row_index("i")
        j = ids.with_row_index("o").filter(c("fold") == fold).join(rows, on="txn_id", how="left", maintain_order="left")
        assert j["i"].null_count() == 0 and j.height == rows.height and j["i"].n_unique() == rows.height
        M = mix(np.load(f"{G}/PD_h6_{kind}.npy").astype(np.float64), [mean(D_SRC[k], kind) for k in ok], w / 100)
        V[j["o"].to_numpy()] = M[j["i"].to_numpy()]; seen[j["o"].to_numpy()] = True
    assert seen.all(), "every validation row must have every source"
    H = np.load("oof/h6.npy"); print(f"validation blend vs h6: argmax differs on {int((V.argmax(1) != H.argmax(1)).sum())} of {len(V)} rows")
    Tt = mix(np.load("preds/h6_test.npy").astype(np.float64), [oct_pred(k, D_OCT[k]) for k in ok], w / 100)
    H = np.load("preds/h6_test.npy"); print(f"test blend vs h6: argmax differs on {int((Tt.argmax(1) != H.argmax(1)).sum())} of {len(Tt)} rows")
    np.save(f"oof/{tag}.npy", V); shutil.copyfile("oof/h6_ids.parquet", f"oof/{tag}_ids.parquet"); np.save(f"preds/{tag}_test.npy", Tt)
    print("written", tag, "sources", ok, flush=True)
