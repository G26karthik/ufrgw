"""Post-processing of class probabilities; the same transform runs on OOF and test (notes/postproc.md).

apply(P, ids_df, tx) -> P'   ids_df: txn_id in P's row order; tx: common.load_all()
  1. Takeover archetype decides the type: takeover__arch_sim=1 -> sim_swap, arch_ato=1 -> account_takeover.
     Inside one archetype the sim/ATO labels are ~30% swapped (notes/takeover.md); the model splits the rows, forcing
     the archetype's majority type wins (m5dm: +0.007 pooled OOF, +0.006 cleaned proxy).
  2. Nov-dated rows (test only) are never legit: every one is spill-over of an Oct fraud event (notes/drift.md).
  3. qr_null=True: QR to a P2P account with qr_vpa_mismatch null -> qr_swap. Oct-only evasion (0 such train rows),
     so it cannot be validated; P2P-QR rows with mismatch=1 are 84% qr_swap on cleaned Sep.
Event smoothing, other structural rules and EM prior shift were tested and dropped (no gain, notes/postproc.md).

usage (cwd=work): python postproc.py TAG [--qr-null]
  reads oof/TAG.npy (+_ids) and preds/TAG_test.npy, re-tunes biases on the post-processed OOF, prints pooled / Sep
  macro-F1 on noisy and cleaned labels, writes subs/TAG_pp.csv (TAG_ppq.csv with --qr-null).
"""
import argparse
import numpy as np
import polars as pl
from common import ROOT, LABELS, LABEL2ID
from density import run_feature_matched

K = len(LABELS)
c = pl.col


def apply(P, ids_df, tx, qr_null=False):
    assert P.shape == (ids_df.height, K)
    tk = run_feature_matched("takeover", tx).select("txn_id", "takeover__arch_sim", "takeover__arch_ato").fill_nan(None)
    D = (ids_df.select("txn_id")
         .join(tx.select("txn_id", "month", "txn_type", "receiver_type", "qr_vpa_mismatch"), on="txn_id", how="left", maintain_order="left")
         .join(tk, on="txn_id", how="left", maintain_order="left"))
    assert D["month"].null_count() == 0, "ids not in tx"
    flag = lambda e: D.select(e.fill_null(False))[:, 0].to_numpy()
    rules = [(c("takeover__arch_sim") == 1, LABEL2ID["sim_swap"]), (c("takeover__arch_ato") == 1, LABEL2ID["account_takeover"])]
    if qr_null:
        rules.append(((c("txn_type") == "QR") & (c("receiver_type") == "P2P") & c("qr_vpa_mismatch").is_null(), LABEL2ID["qr_swap"]))
    Q = P.astype(np.float32, copy=True)
    for e, k in rules:
        h = flag(e); Q[h] = 0; Q[h, k] = 1
    nov = flag(c("month") == 11); Q[nov, 0] = 0; Q[nov] /= Q[nov].sum(1, keepdims=True)
    return Q


def main():
    from common import load_all
    from clean_labels import clean
    from train import tune_bias, decode, macro_f1, write_sub, SAMPLE
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--qr-null", action="store_true")
    a = ap.parse_args()
    tx = load_all()
    ids = pl.read_parquet(ROOT / "oof" / f"{a.tag}_ids.parquet")
    tid = pl.read_csv(SAMPLE).select(pl.col("id").alias("txn_id"))
    y = ids["y"].to_numpy().astype(np.int64); sep = ids["fold"].to_numpy() == 1
    yc = ids.join(tx.select("txn_id", clean(tx).alias("yc")), on="txn_id", how="left", maintain_order="left")["yc"].to_numpy().astype(np.int64)
    P0, T0 = np.load(ROOT / "oof" / f"{a.tag}.npy"), np.load(ROOT / "preds" / f"{a.tag}_test.npy")
    P1, T1 = apply(P0, ids, tx, a.qr_null), apply(T0, tid, tx, a.qr_null)
    for name, P in (("raw ", P0), ("post", P1)):  # same decoding for both: bias re-tuned on the pooled OOF
        b = tune_bias(P, y); pr = decode(P, b)
        print(f"{name}: pooled={macro_f1(y, pr).mean():.4f} sep={macro_f1(y[sep], pr[sep]).mean():.4f} | cleaned "
              f"pooled={macro_f1(yc, pr).mean():.4f} sep={macro_f1(yc[sep], pr[sep]).mean():.4f}  bias={np.round(b, 2).tolist()}", flush=True)
    print(f"test rows whose label changes vs raw probs (post bias): {int((decode(T1, b) != decode(T0, b)).sum())}")
    write_sub(T1, b, f"{a.tag}_pp" + ("q" if a.qr_null else ""))


if __name__ == "__main__":
    main()
