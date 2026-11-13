"""Combined type-split post-rule (judge of typesplit hunters; notes/typesplit_judge.md).

TAG preds -> postproc.apply(qr_null) -> bias tuned on pooled post OOF (postproc.py convention) ->
  1) device-id zone override (ts_event layout; zone edges found label-free by ts_event_post.dp_edges on the model's own
     decoded preds: train ids from OOF, test ids from test preds): every row on a zone device takes the zone's type;
  2) bait rule (feats/ts_bait.py) on rows outside zones decoded ph/fc/loan: rule 1 -> phishing, rule 2 -> fake_cashback.
Reports Sep noisy / cleaned (pooled bias) and OOS (bias tuned on Aug, scored on Sep), writes subs/ts_judge_<TAG>.csv.
--mv also pools mule/velocity preds as zone evidence in the DP (art_zonemax, notes/art_zonemax.md; 1 test row on h6).
usage (cwd=work): python ts_judge_post.py [TAG=h6] [--mv] [--out NAME]
  ts_judge_h6.csv:  python ts_judge_post.py h6        art_zonemax.csv:  python ts_judge_post.py h6 --mv --out art_zonemax
"""
import argparse
import numpy as np
import polars as pl
from common import ROOT, LABELS, LABEL2ID as L, load_all, run_feature
from train import SAMPLE, decode, tune_bias, macro_f1, write_sub
from postproc import apply
from clean_labels import clean
from feats.ts_event import TR_EDGES, TYPES, zone_ids, test_block
from ts_event_post import dp_edges

c = pl.col
SA = [L["sim_swap"], L["account_takeover"]]
BAIT = [L["phishing_collect"], L["fake_cashback"], L["loan_app_extortion"]]
JUDGE, MV = ((3, 4),), ((3, 4), (0, 1))  # dp_edges evidence groups (TYPES indices): sim/ATO; + mule/velocity


def zones(D, p, lo, hi, groups=JUDGE):
    """D: rows (dn) aligned with decoded preds p -> (zone label id per row, edges)."""
    g = (D.with_columns(pl.Series("p", p)).filter(c("dn").is_between(lo, hi - 1)).group_by("dn")
         .agg(c("p"), (pl.len() > 1).alias("multi")).sort("dn"))
    e = dp_edges(g["dn"].to_list(), g["p"].to_list(), g["multi"].to_list(), lo, hi, groups)
    d = D["dn"].to_numpy()
    return zone_ids(d, np.ones(len(d), bool), e), e


def override(p, z, rule):
    q = np.where(z > 0, z, p)
    out = (z == 0) & np.isin(p, BAIT)
    q[out & (rule == 1)] = L["phishing_collect"]; q[out & (rule == 2)] = L["fake_cashback"]
    return q


def get(tag="h6", groups=JUDGE):
    """Decoded arrays of the judge on TAG: OOF (o, pooled bias), OOS (a, Aug bias), test (t); p = model decode,
    q = after zone+bait override, z = zone label id, e = zone edges, r = bait rule."""
    tx = load_all()
    ids = pl.read_parquet(ROOT / "oof" / f"{tag}_ids.parquet")
    tid = pl.read_csv(SAMPLE).select(c("id").alias("txn_id"))
    base = tx.select("txn_id", c("device_id").str.slice(2).cast(pl.Int64).alias("dn")).join(
        run_feature("ts_bait", tx).select("txn_id", c("ts_bait__rule").fill_null(0)), on="txn_id", how="left")
    Do = ids.select("txn_id").join(base, on="txn_id", how="left", maintain_order="left")
    Dt = tid.join(base, on="txn_id", how="left", maintain_order="left")
    y = ids["y"].to_numpy().astype(np.int64); fold = ids["fold"].to_numpy()
    yc = ids.join(tx.select("txn_id", clean(tx).alias("yc")), on="txn_id", how="left", maintain_order="left")["yc"].to_numpy().astype(np.int64)
    P1 = apply(np.load(ROOT / "oof" / f"{tag}.npy"), ids, tx, True)
    T1 = apply(np.load(ROOT / "preds" / f"{tag}_test.npy"), tid, tx, True)
    b = tune_bias(P1, y); po, pt = decode(P1, b), decode(T1, b)
    ba = tune_bias(P1[fold == 0], y[fold == 0]); pa = decode(P1, ba)  # OOS: Aug-tuned bias (edges are label-free)
    zo, eo = zones(Do, po, TR_EDGES[0], TR_EDGES[-1], groups)
    zt, et = zones(Dt, pt, *test_block(tx), groups)
    ro, rt = Do["ts_bait__rule"].to_numpy(), Dt["ts_bait__rule"].to_numpy()
    qo, qt, qa = override(po, zo, ro), override(pt, zt, rt), override(pa, zo, ro)
    return dict(oid=ids["txn_id"].to_numpy(), tid=tid["txn_id"].to_numpy(), y=y, yc=yc, fold=fold, po=po, pt=pt, pa=pa,
                qo=qo, qt=qt, qa=qa, zo=zo, zt=zt, eo=eo, et=et, ro=ro, rt=rt)


def gate(d, qo_new, qa_new, label=""):
    """Sep noisy / cleaned (pooled bias) and OOS (Aug bias) macro for the judge and a candidate."""
    S = d["fold"] == 1
    r = []
    for nm, q, qa in (("judge", d["qo"], d["qa"]), (label, qo_new, qa_new)):
        r.append((nm, macro_f1(d["y"][S], q[S]).mean(), macro_f1(d["yc"][S], q[S]).mean(), macro_f1(d["y"][S], qa[S]).mean()))
        print(f"{nm:24s} Sep noisy {r[-1][1]:.4f} cleaned {r[-1][2]:.4f} OOS {r[-1][3]:.4f}")
    return r


def main(tag, groups, out):
    d = get(tag, groups)
    y, yc, S, po, pt, pa, qo, qt, zo, zt = (d[k] for k in ("y", "yc", "fold", "po", "pt", "pa", "qo", "qt", "zo", "zt"))
    S = S == 1
    print(f"edges train {d['eo']}\n      test  {d['et']}")
    for name, p in (("base", po), ("zone+bait", qo)):
        fn, fc = macro_f1(y[S], p[S]), macro_f1(yc[S], p[S])
        print(f"Sep {tag} {name:9s} noisy {fn.mean():.4f} cleaned {fc.mean():.4f}  per-class cleaned {np.round(fc, 3).tolist()}")
    print(f"OOS Aug->Sep {tag}: base {macro_f1(y[S], pa[S]).mean():.4f}  zone+bait {macro_f1(y[S], d['qa'][S]).mean():.4f}")
    # Oct consistency: share of non-legit preds inside each zone that match the zone (sim/ATO pooled), Sep vs Oct
    for nm, z, p in (("Sep", zo[S], po[S]), ("Oct", zt, pt)):
        r = []
        for k in TYPES:
            m = (z == L[k]) & (p > 0); ok = np.isin(p[m], SA) if L[k] in SA else p[m] == L[k]
            r.append(f"{k[:6]} {int((z == L[k]).sum())}/{ok.mean() if m.any() else float('nan'):.2f}")
        print(f"{nm} zone rows/agree: " + "  ".join(r))
    lab = np.array(LABELS)[qt]
    print(f"test rows changed vs {tag} decode: {int((qt != pt).sum())}")
    ch = np.array(LABELS)[pt] != lab
    print(pl.DataFrame({"from": np.array(LABELS)[pt][ch], "to": lab[ch]}).group_by("from", "to").len().sort("len", descending=True).head(20))
    write_sub(np.eye(len(LABELS), dtype=np.float32)[qt], np.zeros(len(LABELS)), out or f"ts_judge_{tag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", default="h6")
    ap.add_argument("--mv", action="store_true", help="pool mule/velocity preds as zone evidence (art_zonemax)")
    ap.add_argument("--out", default="", help="submission name (default ts_judge_<TAG>)")
    a = ap.parse_args()
    main(a.tag, MV if a.mv else JUDGE, a.out)
