"""Mechanism `rel`: agent B's 26 relational / ordering columns (role transitions, in/out activity change against the account's own
earlier window, secondary device + directional counterparty edges), computed by B's own `feat(ts)` (copied unchanged from
scratch_atomic/relationships/s1_patterns.py at 07:58 into B_s1_patterns_0758.py; the function body is executed from that copy).
Every column uses only rows strictly earlier than the row's own timestamp, plus the row's own ids. No labels are read."""
import numpy as np
import polars as pl

SRC = open("scratch_atomic/verification/B_s1_patterns_0758.py", encoding="utf8").read()
SRC = SRC[SRC.index("def feat(ts):"):SRC.index("def motifs(o):")]


def build(tx):
    num = lambda k: tx[k].str.slice(2).cast(pl.Int64).to_numpy()
    s = num("sender_id"); p2p = (tx["receiver_type"] == "P2P").to_numpy(); ts0 = tx["ts"].to_numpy().astype(np.int64)
    g = dict(np=np, pl=pl, DAY=86400, BIG=10**10, PB=2_000_000, n=tx.height, s=s, p2p=p2p, dn=num("device_id"),
             r=np.where(p2p, num("receiver_id"), 1_000_000 + num("receiver_id")), NACC=420000, T0=int(ts0.min()))  # NACC as printed by B's run
    exec(SRC, g)
    o = g["feat"](ts0)
    return pl.DataFrame({"txn_id": tx["txn_id"], **{"rel__" + k: v.astype(np.float32) for k, v in o.items()}})
