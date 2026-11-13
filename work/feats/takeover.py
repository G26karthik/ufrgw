"""sim_swap / account_takeover: sender drain bursts.

Generator signature (see notes/takeover.md): a burst of 2-4 P2P PAY rows from the victim (sender) within ~1h, each to a
new payee, integer amounts, each ~0.7x the current balance (last rows often 500 > balance). sim_swap bursts: same device,
sim_age < 48h, home city, often at night. ATO bursts: a device the sender never used outside the burst, device_age < 3d,
device_city away from home, more pin retries. Features are per sender-session (gap <= 2h), no labels used.
"""
import polars as pl

GAP = 7200  # session break (s); fraud bursts span < ~1.3h


def build(tx: pl.DataFrame) -> pl.DataFrame:
    P = "takeover__"
    a, bal = pl.col("amount"), pl.col("balance_before")
    s = tx.select("txn_id", "ts", "sender_id", "receiver_id", "receiver_type", "device_id", "device_city", "snd_home_city",
                  "amount", "balance_before", "sim_age_hours", "device_age_days", "new_payee", "pin_retries",
                  "login_to_txn_sec", "txn_type", pl.col("timestamp").dt.hour().alias("hour")).sort("sender_id", "ts")
    s = s.with_columns(
        (pl.col("ts").diff().over("sender_id").fill_null(GAP + 1) > GAP).cum_sum().alias("sess"),
        (a / bal).alias("share"),
        ((pl.col("receiver_type") == "P2P") & (pl.col("new_payee") == 1)).alias("p2p_new"),
        (a % 1 == 0).alias("int_amt"),
        (pl.col("device_city") != pl.col("snd_home_city")).alias("away"),
    ).with_columns(
        (pl.col("p2p_new") & (pl.col("share") > 0.4) & pl.col("int_amt")).alias("drain"),
        pl.len().over("sender_id").alias("n_s"),
        pl.len().over("sender_id", "device_id").alias("n_sd"),
        pl.len().over("sess", "device_id").alias("n_sd_sess"),
        pl.len().over("sender_id", "device_city").alias("n_sc"),
        pl.len().over("sess", "device_city").alias("n_sc_sess"),
        pl.len().over("sess").alias("sess_n"),
    ).with_columns(
        # device only ever used by this sender inside this session, while the sender has other activity
        ((pl.col("n_sd") == pl.col("n_sd_sess")) & (pl.col("n_s") > pl.col("sess_n"))).alias("dev_excl"),
        ((pl.col("n_sc") == pl.col("n_sc_sess")) & (pl.col("n_s") > pl.col("sess_n")) & pl.col("away")).alias("city_excl"),
        (pl.col("sim_age_hours") < 48).alias("sim_new"),
        (pl.col("device_age_days") < 3).alias("dev_young"),
        ((pl.col("hour") <= 5)).alias("night"),
        bal.shift(-1).over("sess").alias("next_bal"),
    )
    # receiver ring reuse: distinct senders that drained into this receiver
    rcv = s.filter(pl.col("drain")).group_by("receiver_id").agg(pl.col("sender_id").n_unique().alias("rcv_drain_snd"))
    s = s.join(rcv, on="receiver_id", how="left").with_columns(pl.col("rcv_drain_snd").fill_null(0))
    g = s.group_by("sess").agg(
        pl.col("sess_n").first(),
        (pl.col("ts").max() - pl.col("ts").min()).alias("sess_span"),
        pl.col("p2p_new").sum().alias("sess_p2p_new"),
        pl.col("drain").sum().alias("sess_drain"),
        pl.col("int_amt").mean().alias("sess_int_frac"),
        pl.col("share").mean().alias("sess_share_mean"),
        pl.col("share").min().alias("sess_share_min"),
        (pl.col("amount").sum() / pl.col("balance_before").max()).alias("sess_drain_ratio"),
        (pl.col("balance_before").min() / pl.col("balance_before").max()).alias("sess_bal_min_max"),
        (pl.col("next_bal") < pl.col("balance_before")).sum().alias("sess_bal_down"),
        pl.col("sim_new").mean().alias("sess_sim_new"),
        pl.col("sim_age_hours").median().log1p().alias("sess_sim_med_log"),
        pl.col("dev_young").mean().alias("sess_dev_young"),
        pl.col("device_age_days").median().log1p().alias("sess_devage_med_log"),
        pl.col("dev_excl").mean().alias("sess_dev_excl"),
        pl.col("city_excl").mean().alias("sess_city_excl"),
        pl.col("away").mean().alias("sess_away"),
        pl.col("night").mean().alias("sess_night"),
        pl.col("pin_retries").mean().alias("sess_pin"),
        pl.col("login_to_txn_sec").median().alias("sess_login_med"),
        pl.col("device_id").n_unique().alias("sess_ndev"),
        (pl.col("txn_type") == "PAY").mean().alias("sess_pay_frac"),
        pl.col("rcv_drain_snd").max().alias("sess_rcv_ring"),
    )
    s = s.join(g.drop("sess_n"), on="sess", how="left").with_columns(
        pl.col("ts").rank("ordinal").over("sess").alias("pos"),
        (pl.col("next_bal") / (pl.col("balance_before") - pl.col("amount")).clip(1, None)).alias("next_bal_ratio"),
        (pl.col("sess_sim_new") - pl.col("sess_dev_young")).alias("sess_sim_vs_dev"),
        # event archetypes (type labels are ~30% swapped between the two in train; archetype is the clean signal)
        ((pl.col("sess_drain") >= 2) & (pl.col("sess_sim_new") >= 0.5) & (pl.col("sess_dev_young") < 0.5)).alias("arch_sim"),
        ((pl.col("sess_drain") >= 2) & (pl.col("sess_dev_young") >= 0.5)).alias("arch_ato"),
        (pl.col("n_sd") / pl.col("n_s")).alias("dev_frac"),
        (pl.col("n_s") - pl.col("n_sd")).alias("n_other_dev"),
    )
    # sess_span / sess_bal_down / next_bal_ratio: nonzero rate halves in Oct (fewer txns per sender -> fewer legit multi-row
    # sessions); prior shift only, and dropping them cost ~0.006 macro (mostly layering), so kept.
    keep = ["dev_excl", "city_excl", "dev_frac", "n_other_dev", "pos", "next_bal_ratio", "rcv_drain_snd", "sess_n",
            *[c for c in g.columns if c.startswith("sess_") and c != "sess_n"], "sess_sim_vs_dev", "arch_sim", "arch_ato"]
    return s.select("txn_id", *[pl.col(c).cast(pl.Float32).alias(P + c) for c in keep])
