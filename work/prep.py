"""Convert raw CSVs to parquet (one-time) so every later script loads fast."""
import polars as pl
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "upi-fraud-rings"
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

for name in ["train", "test", "accounts", "merchants"]:
    df = pl.read_csv(RAW / f"{name}.csv", infer_schema_length=100000)
    if "timestamp" in df.columns:
        df = df.with_columns(pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S"))
    df.write_parquet(OUT / f"{name}.parquet")
    print(name, df.shape)
    print(df.schema)
