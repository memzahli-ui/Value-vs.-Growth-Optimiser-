import pandas as pd, wrds
import numpy as np
from pathlib import Path

# Paths & constants
OUT_DIR = Path("outputs");  OUT_DIR.mkdir(exist_ok=True)
CACHE_SP500 = Path("sp500_tr.parquet")
START, END = "1973-01-31", "2025-12-31"
DATE_IDX = pd.date_range(START, END, freq="ME")

# S&P-500 total-return (sprtrn) from crsp.msi
def load_sp500_tr(force_refresh=False):

    if CACHE_SP500.exists() and not force_refresh:
        return pd.read_parquet(CACHE_SP500)["SP500_TR"]

    print("Pulling S&P-500 total return (sprtrn) from crsp.msi")
    conn = wrds.Connection(wrds_username="maryamahli")

    q = f"""
        SELECT date, sprtrn
        FROM   crsp.msi
        WHERE  date BETWEEN '{START}' AND '{END}'
        ORDER  BY date;
    """
    df = conn.raw_sql(q, date_cols=["date"])
    conn.close()

    df["sprtrn"] = pd.to_numeric(df["sprtrn"], errors="coerce")
    ser = df.set_index("date")["sprtrn"].resample("ME").last().rename("SP500_TR")
    ser.to_frame().to_parquet(CACHE_SP500)
    print(f"Cached S&P-500 TR to {CACHE_SP500.name} ({len(ser):,} rows)")
    return ser


# Equal-weight style helper
def equal_weight(panel: pd.DataFrame, style: str) -> pd.Series:
    sub = panel[panel["style"] == style]
    ew = sub.groupby("date", observed=True, sort=False).agg(EW_Return=("rexcess", "mean")).squeeze()
    ew.name = f"EW_{style}"
    return ew

# Cap-weighted style helper
def value_weight(panel: pd.DataFrame, style: str) -> pd.Series:
    sub = panel[panel["style"] == style]
    def wavg(df):
        if df["rexcess"].isna().all():
            return np.nan
        return (df["rexcess"] * df["me"]).sum() / df["me"].sum()
    vw = sub.groupby("date", observed=True).apply(wavg, include_groups=False)
    vw.name = f"VW_{style}"
    return vw

# Load CRSP-factor panel & RF
panel = pd.read_parquet("outputs/crsp_factors.parquet", columns=["date", "style", "rexcess", "mktcap"]).rename(columns={"mktcap": "me"}).assign(date=lambda d: pd.to_datetime(d["date"])+pd.offsets.MonthEnd(0))

ff = pd.read_parquet("french_factors.parquet")
if ff.index.name != "date":
    ff = ff.set_index("date")
ff.index = pd.to_datetime(ff.index) + pd.offsets.MonthEnd(0)
rf = (ff["RF"] / 100).reindex(DATE_IDX)

# Assemble benchmark DataFrame
bench = pd.DataFrame(index=DATE_IDX)
bench["SP500_TR"] = load_sp500_tr().reindex(DATE_IDX) # total return
bench["EW_Value"] = equal_weight(panel, "Value").reindex(DATE_IDX)
bench["VW_Value"] = value_weight(panel, "Value").reindex(DATE_IDX)
bench["EW_Growth"] = equal_weight(panel, "Growth").reindex(DATE_IDX)
bench["VW_Growth"] = value_weight(panel, "Growth").reindex(DATE_IDX)

# Only convert S&P-500 to excess return
bench["SP500_TR"] = bench["SP500_TR"] - rf # now excess return
bench.rename(columns={"SP500_TR": "SP500_ER"}, inplace=True)
bench.to_parquet(OUT_DIR / "benchmarks.parquet")

print("\n=== Benchmark series created ===")
print(bench.describe().loc[["count", "mean", "std", "min", "max"]])
print(f"\nSaved to {OUT_DIR/'benchmarks.parquet'}")