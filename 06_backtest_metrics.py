import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis

# File paths & crisis windows
OUT_DIR = Path("outputs")
WGT_DIR = OUT_DIR / "weights"

PANEL = pd.read_parquet("outputs/crsp_factors.parquet")[["permno","date","rexcess"]].copy()
PANEL["date"] = pd.to_datetime(PANEL["date"]) + pd.offsets.MonthEnd(0)

CRISES = {
    "Full"        : ("1900-01-31",  "2100-12-31"),
    "Black87"     : ("1987-08-31",  "1987-12-31"),
    "DotCom"      : ("2000-03-31",  "2002-09-30"),
    "GFC"         : ("2007-07-31",  "2009-03-31"),
    "Covid"       : ("2020-02-29",  "2021-03-31"),
    "RateShock22" : ("2022-01-31",  "2023-10-31"),
}

# Functions
def cvar(series, p=0.05):
    cutoff = series.quantile(p)
    return series[series <= cutoff].mean()

def max_dd_and_recovery(series):
    cum = (1 + series).cumprod()
    peak = cum.cummax()
    dd = (cum/peak) - 1
    max_dd = dd.min()

    recov = np.nan
    if max_dd < 0:
        trough_idx = dd.idxmin()
        post = cum.loc[trough_idx:]
        recov_idx = post[post >= peak.loc[trough_idx]].first_valid_index()
        if recov_idx is not None:
            recov = (recov_idx.to_period("M") - trough_idx.to_period("M")).n
    return max_dd, recov

def add_arrows(df):
    nice = df.copy()
    num_cols = ["mean","stdev","skew","ex_kurt","cvar_5","max_dd","recov_m","sharpe","sortino"]
    high_good = {"mean","skew","max_dd","recov_m","sharpe","sortino"}
    # low_good  = {"stdev","kurt","cvar_5"}

    for col in num_cols:
        vals = nice[col].astype(float)
        best = vals.max() if col in high_good else vals.min()
        worst = vals.min() if col in high_good else vals.max()

        mark = pd.Series([""]*len(vals), index=vals.index, dtype=object)
        mark.loc[vals == best]  = "↑"
        mark.loc[vals == worst] = "↓"
        nice[col] = mark + " " + vals.map("{:+.4f}".format)

    nice.insert(1, "period", nice.pop("start") + " → " + nice.pop("end"))
    return nice

# Build portfolio return series
returns_all = []

for fp in sorted(WGT_DIR.glob("weights_*.parquet")):
    style, model = fp.stem.split("_")[1:]
    tag = f"{style}_{model}"

    wgt = pd.read_parquet(fp)
    wgt["date"] = pd.to_datetime(wgt["date"]) + pd.offsets.MonthEnd(0)
    wgt["hold_date"] = wgt["date"] + pd.offsets.MonthEnd(1)

    merged = wgt.merge(PANEL, left_on=["permno","hold_date"], right_on=["permno","date"], how="left", suffixes=("","_ret")).dropna(subset=["rexcess"])

    port_ret = merged.groupby("hold_date").apply(lambda df: np.dot(df["weight"],df["rexcess"]), include_groups=False).rename(tag)
    returns_all.append(port_ret)

# Combine into a single DataFrame
rets = pd.concat(returns_all, axis=1).sort_index()
rets.to_parquet(OUT_DIR / "pnl_series.parquet")
print(f"Built P&L series – shape {rets.shape}")

# Compute risk metrics
rows = []
for lbl, (t0, t1) in CRISES.items():
    sub = rets.loc[t0:t1]
    for strat in rets.columns:
        s = sub[strat].dropna()
        if s.empty:  # nothing to measure
            continue

        # ---- path-dependent helpers -----------------------------------------
        max_dd, recov = max_dd_and_recovery(s)

        # ---- moment statistics ----------------------------------------------
        mu = s.mean()
        sd = s.std()

        # downside deviation (√E[min(r,0)²])
        ddv = np.sqrt(np.mean(np.square(np.minimum(s, 0))))

        rows.append({
            "window": lbl,
            "start": t0,
            "end": t1,
            "strategy": strat,
            "n_months": len(s),

            # level-1 moments
            "mean": mu,
            "stdev": sd,
            "skew": skew(s, bias=False),
            "ex_kurt": kurtosis(s, fisher=True, bias=False),

            # downside & tail
            "cvar_5": cvar(s, 0.05),
            "max_dd": max_dd,
            "recov_m": recov,

            # NEW risk-adjusted ratios
            "sharpe": mu / sd if sd > 0 else np.nan,
            "sortino": mu / ddv if ddv > 0 else np.nan,
        })

metrics = pd.DataFrame(rows)
metrics.to_parquet(OUT_DIR / "risk_metrics.parquet")

# Display
print("\n=====  Risk metrics  =====")
for win in metrics["window"].unique():
    block = metrics[metrics["window"] == win]
    print(f"\n### {win} ###")
    print(add_arrows(block).to_string(index=False))