# Write console output to a text file
import sys, io, pathlib

class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams
    def write(self, s):
        for stream in self._streams:
            stream.write(s)
    def flush(self):
        for stream in self._streams:
            stream.flush()

log_path = pathlib.Path("outputs/compare_tables.txt")
log_path.parent.mkdir(exist_ok=True)
sys.stdout = Tee(sys.stdout, log_path.open("w", encoding="utf-8"))

# Main script
import warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore", category=FutureWarning)

# Constants & helpers
CRISES = {
    "Full"        : ("1900-01-31", "2100-12-31"),
    "Black87"     : ("1987-08-31", "1987-12-31"),
    "DotCom"      : ("2000-03-31", "2002-09-30"),
    "GFC"         : ("2007-07-31", "2009-03-31"),
    "Covid"       : ("2020-02-29", "2021-03-31"),
    "RateShock22" : ("2022-01-31", "2023-10-31"),
}

DEC = 3 # decimals to print
MIN_OBS = 3 # require ≥3 monthly obs for a window

def downside_std(x: pd.Series) -> float:
    neg = x[x < 0]
    return np.sqrt(np.mean(np.square(neg))) if len(neg) else np.nan

def max_dd_and_recov(s: pd.Series):
    # draw-down & months-to-recover
    cum = (1 + s).cumprod()
    peak = cum.cummax()
    dd   = cum / peak - 1
    mdd  = dd.min()
    if mdd == 0:
        return 0.0, 0
    trough = dd.idxmin()
    recov  = cum[trough:] >= peak[trough]
    idx    = recov.idxmax() if recov.any() else None
    months = ((idx.to_period("M") - trough.to_period("M")).n
              if idx is not None else np.nan)
    return mdd, months

# Load data
rets = pd.read_parquet("outputs/pnl_series.parquet") # strategies
bench = pd.read_parquet("outputs/benchmarks.parquet") # benchmarks

# Synchronise dates
rets.index = pd.to_datetime(rets.index)
bench.index = pd.to_datetime(bench.index)
common_idx = rets.index.intersection(bench.index)
rets = rets.reindex(common_idx)
bench = bench.reindex(common_idx)

# Compute metrics
rows = []
for win, (t0, t1) in CRISES.items():
    sub_r = rets.loc[t0:t1]
    sub_b = bench.loc[t0:t1]

    for strat in sub_r.columns:
        for bm in sub_b.columns:
            s = sub_r[strat].dropna()
            b = sub_b[bm].dropna()
            common = s.index.intersection(b.index)
            if len(common) < MIN_OBS:
                continue

            p = s.loc[common]
            q = b.loc[common]
            active = p - q

            # Active-risk stats
            te = active.std(ddof=0)
            ir = active.mean() / te if te else np.nan
            act_sharpe = ir  # identical, but printed separately for now
            sort = active.mean() / downside_std(active)
            cvar5 = active[active <= active.quantile(0.05)].mean()
            mdd, recov = max_dd_and_recov(active)

            # Co-movement stats
            beta = np.cov(p, q, ddof=0)[0, 1] / q.var(ddof=0)
            rho = np.corrcoef(p, q)[0, 1]

            rows.append({
                "window": win,
                "strategy": strat,
                "benchmark": bm,

                "active_mu": active.mean(),
                "TE": te,
                "IR": ir,
                "act_sharpe": act_sharpe,
                "act_sortino": sort,
                "act_cvar_5": cvar5,
                "act_max_dd": mdd,
                "act_recov_m": recov,

                "beta": beta,
                "rho": rho,
            })

cmp = pd.DataFrame(rows)

# Save
OUT = Path("outputs/compare_metrics.parquet")
cmp.to_parquet(OUT)
print(f"\nSaved comparison metrics to {OUT}")

# Console report
SHOW = ["active_mu", "TE", "IR", "act_sharpe", "act_sortino", "act_cvar_5", "act_max_dd", "act_recov_m", "beta", "rho"]
HIGH_GOOD = {"active_mu", "IR", "act_sharpe", "act_sortino", "act_max_dd", "act_cvar_5"} # metrics where larger values are preferred

fmt = lambda x: "" if pd.isna(x) else f"{x:+.{DEC}f}"

for win in cmp["window"].unique():
    block = (cmp[cmp["window"] == win]
             .set_index(["strategy", "benchmark"])[SHOW])

    # Arrow layer
    arr = pd.DataFrame("", index=block.index, columns=block.columns, dtype=object)
    for c in SHOW:
        col = block[c]
        if col.notna().sum() == 0:
            continue
        best = col.idxmax() if c in HIGH_GOOD else col.idxmin()
        worst = col.idxmin() if c in HIGH_GOOD else col.idxmax()
        arr.at[best,  c] = "↑"
        arr.at[worst, c] = "↓"

    pretty = arr + " " + block.map(fmt)

    print(f"\n===== Information-ratio summary ({win}) =====")
    print(pretty.to_string())

print("\nDone.")