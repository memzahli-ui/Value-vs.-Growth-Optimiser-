import numpy as np
import pandas as pd
from pathlib import Path
from arch.utility import cov_nw
from scipy import stats

OUT_DIR = Path("outputs")
mu = (pd.read_parquet(OUT_DIR / "mu_vectors.parquet")
        .rename(columns={"mu_capm": "CAPM",
                         "mu_ff3" : "FF3",
                         "mu_ff5" : "FF5"}))

rf = pd.read_parquet(OUT_DIR / "crsp_factors.parquet")[["permno", "date", "rexcess", "RF"]]
mu["date"] = pd.to_datetime(mu["date"]) + pd.offsets.MonthEnd(1)  # align forecasts to t+1
panel = rf.merge(mu, on=["permno", "date"], how="inner").dropna()

for m in ["CAPM", "FF3", "FF5"]:
    panel[f"{m}_EXC"] = panel[m] - panel["RF"]

def mse(x): return (x**2).mean()

loss = (panel.groupby("date")
              .agg(
                  LCAPM = ("rexcess", lambda y: mse(y - panel.loc[y.index, "CAPM_EXC"])),
                  LFF3  = ("rexcess", lambda y: mse(y - panel.loc[y.index, "FF3_EXC"])),
                  LFF5  = ("rexcess", lambda y: mse(y - panel.loc[y.index, "FF5_EXC"])),
                  ADJ35 = ("rexcess", lambda y: mse(panel.loc[y.index, "FF3_EXC"] - panel.loc[y.index, "CAPM_EXC"])),
                  ADJ53 = ("rexcess", lambda y: mse(panel.loc[y.index, "FF5_EXC"] - panel.loc[y.index, "FF3_EXC"])),
                  ADJ55 = ("rexcess", lambda y: mse(panel.loc[y.index, "FF5_EXC"] - panel.loc[y.index, "CAPM_EXC"]))
              ))

cw = pd.DataFrame({
    "d_CPvsFF3":  loss["LCAPM"] - (loss["LFF3"] - loss["ADJ35"]),
    "d_FF3vsFF5": loss["LFF3"] - (loss["LFF5"] - loss["ADJ53"]),
    "d_CPvsFF5":  loss["LCAPM"] - (loss["LFF5"] - loss["ADJ55"]),
})

# HAC t‑statistics
T = len(cw)
lag = int(np.sqrt(T))

records = []
for col in cw.columns:
    f = np.asarray(cw[col].dropna().to_numpy(), dtype=float)
    var = cov_nw(f.reshape(-1, 1), lags=lag)[0, 0]
    t_stat = f.mean() / np.sqrt(var / len(f))
    p_val  = 1 - stats.t.cdf(t_stat, df=len(f) - 1)   # H1: richer beats simpler
    records.append({"pair": col, "t_stat": t_stat, "p_value": p_val})

pd.DataFrame(records).to_csv(OUT_DIR / "cw_factor_tests.csv", index=False)
print("Saved results: outputs/cw_factor_tests.csv")
