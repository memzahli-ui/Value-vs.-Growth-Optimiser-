import numpy as np, pandas as pd
from pathlib import Path

np.random.seed(42)  # reproducibility
OUT = Path("outputs")
d = pd.read_csv(OUT / "ownPL_tests_detail.csv", parse_dates=["end_date"])

BLOCK = 12  # months
REPS = 2000

def moving_blocks(T, block):
    return np.arange(0, T - block + 1)

def draw_series(errs, block):
    T = len(errs)
    starts = moving_blocks(T, block)

    idx = []
    while len(idx) < T:
        s = np.random.choice(starts)
        idx.extend(range(s, min(s + block, T)))
    return errs[np.array(idx[:T])]


rows = []
for (style, model), g in d.groupby(["style", "model"]):
    errs = g[["rmse", "cvar5"]].to_numpy()
    T = len(errs)

    def _stat(x, axis=0):
        return np.mean(x, axis=axis)

    boot = np.empty((REPS, 2))
    for r in range(REPS):
        boot[r] = _stat(draw_series(errs, BLOCK), axis=0)

    for j, metric in enumerate(["mean_RMSE", "mean_CVaR5"]):
        low, med, high = np.percentile(boot[:, j], [2.5, 50, 97.5])
        rows.append({"style": style, "model": model,
                     "metric": metric, "lower": low,
                     "median": med, "upper": high})

pd.DataFrame(rows).to_csv(OUT / "bootstrap_CI.csv", index=False)
print("Saved results: outputs/bootstrap_CI.csv")
