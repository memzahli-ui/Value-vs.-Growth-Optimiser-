import time
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.covariance import LedoitWolf
import joblib

OUT = Path("outputs")
COV_DIR = OUT / "cov_mats"
COV_DIR.mkdir(parents=True, exist_ok=True)

PANEL_PARQUET = OUT / "crsp_factors.parquet"
FEASIBLE_TXT = OUT / "permnos_feasible.txt"

WINDOW = 60
STYLE_BUCKETS = ["Value", "Growth"]

def shrink_cov(ret_matrix: pd.DataFrame) -> pd.DataFrame:
    lw = LedoitWolf().fit(ret_matrix)
    cov = pd.DataFrame(lw.covariance_, index=ret_matrix.columns, columns=ret_matrix.columns)
    return cov

def already_done(style: str, date: pd.Timestamp) -> bool:
    return (COV_DIR / f"Σ_{style}_{date:%Y%m%d}.joblib").exists()

def save_cov(style: str, date: pd.Timestamp, perm_list: list[int], cov_df: pd.DataFrame) -> None:
    payload = {
        "permnos": perm_list,
        "cov": cov_df.values.astype(np.float32)
    }
    joblib.dump(payload, COV_DIR / f"Σ_{style}_{date:%Y%m%d}.joblib", compress=3)

# Load panel
print("Loading CRSP-factor panel")
panel = pd.read_parquet(PANEL_PARQUET)
feas_permnos = {int(x) for x in Path(FEASIBLE_TXT).read_text().split()}
panel = panel[panel["permno"].isin(feas_permnos)]
panel["date"] = pd.to_datetime(panel["date"]) + pd.offsets.MonthEnd(0)

# Rolling loop
start_time = time.time()

for style in STYLE_BUCKETS:
    g_sty = panel[panel["style"] == style].copy()
    if g_sty.empty:
        print(f"[{style}] no rows – skipping")
        continue

    unique_dates = sorted(g_sty["date"].unique())
    print(f"[{style}] {len(unique_dates)} month-ends to process")

    for i, date in enumerate(unique_dates):
        if i < WINDOW - 1:
            continue
        if already_done(style, date):
            continue

        win = g_sty[g_sty["date"].between(unique_dates[i-WINDOW+1], date)]
        ret_mat = win.pivot_table(index="date", columns="permno", values="rexcess")

        ret_mat = ret_mat.dropna(axis=1)
        if ret_mat.shape[1] < 2:
            continue

        cov_df = shrink_cov(ret_mat)
        save_cov(style, date, cov_df.columns.tolist(), cov_df)

        if (i+1) % 24 == 0 or i == len(unique_dates) - 1:
            elapsed = time.time() - start_time
            done = i + 1
            pct = done / len(unique_dates)
            eta = (len(unique_dates)-done) * (elapsed / done)
            print(f"[{style}] {done:4}/{len(unique_dates)} ({pct:4.1%}) | elapsed {elapsed / 60:5.1f} min | ETA {eta / 60:5.1f} min")

print("\nAll covariance matrices saved to outputs/cov_mats/")
