import time
from pathlib import Path
import pandas as pd, numpy as np, cvxpy as cp
import joblib

OUT = Path("outputs")
COV_DIR = OUT / "cov_mats"
WEIGHT_DIR = OUT / "weights"
WEIGHT_DIR.mkdir(exist_ok=True)

MU_FILE = OUT / "mu_vectors.parquet"
PANEL_FILE = OUT / "crsp_factors.parquet"

MODELS = {"CAPM": "mu_capm", "FF3": "mu_ff3", "FF5": "mu_ff5"}
TARGET = 0.005 # ≥0.5% monthly expected excess return
SOLVER = "ECOS"

# Load μ-vectors and style panel
mu_all = pd.read_parquet(MU_FILE)
panel = pd.read_parquet(PANEL_FILE)[["permno", "date", "style"]]
panel["date"] = pd.to_datetime(panel["date"]) + pd.offsets.MonthEnd(0)

print("μ-vector rows:", len(mu_all))
print("Panel rows:", len(panel))

def optimise(mu_vec, Sigma):
    n = len(mu_vec)
    w = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), [cp.sum(w) == 1, w >= 0, mu_vec @ w >= TARGET])
    prob.solve(solver=SOLVER, warm_start=True)
    return None if w.value is None else w.value.round(10)

start = time.time()

# Optimise weights for each style and model
for style in ["Value", "Growth"]:

    style_mask = panel[panel["style"] == style][["permno", "date"]]
    writers = {m: [] for m in MODELS}

    cov_files = sorted(COV_DIR.glob(f"Σ_{style}_*.joblib"))
    if not cov_files:
        print(f"[{style}] no covariance matrices found; skipping.")
        continue

    print(f"\n[{style}] {len(cov_files)} month-ends to optimise")
    for k, fp in enumerate(cov_files, 1):
        date_str = fp.stem.split("_")[-1]
        date     = pd.to_datetime(date_str) + pd.offsets.MonthEnd(0)

        payload  = joblib.load(fp)
        permnos  = payload["permnos"]
        Sigma    = payload["cov"].astype(float)

        live = set(style_mask[style_mask["date"] == date]["permno"])
        keep_idx = [i for i, p in enumerate(permnos) if p in live]
        if len(keep_idx) < 2:
            continue
        Sigma = Sigma[np.ix_(keep_idx, keep_idx)]
        permnos = [permnos[i] for i in keep_idx]

        mu_row = mu_all[mu_all["date"] == date].drop_duplicates("permno", keep="last").set_index("permno").reindex(permnos)

        for mdl, col in MODELS.items():
            mu_vec = mu_row[col].values
            if np.isnan(mu_vec).any():
                continue

            w = optimise(mu_vec, Sigma)
            if w is None:
                continue

            writers[mdl].extend({"date": date, "permno": p, "weight": w_i} for p, w_i in zip(permnos, w) if w_i > 0)

        if k % 24 == 0 or k == len(cov_files):
            elapsed = (time.time() - start) / 60
            pct = k / len(cov_files)
            print(f"{k:4}/{len(cov_files)} months, ({pct:4.1%}) elapsed {elapsed:5.1f} min")

    for mdl, rows in writers.items():
        if rows:
            out_path = WEIGHT_DIR / f"weights_{style}_{mdl}.parquet"
            pd.DataFrame(rows).to_parquet(out_path, index=False)
            print(f"  [{style}] wrote {len(rows):,} rows → {out_path}")

print(f"\nAll optimisations finished in {(time.time()-start)/60:5.1f} minutes.")