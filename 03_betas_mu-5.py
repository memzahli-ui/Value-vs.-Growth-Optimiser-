import time, sys
from pathlib import Path
import pandas as pd, statsmodels.api as sm
import pyarrow.parquet as pq

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
PARTS_DIR = OUT / "beta_parts"
PARTS_DIR.mkdir(exist_ok=True)

CRSP_F = OUT / "crsp_factors.parquet"
FEAS_TXT = OUT / "permnos_feasible.txt"
DONE_TXT = OUT / "permnos_done.txt"
MU_PQ = OUT / "mu_vectors.parquet"

WINDOW = 36
LAMBDA_ROLL = 180
REQUIRED = ["rexcess","MKT_RF","SMB","HML","RMW","CMA"]
factor_sets = {
    "CAPM": ["MKT_RF"],
    "FF3" : ["MKT_RF","SMB","HML"],
    "FF5" : ["MKT_RF","SMB","HML","RMW","CMA"],
}

# Load data
crsp_f = pd.read_parquet(CRSP_F)
feas_permnos = {int(x) for x in Path(FEAS_TXT).read_text().split()}
done_permnos = set()
if DONE_TXT.exists():
    done_permnos = {int(x) for x in Path(DONE_TXT).read_text().split()}

todo_permnos = sorted(feas_permnos - done_permnos)
print(f"Feasible {len(feas_permnos):,}; done {len(done_permnos):,}; left {len(todo_permnos):,}")

t0 = time.time()

try:
    for idx, permno in enumerate(todo_permnos, 1):
        g_clean = crsp_f[crsp_f.permno == permno].sort_values("date").dropna(subset=REQUIRED).reset_index(drop=True)
        if len(g_clean) < WINDOW:
            with DONE_TXT.open("a") as f: f.write(f"{permno}\n")
            continue

        rows = []
        for end_ix in range(WINDOW-1, len(g_clean)):
            win = g_clean.iloc[end_ix-WINDOW+1:end_ix+1]
            row = {"permno": permno, "date": win.iloc[-1]["date"]}
            for mdl, cols in factor_sets.items():
                res = sm.OLS(win["rexcess"], sm.add_constant(win[cols])).fit()
                for c in cols:
                    row[f"beta_{mdl}_{c}"] = res.params[c]
            rows.append(row)

        if rows:
            pd.DataFrame(rows).to_parquet(PARTS_DIR / f"permno_{permno}.parquet", index=False, compression="snappy")
            #print(f"permno {permno}: {len(rows):,} rows")

        with DONE_TXT.open("a") as f: f.write(f"{permno}\n")

        if idx % 250 == 0 or idx == len(todo_permnos):
            elapsed = time.time() - t0
            eta = (len(todo_permnos) - idx) * (elapsed / idx)
            pct = idx / len(todo_permnos)

            print(f"  • {idx:>6}/{len(todo_permnos):,} permnos "
                  f"({pct:4.1%}) | elapsed {elapsed / 60:5.1f} min | ETA {eta / 60:5.1f} min")


except KeyboardInterrupt:
    print("\nInterrupted – progress saved. Re-run to resume.")
    sys.exit(0)

print("\nRolling betas finished: files in outputs/beta_parts/")

# μ-vectors
if MU_PQ.exists():
    print("μ-vectors already exist")
    sys.exit(0)

print("Building μ-vectors")

beta_ds = pq.ParquetDataset(PARTS_DIR)
betas = beta_ds.read().to_pandas()

factors = (pd.read_parquet("french_factors.parquet").assign(date=lambda d: pd.to_datetime(d["date"]) + pd.offsets.MonthEnd(0))
             .sort_values("date").set_index("date"))

for col in ["MKT_RF","SMB","HML","RMW","CMA","RF"]:
    factors[col] = pd.to_numeric(factors[col], errors="coerce")

lambda_parts = []
for col in ["MKT_RF","SMB","HML","RMW","CMA","RF"]:
    part = factors[[col]].rolling(LAMBDA_ROLL, min_periods=LAMBDA_ROLL).mean()
    lambda_parts.append(part)

lambda_bar = pd.concat(lambda_parts, axis=1).dropna(subset=["MKT_RF"])

betas["date"] = pd.to_datetime(betas["date"]) + pd.offsets.MonthEnd(0)
betas = betas.merge(lambda_bar, left_on="date", right_index=True, how="left")

mu_rows = []
for r in betas.itertuples():
    mu_rows.append({
        "permno": r.permno, "date": r.date,
        "mu_capm": r.beta_CAPM_MKT_RF * r.MKT_RF + r.RF,
        "mu_ff3":  r.beta_FF3_MKT_RF*r.MKT_RF + r.beta_FF3_SMB*r.SMB + r.beta_FF3_HML*r.HML + r.RF,
        "mu_ff5":  (r.beta_FF5_MKT_RF*r.MKT_RF + r.beta_FF5_SMB*r.SMB + r.beta_FF5_HML*r.HML + r.beta_FF5_RMW*r.RMW +
                    r.beta_FF5_CMA*r.CMA + r.RF)
    })

mu_df = pd.DataFrame(mu_rows).drop_duplicates(["permno", "date"], keep="last")
pd.DataFrame(mu_rows).to_parquet(MU_PQ, index=False)
print(f"Saved μ-vectors to {MU_PQ} ({len(mu_rows):,} rows)")
