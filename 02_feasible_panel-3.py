import pandas as pd
from pathlib import Path

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# Load files
print("Loading crsp_clean.parquet and french_factors.parquet")
crsp = pd.read_parquet("crsp_clean.parquet")
factors   = pd.read_parquet("french_factors.parquet")

if factors.index.name == "date":
    factors = factors.reset_index()

if factors.columns.duplicated().any():
    factors = factors.loc[:, ~factors.columns.duplicated()]

# Merge and compute excess return
print("Merging factors onto CRSP")
crsp = crsp.merge(factors[["date", "RF"]], on="date", how="left")
crsp["rexcess"] = crsp["retx"] - crsp["RF"]

# Merge the remaining factor columns
factor_cols = [c for c in factors.columns if c not in ("date", "RF")]
crsp_f = crsp.merge(factors[["date"] + factor_cols], on="date", how="left")

front = ["permno", "date", "rexcess", "RF"] + factor_cols
crsp_f = crsp_f[[*front, *[c for c in crsp_f.columns if c not in front]]]

crsp_f.to_parquet(OUT_DIR / "crsp_factors.parquet")
print("Saved crsp_factors.parquet  →  shape", crsp_f.shape)

# Find feasible stocks
print("Scanning for PERMNOs with at least 60 factor complete months")
req_cols = ["rexcess", "MKT_RF", "SMB", "HML", "RMW", "CMA"]

mask_complete = crsp_f[req_cols].notna().all(axis=1)
feasible_permnos = (crsp_f[mask_complete].groupby("permno")["date"].count().pipe(lambda s: s[s >= 60]).index.astype(int)).tolist()

Path(OUT_DIR / "permnos_feasible.txt").write_text("\n".join(map(str, feasible_permnos)))

# Summary
print("\n*** Summary ***")
print(f"Total CRSP rows: {len(crsp_f):,}")
print(f"\nPERMNOs with at least 60 clean months: {len(feasible_permnos):,}")
print("First 10 viable PERMNOs:", feasible_permnos[:10])
print("\nOutputs written:")
print("outputs/crsp_factors.parquet")
print("outputs/permnos_feasible.txt")