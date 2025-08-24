import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path("outputs")
WEIGHTS_DIR = OUT_DIR / "weights"
WINDOW = 60
STYLES = ["Growth", "Value"]
MODELS = ["CAPM", "FF3", "FF5"]

# Portfolio‑level forecasts
mu = (pd.read_parquet(OUT_DIR / "mu_vectors.parquet")
        .rename(columns={"mu_capm": "CAPM",
                         "mu_ff3" : "FF3",
                         "mu_ff5" : "FF5"})
        .melt(id_vars=["permno", "date"], var_name="model", value_name="mu").dropna())

rf = pd.read_parquet(OUT_DIR / "crsp_factors.parquet")[["date", "RF"]].drop_duplicates()

rows = []
for style in STYLES:
    for model in MODELS:
        w_path = WEIGHTS_DIR / f"weights_{style}_{model}.parquet"

        w = pd.read_parquet(w_path)  # date, permno, weight
        df = w.merge(mu[mu.model == model], on=["permno", "date"]).merge(rf, on="date", how="left")

        df["excess_mu"] = df["mu"] - df["RF"]
        port = (df.assign(prod=lambda d: d["weight"] * d["excess_mu"])
                  .groupby("date")["prod"].sum()
                  .rename(f"{style} {model}_FORECAST_EXCESS")
                  .reset_index())

        rows.append(port)

fc = rows[0]
for r in rows[1:]:
    fc = fc.merge(r, on="date", how="outer")

fc["date"] = pd.to_datetime(fc["date"]) + pd.offsets.MonthEnd(1)  # align to t+1
fc = fc.sort_values("date")
fc.to_parquet(OUT_DIR / "model_port_mu_forecasts.parquet", index=False)
print("Forecasts saved: outputs/model_port_mu_forecasts.parquet")

# Rolling RMSE & CVaR
pnl = pd.read_parquet(OUT_DIR / "pnl_series.parquet").reset_index(names="date")
pnl["date"] = pd.to_datetime(pnl["date"])
pnl.columns = ["date"] + [c.replace("_", " ") for c in pnl.columns[1:]]

frame = pnl.merge(fc, on="date", how="inner").set_index("date").sort_index()

rmse = lambda x: float(np.sqrt(np.mean(x**2)))
cvar5 = lambda x: float(x[x <= np.quantile(x, 0.05)].mean())

records = []
for style in STYLES:
    for model in MODELS:
        real = f"{style} {model}"
        pred = f"{style} {model}_FORECAST_EXCESS"
        if real not in frame or pred not in frame:
            continue

        ser = frame[[real, pred]].dropna()
        for end in range(WINDOW, len(ser)):
            win = ser.iloc[end - WINDOW:end]
            err = win[real] - win[pred]
            records.append({"style": style, "model": model, "end_date": win.index[-1], "rmse": rmse(err), "cvar5": cvar5(err)})

detail = pd.DataFrame(records)
detail.to_csv(OUT_DIR / "ownPL_tests_detail.csv", index=False)

(detail.groupby(["style", "model"]).agg(mean_RMSE=("rmse", "mean"), mean_CVaR5=("cvar5", "mean"))
       .reset_index().to_csv(OUT_DIR / "ownPL_tests_avg.csv", index=False))

print("Details saved: outputs/ownPL_tests_detail.csv\nAverages saved: outputs/ownPL_tests_avg.csv")
