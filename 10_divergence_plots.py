from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("outputs")
WEI_DIR = OUT / "weights"


# Rolling volatility of factor returns
fac = (
    pd.read_parquet(OUT / "crsp_factors.parquet",
                    columns=["date", "MKT_RF", "RMW", "CMA", "HML", "SMB"])
      .assign(date=lambda d: pd.to_datetime(d["date"]) + pd.offsets.MonthEnd(0))
      .set_index("date")
      .groupby("date").first()
)

ts_vol = fac.rolling(window=12, min_periods=3).std()

fig, ax = plt.subplots(figsize=(10, 4))
ts_vol.plot(ax=ax)
ax.set_title("12‑month rolling σ of factor returns")
ax.set_ylabel("σ")
ax.set_xlabel("")
fig.tight_layout()
fig.savefig(OUT / "factor_vol.png", dpi=300)
print("Saved plot: factor_vol.png")


# Cross-model correlation of μ‑vectors
MU_PARQ = OUT / "mu_vectors.parquet"

mu = pd.read_parquet(MU_PARQ)
mu["date"] = pd.to_datetime(mu["date"]) + pd.offsets.MonthEnd(0)

def corr_row(df):
    return pd.Series({
        "rho_CAPM_FF3": df["mu_capm"].corr(df["mu_ff3"]),
        "rho_CAPM_FF5": df["mu_capm"].corr(df["mu_ff5"]),
    })

corrs = mu.groupby("date", observed=True).apply(corr_row, include_groups=False).rolling(36, min_periods=24).mean()

ax = corrs.plot(title="Rolling 36‑month correlation of μ‑vectors", figsize=(10,4))
ax.set_ylabel("ρ")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig(OUT / "mu_corr.png", dpi=300)
print("Saved plot: mu_corr.png")


# Cross‑model correlation of portfolio returns
pnl = pd.read_parquet(OUT / "pnl_series.parquet")
pnl.index = pd.to_datetime(pnl.index) + pd.offsets.MonthEnd(0)

pairs = {
    # Growth
    "rho_CAPM_FF3_Growth": ["Growth_CAPM", "Growth_FF3"],
    "rho_CAPM_FF5_Growth": ["Growth_CAPM", "Growth_FF5"],
    # Value
    "rho_CAPM_FF3_Value":  ["Value_CAPM",  "Value_FF3"],
    "rho_CAPM_FF5_Value":  ["Value_CAPM",  "Value_FF5"],
}

rho60 = {}
for name, (a, b) in pairs.items():
    rho60[name] = pnl[[a, b]].rolling(60, min_periods=36).corr().unstack()[a][b]  # pick the off‑diagonal element

rho60 = pd.DataFrame(rho60)
ax = rho60.plot(title="Rolling 60‑month ρ of portfolio excess return (Growth & Value)", figsize=(10,4))
ax.set_ylabel("ρ")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig(OUT / "return_corr.png", dpi=300)
print("Saved plot: return_corr.png")


# Maximum absolute weight difference
weight_files = sorted(WEI_DIR.glob("weights_*.parquet"))

frames = []
for fp in weight_files:
    _, style, model = fp.stem.split("_", maxsplit=2)
    tmp = pd.read_parquet(fp)
    tmp["style"] = style  # Value / Growth
    tmp["model"] = model  # CAPM / FF3 / FF5
    frames.append(tmp)

w = (
    pd.concat(frames, ignore_index=True)
        .assign(date=lambda d: pd.to_datetime(d["date"]) + pd.offsets.MonthEnd(0))
        .pivot_table(index=["permno", "date", "style"], # 3 level index
                     columns="model", # CAPM / FF3 / FF5 columns
                     values="weight")
        .reset_index()
)

def max_diff(df):
    return pd.Series({
        "diff_CAPM_FF3": (df["FF3"] - df["CAPM"]).abs().max(),
        "diff_CAPM_FF5": (df["FF5"] - df["CAPM"]).abs().max(),
    })


diff = w.groupby(["date", "style"], observed=True).apply(max_diff, include_groups=False)  # MultiIndex columns

to_plot = diff.unstack("style")  # (metric, style) columns
to_plot.columns = [f"{metric}_{style}" for metric, style in to_plot.columns]

ax = to_plot.plot(title="Max |weight difference| vs CAPM", figsize=(10, 4))
ax.set_ylabel("Max |weight difference|")
ax.set_xlabel("")
ax.legend(fontsize=8, ncol=2)

plt.tight_layout()
plt.savefig(OUT / "weight_diff.png", dpi=300)

print("Saved plot: weight_diff.png")
