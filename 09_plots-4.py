import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from functools import partial
from scipy.stats import skew, kurtosis

# Settings
ROLL = 60 # Rolling moment
MIN_VALID = 20 # Min valid observations for moment

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

CRISES = {
    "Black87"    : ("1987-08-31", "1987-12-31"),
    "DotCom"     : ("2000-03-31", "2002-09-30"),
    "GFC"        : ("2007-07-31", "2009-03-31"),
    "Covid"      : ("2020-02-29", "2021-03-31"),
    "RateShock22": ("2022-01-31", "2023-10-31"),
}

# Load series
rets  = pd.read_parquet(OUT / "pnl_series.parquet")
bench = pd.read_parquet(OUT / "benchmarks.parquet")
rets.index = pd.to_datetime(rets.index); bench.index = pd.to_datetime(bench.index)
all_ser = pd.concat([rets, bench], axis=1).sort_index()
all_ser.columns.name = "series"

ACTIVE = rets.columns.tolist() # 6 actively managed series
BENCH = bench.columns.tolist() # 5 benchmark series
GROUPS = [("active", ACTIVE), ("benchmarks", BENCH)]

# Functions
skew_f = partial(skew, bias=False)
kurt_f = partial(kurtosis, fisher=True, bias=False)

def ls(name: str) -> str:
    if "Growth" in name:
        return ":"
    if "Value" in name:
        return "--"
    if name == "SP500_ER":
        return "-"
    return "-"

# Rolling skewness & kurtosis
def roll_moment(series, func):
    return (series.rolling(ROLL, min_periods=MIN_VALID)
            .apply(lambda x: func(pd.Series(x).dropna()), raw=False))

# Compute rolling matrices
roll_var  = all_ser.apply(lambda col: col.rolling(ROLL, min_periods=MIN_VALID).var())
roll_skew = all_ser.apply(lambda col: roll_moment(col, skew_f))
roll_kurt = all_ser.apply(lambda col: roll_moment(col, kurt_f))

pd.concat({"var":  roll_var, "skew": roll_skew, "kurt": roll_kurt}, axis=1)\
  .to_parquet(OUT / f"rolling_moments_{ROLL}m.parquet")

# Crisis shading
def shade_crises(ax, y_pos=0.02, fontsize=7):

    for lbl, (t0, t1) in CRISES.items():

        t0 = pd.to_datetime(t0)
        t1 = pd.to_datetime(t1)

        ax.axvspan(t0, t1, alpha=0.12, color="C0")

        x_mid = t0 + (t1 - t0) / 2
        ax.text(
            x_mid,
            y_pos,
            lbl,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            transform=ax.get_xaxis_transform(),
            bbox=dict(
                facecolor="white",
                alpha=0.6,
                edgecolor="none",
                boxstyle="round,pad=0.15",
            ),
        )

# Figure 1 – full-period returns
# Figure 1a – cumulative returns (linear)
for tag, cols in GROUPS:
    fig, ax = plt.subplots(figsize=(12, 6))

    for col in cols:
        ax.plot(all_ser[col].cumsum(),
                #linestyle=ls(col),
                label=col)

    shade_crises(ax)
    ax.set_title(f"Cumulative Excess Returns – {tag.capitalize()}")
    ax.set_xlabel("")
    ax.set_ylabel("Cumulative return (decimal)")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / f"full_returns_linear_{tag}.png", dpi=300)


# # Figure 1b – cumulative returns (log scale)
# for tag, cols in GROUPS:
#     fig, ax = plt.subplots(figsize=(12, 6))
#     for col in cols:
#         ax.plot((1 + all_ser[col]).cumprod(),
#                 #linestyle=ls(col),
#                 label=col)
#     ax.set_yscale("log")
#     shade_crises(ax)
#     ax.set_title(f"Cumulative Excess Returns – Log Scale – {tag.capitalize()}")
#     ax.set_xlabel("");  ax.set_ylabel("Cumulative return (log axis)")
#     ax.legend(cols, fontsize=8, ncol=3)
#     fig.tight_layout()
#     fig.savefig(OUT / f"full_returns_log_{tag}.png", dpi=300)

# Figure 2 – rolling variance
for tag, cols in GROUPS:
    fig, ax = plt.subplots(figsize=(12, 6))

    for col in cols:
        ax.plot(roll_var[col], linestyle=ls(col), label=col)

    shade_crises(ax)
    ax.set_title(f"{ROLL}-Month Rolling Variance – {tag.capitalize()}")
    ax.set_xlabel("")
    ax.set_ylabel("Variance")
    ax.legend(cols, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / f"rolling_var_{ROLL}m_{tag}.png", dpi=300)


# Figure 3 – rolling skewness
for tag, cols in GROUPS:
    fig, ax = plt.subplots(figsize=(12, 6))

    for col in cols:
        ax.plot(roll_skew[col], linestyle=ls(col), label=col)

    shade_crises(ax)
    ax.set_title(f"{ROLL}-Month Rolling Skewness – {tag.capitalize()}")
    ax.set_xlabel("")
    ax.set_ylabel("Skewness")
    ax.legend(cols, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / f"rolling_skew_{ROLL}m_{tag}.png", dpi=300)


# Figure 4 – rolling kurtosis
for tag, cols in GROUPS:
    fig, ax = plt.subplots(figsize=(12, 6))

    for col in cols:
        ax.plot(roll_kurt[col], linestyle=ls(col), label=col)

    shade_crises(ax)
    ax.set_title(f"{ROLL}-Month Rolling Excess Kurtosis – {tag.capitalize()}")
    ax.set_xlabel("")
    ax.set_ylabel("Excess kurtosis (Fisher)")
    ax.legend(cols, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / f"rolling_kurt_{ROLL}m_{tag}.png", dpi=300)


# Figures 5 - crisis-specific return plots
for tag, cols in GROUPS:

    for lbl, (t0, t1) in CRISES.items():

        sub = all_ser.loc[t0:t1, cols]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))

        for col in cols:
            ax.plot(sub[col].cumsum(), linestyle=ls(col), marker='o', label=col)

        ax.set_title(f"Cumulative Excess Returns – {lbl} – {tag.capitalize()}")
        ax.set_xlabel("")
        ax.set_ylabel("Cumulative return (decimal)")
        ax.legend(cols, fontsize=7, ncol=3)

        fig.tight_layout()
        fig.savefig(OUT / f"returns_{lbl}_{tag}.png", dpi=300)


print("Rolling moments & plots saved to outputs/")

