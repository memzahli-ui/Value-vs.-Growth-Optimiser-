import pandas as pd
from scipy.stats import skew, kurtosis
from pathlib import Path

OUT = Path("outputs")
bench = OUT / "benchmarks.parquet"

df = pd.read_parquet(bench)
spx = df["SP500_ER"].copy()
spx = spx.loc["1973-01-31":"2025-12-31"].dropna()

var_m = spx.var(ddof=1)
var_a = 12 * var_m

sk = skew(spx.values, bias=False)
ex_kurt = kurtosis(spx.values, fisher=True, bias=False)

print("S&P 500 (excess returns) from 1973-01 to 2025-12")
print(f"Months: {spx.size}")
print(f"Annualised variance: {var_a:.3f}")
print(f"Skewness:           {sk:.2f}")
print(f"Excess kurtosis:    {ex_kurt:.1f}")

pd.DataFrame(
    {
        "var_monthly": [var_m],
        "var_annual": [var_a],
        "skew": [sk],
        "excess_kurtosis": [ex_kurt],
        "start": [spx.index.min()],
        "end": [spx.index.max()],
        "n_months": [spx.size],
    }
).to_csv(OUT / "sp500_moments_summary.csv", index=False)
print(f"\nSaved summary to {OUT/'sp500_moments_summary.csv'}")
