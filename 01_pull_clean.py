import os, time, numpy as np, pandas as pd, wrds

CRSP_FILE = "crsp_raw.parquet"

# WRDS connection
print("Login to WRDS")
conn = wrds.Connection(wrds_username="maryamahli")

# CRSP pull or load saved file
if os.path.exists(CRSP_FILE):
    print(f"Reading CRSP from {CRSP_FILE}")
    crsp = pd.read_parquet(CRSP_FILE)
else:
    print("Pulling CRSP year by year")
    t0, chunks = time.time(), []
    for yr in range(1973, 2026):
        q = f"""
        SELECT m.permno, m.date, n.shrcd, n.exchcd,
               m.ret, d.dlret, m.prc, m.shrout
        FROM   crsp.msf AS m
        LEFT JOIN crsp.msedelist AS d
               ON m.permno = d.permno AND m.date = d.dlstdt
        JOIN   crsp.msenames AS n
               ON m.permno = n.permno
              AND m.date BETWEEN n.namedt AND n.nameendt
        WHERE  m.date BETWEEN '{yr}-01-31' AND '{yr}-12-31'
          AND  n.shrcd IN (10,11);
        """
        df = conn.raw_sql(q, date_cols=["date"])
        print(f"{yr}: {len(df):,} rows")
        if df.empty:
            continue
        df["mktcap"] = df["prc"].abs() * df["shrout"]
        df["retx"]   = (1 + df["ret"].fillna(0)) * (1 + df["dlret"].fillna(0)) - 1
        chunks.append(df)

    crsp = pd.concat(chunks, ignore_index=True)
    print(f"CRSP concat complete ({len(crsp):,} rows) in {time.time() - t0:.1f} sec")
    crsp.to_parquet(CRSP_FILE)
    print(f"Saved raw CRSP to {CRSP_FILE}")

# Compustat pull
print("Pulling Compustat fundamentals")
q_comp = """
SELECT gvkey, datadate, fyear, seq, ceq, txditc, pstkrv
FROM   comp.funda
WHERE  indfmt='INDL' AND datafmt='STD' AND consol='C' AND popsrc='D'
  AND  datadate BETWEEN '1972-12-31' AND '2024-12-31';
"""
comp = conn.raw_sql(q_comp, date_cols=["datadate"])
comp["be"] = comp["ceq"].fillna(0) + comp["txditc"].fillna(0) - comp["pstkrv"].fillna(0)
comp = comp[comp["be"] > 0]

# Link table
print("Linking Compustat to CRSP")
q_ccm = """
SELECT gvkey, lpermno AS permno,
       linkdt, COALESCE(linkenddt,'9999-12-31') AS linkenddt,
       linktype, linkprim
FROM crsp.ccmxpf_linktable
WHERE linktype IN ('LU','LC') AND linkprim IN ('P','C');
"""
link = conn.raw_sql(q_ccm, date_cols=["linkdt","linkenddt"])
comp = comp.merge(link, on="gvkey", how="inner")
mask = (comp["datadate"] >= comp["linkdt"]) & (comp["datadate"] <= comp["linkenddt"])
comp = comp.loc[mask]

# Book-to-market labels
crsp["year"] = crsp["date"].dt.year
dec = (crsp[crsp["date"].dt.month == 12]
       .loc[:, ["permno","year","mktcap","exchcd"]]
       .rename(columns={"year":"fyear","mktcap":"dec_mktcap"}))
bm = comp.merge(dec, on=["permno","fyear"], how="inner")
bm["bm"] = bm["be"] / bm["dec_mktcap"]

labels = []
for yr, grp in bm[bm["exchcd"] == 1].groupby("fyear"):
    p30, p70 = np.percentile(grp["bm"].dropna(), [30, 70])
    tmp = bm[bm["fyear"] == yr].copy()
    tmp["style"] = "Neutral"
    tmp.loc[tmp["bm"] >= p70, "style"] = "Value"
    tmp.loc[tmp["bm"] <= p30, "style"] = "Growth"
    labels.append(tmp[["permno","fyear","style"]])
labels = pd.concat(labels)

labels["start"] = pd.to_datetime(labels["fyear"].astype(str)) + pd.offsets.MonthEnd(6)
crsp = crsp.merge(labels[["permno","style","start"]],
                  how="left", left_on=["permno","date"], right_on=["permno","start"])
crsp["style"] = crsp.groupby("permno")["style"].ffill()

# French factor pull
def load_french(url, skiprows):
    raw = pd.read_csv(url, skiprows=skiprows)
    raw = raw.rename(columns={"Unnamed: 0": "yyyymm"})
    raw = raw[raw["yyyymm"].fillna("").astype(str).str.strip().str.isdigit()]

    df = raw.melt(id_vars=["yyyymm"], var_name="factor", value_name="ret")

    df["yyyymm"] = df["yyyymm"].astype(str).str.strip().str.slice(0, 6).str.zfill(6)
    df["date"] = pd.to_datetime(df["yyyymm"], format="%Y%m", errors="coerce") + pd.offsets.MonthEnd(0)

    df["ret"] = pd.to_numeric(df["ret"], errors="coerce") / 100
    df = df.dropna(subset=["date", "ret"])

    return df.pivot(index="date", columns="factor", values="ret").reset_index()

print("Pulling French factors")
ff3 = load_french(
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors.CSV",
    skiprows=3)
ff5 = load_french(
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3.CSV",
    skiprows=3)
factors = ff3.merge(ff5[["date","RMW","CMA"]], on="date", how="left").rename(columns={"Mkt-RF": "MKT_RF"})

# Save parquet files
print("Saving Parquet files")
crsp.to_parquet("crsp_clean.parquet")
comp[["permno","fyear","be"]].to_parquet("comp_clean.parquet")
factors.to_parquet("french_factors.parquet", index=False)

print("Script complete. Files saved")
conn.close()
