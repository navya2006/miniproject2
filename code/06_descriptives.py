"""Step 6 - descriptive statistics and group means (referenced in Section 3)."""
import pandas as pd
from common import *

d = pd.read_csv(CLEAN_CSV)
desc = d[ALL_VARS].describe().T
desc["skew"] = d[ALL_VARS].skew()
desc.round(4).to_csv(TABLES / "descriptive_stats.csv")

rows = []
for v in BINARY:
    g = d.groupby(v)["sal"].agg(["mean", "median", "count"])
    for lvl, r in g.iterrows():
        rows.append(dict(variable=v, value=int(lvl), mean_salary=round(r["mean"], 3), median_salary=round(r["median"], 3), n=int(r["count"])))
pd.DataFrame(rows).to_csv(TABLES / "group_means.csv", index=False)

c = d[["lnsal"] + [v for v in ALL_VARS if v != "sal"]].corr()["lnsal"].drop("lnsal").sort_values(key=abs, ascending=False)
print("top correlations with log salary:\n", c.head(5).round(2).to_string())
print("edu-itrd:", round(d.edu.corr(d.itrd), 2), "| exper-ten:", round(d.exper.corr(d.ten), 2))
print("salary range:", d.sal.min(), d.sal.max(), "| skew sal, ln:", round(d.sal.skew(), 2), round(d.lnsal.skew(), 2))
