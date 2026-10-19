"""Step 4 - specification checks: partial-residual tests, CV benchmark, influence, robustness of quadratics."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from common import *
from modelspec import *

d = load_clean()
m5 = fit(d, SPECS["M5 Full main effects"])
m7 = fit(d, SPECS["M7 + Interactions"])

# ---- Table 8: partial-residual test of unmodelled curvature (square and cube), preferred model M7 ----
rows = []
for v in ["c_edu", "c_exper", "c_ten", "c_dep", "c_momed", "c_daded", "c_sib"]:
    pr = m7.resid + m7.params[v] * d[v]
    Z0 = sm.add_constant(pd.DataFrame({v: d[v]}))
    Z1 = sm.add_constant(pd.DataFrame({v: d[v], "sq": d[v] ** 2, "cu": d[v] ** 3}))
    a0, a1 = sm.OLS(pr, Z0).fit(), sm.OLS(pr, Z1).fit()
    F, p, _ = a1.compare_f_test(a0)
    rows.append(dict(Regressor=v, F_statistic=round(F, 4), p_value=round(p, 4), Evidence_of_nonlinearity=p < .05))
t8 = pd.DataFrame(rows); t8.to_csv(TABLES / "table8_partial_residual_tests.csv", index=False)
print(t8.to_string(index=False))

# ---- Table 9: five-fold CV R^2 ----
raw_cols = ["edu", "exper", "ten", "female", "married", "dep", "nonprev"] + GEO + JOB + FAMILY
kf = KFold(5, shuffle=True, random_state=42)
y = d["lnsal"]
cv = {
    "OLS, linear in X": cross_val_score(LinearRegression(), d[raw_cols], y, cv=kf, scoring="r2").mean(),
    "OLS, quadratics + interactions (M7 design)": cross_val_score(LinearRegression(), d[SPECS["M7 + Interactions"]], y, cv=kf, scoring="r2").mean(),
    "Gradient boosting (flexible benchmark, raw X)": cross_val_score(GradientBoostingRegressor(random_state=42), d[raw_cols], y, cv=kf, scoring="r2").mean(),
}
t9 = pd.DataFrame({"Model": list(cv), "CV_R2": [round(v, 4) for v in cv.values()]})
t9.to_csv(TABLES / "table9_cv_benchmark.csv", index=False)
print(t9.to_string(index=False))

# ---- Influence: Cook's D, dropping the ten most influential observations ----
cook = m7.get_influence().cooks_distance[0]
top10 = np.argsort(cook)[-10:]
keep = d.drop(d.index[top10])
m7_drop = fit(prepare(keep[[c for c in pd.read_csv(CLEAN_CSV).columns]]), SPECS["M7 + Interactions"])
print("max Cook D", cook.max().round(3), "| share > 4/n", (cook > 4 / len(d)).mean().round(3),
      "| adj R2 full vs drop-10:", round(m7.rsquared_adj, 3), round(m7_drop.rsquared_adj, 3))

# ---- Turning points of the quadratics (evaluated with other terms at their means) ----
b = m7.params
tp_exper = d["exper"].mean() - b["c_exper"] / (2 * b["c_exper_sq"])
tp_ten = d["ten"].mean() - b["c_ten"] / (2 * b["c_ten_sq"])
print("turning point experience (years):", round(tp_exper, 1), "| tenure:", round(tp_ten, 1),
      "| median tenure:", d["ten"].median())

# ---- Robustness of the education quadratic ----
def edu_sq_p(frame, label):
    f = prepare(frame)
    mm = fit(f, SPECS["M7 + Interactions"])
    mh = fit(f, SPECS["M7 + Interactions"], cov="HC3")
    return dict(Sample=label, N=len(f), coef=round(mm.params["c_edu_sq"], 4),
                p_classical=round(mm.pvalues["c_edu_sq"], 4), p_HC3=round(mh.pvalues["c_edu_sq"], 4))

base = pd.read_csv(CLEAN_CSV)
lo, hi = base["sal"].quantile([.01, .99])
rob = pd.DataFrame([
    edu_sq_p(base, "Full sample"),
    edu_sq_p(base[base["edu"] >= 8], "Drop edu < 8 years"),
    edu_sq_p(base[(base["sal"] >= lo) & (base["sal"] <= hi)], "Trim top/bottom 1% of salary"),
])
rob.to_csv(TABLES / "table_edu_quadratic_robustness.csv", index=False)
print(rob.to_string(index=False))

# ---- Section 5.2 / 5.5 supporting numbers ----
m4 = fit(d, SPECS["M4 + Job function"])
print("edu coef M4 vs M5:", round(m4.params["c_edu"], 4), round(fit(d, SPECS["M5 Full main effects"]).params["c_edu"], 4))
R = np.zeros((3, len(b))); idx = list(b.index)
for i, t in enumerate(FAM_C): R[i, idx.index(t)] = 1
f = m7.f_test(R); print("family background joint F, p:", round(float(np.squeeze(f.fvalue)), 3), round(float(f.pvalue), 4))
print("skewness sal / lnsal:", round(stats_skew := float(pd.Series(d["sal"]).skew()), 2), round(float(d["lnsal"].skew()), 2))
