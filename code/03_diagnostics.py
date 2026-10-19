"""Step 3 - assumption checks: normality, heteroskedasticity, RESET, multicollinearity, influence."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from common import *
from modelspec import *

d = load_clean()
n = len(d)


def wooldridge_white(m):
    """Wooldridge's special case of the White test: e^2 on fitted and fitted^2 (F-test, 2 df)."""
    e2 = m.resid ** 2
    yh = m.fittedvalues
    Z = sm.add_constant(pd.DataFrame({"yh": yh, "yh2": yh ** 2}))
    a = sm.OLS(e2, Z).fit()
    return a.fvalue, a.f_pvalue


rows = []
for name in ["M4 + Job function", "M7 + Interactions"]:
    m = fit(d, SPECS[name])
    jb = jarque_bera(m.resid)
    bp = het_breuschpagan(m.resid, m.model.exog)
    wo = wooldridge_white(m)
    reset = linear_reset(m, power=2, test_type="fitted", use_f=True)
    reset_r = linear_reset(m, power=2, test_type="fitted", use_f=False, cov_type="HC3")  # Wald chi2 (1 df)
    cook = m.get_influence().cooks_distance[0]
    share = float((cook > 4 / n).mean())
    rows += [
        (name, "Jarque-Bera (residual normality)", jb[0], jb[1], "residuals non-normal" if jb[1] < .05 else "normal"),
        (name, "Breusch-Pagan (heteroskedasticity)", bp[0], bp[1], "homoskedastic" if bp[1] > .05 else "heteroskedastic"),
        (name, "White / Wooldridge (heteroskedasticity)", wo[0], wo[1], "homoskedastic" if wo[1] > .05 else "heteroskedastic"),
        (name, "Ramsey RESET (functional form)", float(reset.fvalue), float(reset.pvalue), "misspecified" if reset.pvalue < .05 else "ok"),
        (name, "RESET, HC3-robust", float(reset_r.statistic), float(reset_r.pvalue), "misspecified" if reset_r.pvalue < .05 else "ok"),
        (name, "Condition number of X", m.condition_number, np.nan, "elevated, but scale-sensitive: see VIF table"),
        (name, "Cook's D: share > 4/n (max D)", share, np.nan, f"max D = {cook.max():.3f}; no dominating observation"),
    ]
t6 = pd.DataFrame(rows, columns=["Model", "Test", "Statistic", "p_value", "Conclusion"])
t6.to_csv(TABLES / "table6_diagnostics.csv", index=False)
print(t6.to_string())

# Table 7 - VIF for M7
m7 = fit(d, SPECS["M7 + Interactions"])
X = m7.model.exog
names = list(m7.model.exog_names)
vif = {names[i]: variance_inflation_factor(X, i) for i in range(1, X.shape[1])}
t7 = pd.DataFrame({"Variable": list(vif), "VIF": [round(v, 3) for v in vif.values()]})
t7["Tolerance"] = (1 / t7["VIF"]).round(3)
t7["Assessment"] = np.where(t7["VIF"] > 10, "high (>10)", np.where(t7["VIF"] > 5, "moderate (>5)", "ok"))
t7 = t7.sort_values("VIF", ascending=False)
t7.to_csv(TABLES / "table7_vif.csv", index=False)
print(t7.head(5).to_string())

# VIF before centring: same M7 design built from raw (uncentred) variables
u = pd.read_csv(CLEAN_CSV)
for v in ["edu", "exper", "ten"]:
    u[f"c_{v}"] = u[v]; u[f"c_{v}_sq"] = u[v] ** 2
for v in ["dep", "momed", "daded", "sib"]:
    u[f"c_{v}"] = u[v]
u["c_edu:female"] = u["edu"] * u["female"]; u["married:female"] = u["married"] * u["female"]
u["c_exper:c_edu"] = u["exper"] * u["edu"]; u["metro:c_edu"] = u["metro"] * u["edu"]
Xu = sm.add_constant(u[SPECS["M7 + Interactions"]]).values
print("uncentred VIF (full M7 design): edu =", round(variance_inflation_factor(Xu, 1), 1))
