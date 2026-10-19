"""
MP2 Salary Analysis — reproducible Python script
Input: ITEmp.xlsx (Salary worksheet)
Outputs: cleaned_data.xlsx, figures, model tables, diagnostics.
Run: python analysis.py
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

HERE = Path(__file__).resolve().parent
INPUT = HERE / "ITEmp.xlsx"
OUT = HERE
raw = pd.read_excel(INPUT, sheet_name="Salary")
data = raw.copy()
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Cleaning rules are conservative and documented in the report.
rules = {
    "sal": (0, 100), "edu": (1, 30), "exper": (0, 60), "ten": (0, 60),
    "nonprev": (0, 1), "female": (0, 1), "married": (0, 1), "dep": (0, 20),
    "metro": (0, 1), "norcen": (0, 1), "south": (0, 1), "west": (0, 1),
    "itreq": (0, 1), "itprog": (0, 1), "itqual": (0, 1), "itarch": (0, 1),
    "itai": (0, 1), "itclient": (0, 1), "itrd": (0, 1), "ittech": (0, 1),
    "wrkhm": (0, 1), "momed": (0, 30), "daded": (0, 30), "sib": (0, 20)
}
cleaned = data.copy()
for col, (low, high) in rules.items():
    bad = cleaned[col] < low
    if high is not None:
        bad |= cleaned[col] > high
    cleaned.loc[bad, col] = np.nan

predictors = [
    "sal", "edu", "exper", "ten", "nonprev", "female", "married", "dep",
    "metro", "norcen", "south", "west", "itreq", "itprog", "itqual",
    "itarch", "itai", "itclient", "itrd", "ittech", "wrkhm", "momed",
    "daded", "sib"
]
d = cleaned[predictors].copy()
d["exper_c"] = d["exper"] - d["exper"].mean()
d["ten_c"] = d["ten"] - d["ten"].mean()
d["edu_c"] = d["edu"] - d["edu"].mean()

base = ("sal ~ edu_c + exper_c + ten_c + nonprev + female + married + dep + "
        "metro + norcen + south + west + momed + daded + sib")
job = " + itreq + itprog + itqual + itarch + itai + itclient + itrd + ittech + wrkhm"
formulas = {
    "Model 1": base,
    "Model 2": base + job,
    "Model 3": base + job + " + I(exper_c**2) + I(ten_c**2) + edu_c:exper_c"
}
models = {name: smf.ols(formula, data=d).fit(cov_type="HC3")
          for name, formula in formulas.items()}
d["log_sal"] = np.log(d["sal"])
models["Model 4 (log salary)"] = smf.ols(
    "log_sal ~ " + base.split("~")[1] + job +
    " + I(exper_c**2) + I(ten_c**2) + edu_c:exper_c", data=d
).fit(cov_type="HC3")

cleaned.to_excel(OUT / "cleaned_data.xlsx", index=False)

# Figures
figures = [
    ("salary_distribution.png", "Distribution of cleaned hourly salary",
     d["sal"].dropna(), "Average hourly salary (as recorded)", "Employee count", "hist"),
]
for filename, title, values, xlabel, ylabel, kind in figures:
    plt.figure(figsize=(7, 4.5))
    plt.hist(values, bins=35, edgecolor="black")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(OUT / filename, dpi=180); plt.close()

tmp = d[["exper", "sal"]].dropna()
plt.figure(figsize=(7, 4.5))
plt.scatter(tmp["exper"], tmp["sal"], alpha=.45, s=18)
x = np.linspace(tmp["exper"].min(), tmp["exper"].max(), 100)
plt.plot(x, np.poly1d(np.polyfit(tmp["exper"], tmp["sal"], 1))(x))
plt.xlabel("Years of professional experience"); plt.ylabel("Average hourly salary")
plt.title("Salary and professional experience")
plt.tight_layout(); plt.savefig(OUT / "salary_experience.png", dpi=180); plt.close()

means = d.groupby("female")["sal"].mean()
plt.figure(figsize=(7, 4.5))
plt.bar(["Male (0)", "Female (1)"], [means.get(0, np.nan), means.get(1, np.nan)],
        edgecolor="black")
plt.ylabel("Mean hourly salary"); plt.title("Mean salary by recorded gender indicator")
plt.tight_layout(); plt.savefig(OUT / "salary_gender.png", dpi=180); plt.close()

m3_plain = smf.ols(formulas["Model 3"], data=d).fit()
plt.figure(figsize=(7, 4.5))
plt.scatter(m3_plain.fittedvalues, m3_plain.resid, alpha=.45, s=18)
plt.axhline(0, linewidth=1)
plt.xlabel("Fitted salary"); plt.ylabel("OLS residual")
plt.title("Residuals versus fitted values (Model 3)")
plt.tight_layout(); plt.savefig(OUT / "residuals_fitted.png", dpi=180); plt.close()

comparison = pd.DataFrame([
    {"Model": name, "N": int(m.nobs), "R_squared": m.rsquared,
     "Adjusted_R_squared": m.rsquared_adj, "AIC": m.aic}
    for name, m in models.items()
])
comparison.to_csv(OUT / "model_comparison.csv", index=False)

coef_rows = []
for name, m in models.items():
    ci = m.conf_int()
    for term in m.params.index:
        coef_rows.append({
            "Model": name, "Term": term, "Coefficient": m.params[term],
            "Robust_SE_HC3": m.bse[term], "p_value": m.pvalues[term],
            "CI_95_low": ci.loc[term, 0], "CI_95_high": ci.loc[term, 1]
        })
pd.DataFrame(coef_rows).to_csv(OUT / "regression_coefficients.csv", index=False)

X = m3_plain.model.exog
vif = pd.DataFrame({
    "Term": m3_plain.model.exog_names,
    "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])]
})
vif.to_csv(OUT / "vif_model3.csv", index=False)
bp = het_breuschpagan(m3_plain.resid, m3_plain.model.exog)
reset = linear_reset(m3_plain, power=2, use_f=True)
jb = stats.jarque_bera(m3_plain.resid)
with open(OUT / "diagnostics.txt", "w", encoding="utf-8") as f:
    f.write(f"Model 3 N: {int(m3_plain.nobs)}\\n")
    f.write(f"Breusch-Pagan LM={bp[0]:.4f}, p={bp[1]:.6f}\\n")
    f.write(f"Ramsey RESET F={float(reset.fvalue):.4f}, p={float(reset.pvalue):.8f}\\n")
    f.write(f"Jarque-Bera statistic={jb.statistic:.4f}, p={jb.pvalue:.8g}\\n")
    f.write("HC3 robust standard errors are reported for coefficient inference.\\n")

log = []
for col in predictors:
    original = pd.to_numeric(raw[col], errors="coerce")
    log.append({
        "Variable": col,
        "Original_missing_or_non_numeric": int(original.isna().sum()),
        "After_range_checks_missing": int(cleaned[col].isna().sum()),
        "Newly_flagged": int(cleaned[col].isna().sum() - original.isna().sum())
    })
pd.DataFrame(log).to_csv(OUT / "data_cleaning_log.csv", index=False)
print(comparison.to_string(index=False))
print("\\nDiagnostics saved to diagnostics.txt")
