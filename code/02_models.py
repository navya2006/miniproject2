"""Step 2 - nested models M1-M7, alternatives M8-M11, joint tests, magnitudes, hypotheses."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from common import *
from modelspec import *

d = load_clean()
res = {name: fit(d, cols) for name, cols in SPECS.items()}
m7 = res["M7 + Interactions"]
m7_hc3 = fit(d, SPECS["M7 + Interactions"], cov="HC3")

# ---------------- Table 3: coefficients (classical SE) ----------------
def stars(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""

rows = []
allvars = sorted({v for c in SPECS.values() for v in c})
for v in ["const"] + allvars:
    r1, r2 = {"variable": "Intercept" if v == "const" else v}, {"variable": ""}
    for name, m in res.items():
        short = name.split()[0] + " " + " ".join(name.split()[1:])
        if v in m.params.index:
            r1[short] = f"{m.params[v]:.4f}{stars(m.pvalues[v])}"
            r2[short] = f"({m.bse[v]:.4f})"
        else:
            r1[short] = r2[short] = ""
    rows += [r1, r2]
pd.DataFrame(rows).to_csv(TABLES / "table3_coefficients.csv", index=False)

# ---------------- Table 4: fit statistics ----------------
fitrows = [dict(Model=n, Parameters=int(m.df_model) + 1, R2=round(m.rsquared, 4),
                Adj_R2=round(m.rsquared_adj, 4), AIC=round(m.aic, 4), BIC=round(m.bic, 4),
                F_pvalue=round(m.f_pvalue, 4)) for n, m in res.items()]
pd.DataFrame(fitrows).to_csv(TABLES / "table4_fit_statistics.csv", index=False)

# ---------------- Table 5: joint block tests (M7) ----------------
def joint(m, terms):
    R = np.zeros((len(terms), len(m.params)))
    for i, t in enumerate(terms):
        R[i, list(m.params.index).index(t)] = 1
    f = m.f_test(R)
    return float(np.squeeze(f.fvalue)), float(f.pvalue)

blocks = {
    "Human capital (edu, exper, ten)": HUMAN_C,
    "Demographic controls": DEMOG_C,
    "Geographic controls": GEO_C,
    "IT work-groups": WORKGROUPS,
    "IT divisions": DIVISIONS,
    "Quadratic human capital": QUAD_C,
    "Interactions": INTER_C,
}
brow = []
for k, t in blocks.items():
    F, p = joint(m7, t)
    brow.append(dict(Block=k, F_statistic=round(F, 4), p_value=round(p, 4), Significant_at_5pct=p < .05))
pd.DataFrame(brow).to_csv(TABLES / "table5_joint_tests.csv", index=False)

# ---------------- Table 10: alternative specifications ----------------
def stat_row(name, m):
    return dict(Model=name, N=int(m.nobs), R2=round(m.rsquared, 4), Adj_R2=round(m.rsquared_adj, 4),
                AIC=round(m.aic, 4), BIC=round(m.bic, 4))

alt = [stat_row(n, m) for n, m in res.items()]
alt.append(stat_row("M8 M7 with HC3 robust SE", m7_hc3))
m9 = fit(d, SPECS["M7 + Interactions"], y="sal")
alt.append(stat_row("M9 DV in levels", m9))
dc = prepare(pd.read_csv(COMPLETE_CSV).assign(lnsal=lambda x: np.log(x["sal"])))
m10 = fit(dc, SPECS["M7 + Interactions"])
alt.append(stat_row("M10 Complete-case sample", m10))
lo, hi = d["sal"].quantile([.01, .99])
dt = prepare(pd.read_csv(CLEAN_CSV).query("@lo <= sal <= @hi"))
m11 = fit(dt, SPECS["M7 + Interactions"])
alt.append(stat_row("M11 1% trimmed salary", m11))
pd.DataFrame(alt).to_csv(TABLES / "table10_alternative_specs.csv", index=False)

# ---------------- Table 11: economic magnitudes (HC3) ----------------
b, V = m7_hc3.params, m7_hc3.cov_params()
def lincomb(w):
    """Estimate and two-sided p-value (t distribution) of a linear combination, HC3 covariance."""
    w = pd.Series(w).reindex(b.index).fillna(0.0)
    est = float(w @ b)
    se = float(np.sqrt(w @ V @ w))
    return est, float(2 * (1 - stats.t.cdf(abs(est / se), m7_hc3.df_resid)))


effects = [
    ("Women, unmarried (vs comparable men)", {"female": 1}),
    ("Women, married (vs comparable married men)", {"female": 1, "married:female": 1}),
    ("Married men (vs comparable unmarried men)", {"married": 1}),
    ("Married women (vs comparable unmarried women)", {"married": 1, "married:female": 1}),
    ("One extra year of education (at mean)", {"c_edu": 1}),
    ("One extra year of experience (at mean)", {"c_exper": 1}),
    ("One extra year of tenure (at mean)", {"c_ten": 1}),
    ("Metro location", {"metro": 1}),
    ("IT architecture (vs other work-groups)", {"itarch": 1}),
    ("IT R&D division", {"itrd": 1}),
    ("AI-exposed IT work", {"itai": 1}),
]
mag = []
for n, w in effects:
    est, p = lincomb(w)
    mag.append(dict(Effect=n, Log_points=round(est, 4), Approx_pct_effect=round(100 * (np.exp(est) - 1), 4), p_value=round(p, 4)))
pd.DataFrame(mag).to_csv(TABLES / "table11_economic_magnitudes.csv", index=False)

# gender-gap difference test quoted in Section 5.4 (p = 0.012)
est, p = lincomb({"married:female": 1})
print("married:female HC3 p =", round(p, 4))

# ---------------- Table 12: hypothesis verdicts (HC3) ----------------
def single(term):
    return float(m7_hc3.params[term]), float(m7_hc3.pvalues[term])

def wald(terms):
    R = np.zeros((len(terms), len(b)))
    for i, t in enumerate(terms):
        R[i, list(b.index).index(t)] = 1
    return float(m7_hc3.wald_test(R, use_f=True, scalar=True).pvalue)

H = [
    ("H1", "Education raises pay", "c_edu"),
    ("H2", "Experience raises pay but with diminishing returns", "c_exper"),
    ("H2b", "The experience-pay profile is concave", "c_exper_sq"),
    ("H3", "Firm-specific tenure raises pay", "c_ten"),
    ("H4", "Women earn less than comparable men", "female"),
    ("H5", "Married employees earn a premium", "married"),
    ("H6", "Metro employees earn a premium", "metro"),
    ("H7", "Pay differs across IT work-groups and divisions", WORKGROUPS),
    ("H7b", "R&D and tech-support divisions shift pay", DIVISIONS),
    ("H8", "Non-privileged employees earn less", "nonprev"),
    ("H9", "Working from home is associated with lower pay", "wrkhm"),
    ("H9b", "AI-exposed IT work is associated with lower pay", "itai"),
    ("H10", "Parental education and siblings matter", "c_momed"),
    ("H11", "The education premium differs by gender", "c_edu:female"),
    ("H12", "Education and experience are complements", "c_exper:c_edu"),
    ("H13", "The marriage premium differs by gender", "married:female"),
]
hrows = []
for h, txt, t in H:
    if isinstance(t, list):
        p = wald(t); est = np.nan; term = ", ".join(f"{x} = 0" for x in t)
    else:
        est, p = single(t); term = t
    hrows.append(dict(H=h, Hypothesis=txt, Term=term,
                      Estimate="" if np.isnan(est) else round(est, 4),
                      p_value=round(p, 4), Verdict="Supported" if p < .05 else "Not supported"))
pd.DataFrame(hrows).to_csv(TABLES / "table12_hypothesis_verdicts.csv", index=False)
