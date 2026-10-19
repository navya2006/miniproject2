"""Stage E - quadratic and interaction search, plus alternative functional forms.

Each candidate term is added one at a time to a base model; p-values are HC3-robust and are adjusted
for multiple testing (Holm and Benjamini-Hochberg).
"""
import os
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.stats.multitest import multipletests
from .config import FORM, QUAD_TERMS, INTER, PERS, LOC, JOB


def battery(base, terms, data):
    rows = []
    base_names = set(smf.ols(base, data).fit().params.index)
    for t in terms:
        r = smf.ols(base + " + " + t, data).fit(cov_type="HC3", use_t=True)
        k = [n for n in r.params.index if n not in base_names][0]
        rows.append(dict(term=t, name=k, b=float(r.params[k]), se=float(r.bse[k]), t=float(r.tvalues[k]), p=float(r.pvalues[k])))
    o = pd.DataFrame(rows)
    o["p_holm"] = multipletests(o.p, method="holm")[1]
    o["p_bh"] = multipletests(o.p, method="fdr_bh")[1]
    return o


def reset_pair(f, data):
    """Fit statistics and RESET p-values (classical and HC3-robust) for one specification."""
    m = smf.ols(f, data).fit()
    d_ = data.copy()
    d_["yh2"], d_["yh3"] = m.fittedvalues ** 2, m.fittedvalues ** 3
    r = smf.ols(f + " + yh2 + yh3", d_).fit(cov_type="HC3", use_t=True)
    w = r.wald_test("yh2 = 0, yh3 = 0", use_f=True, scalar=True)
    return dict(aic=float(m.aic), bic=float(m.bic), adj_r2=float(m.rsquared_adj),
                reset_F_p=float(linear_reset(m, power=[2, 3], use_f=True).pvalue), reset_rob_p=float(w.pvalue))


def run(ctx):
    S, R = ctx.S, ctx.R
    B1q = battery(FORM["M4"], QUAD_TERMS, S)
    B1i = battery(FORM["M4"], list(INTER), S)
    B1i["hypothesis"] = B1i.term.map(INTER)
    B2i = battery(FORM["M6"], [t for t in INTER if t != "female:married"], S)
    B2i["hypothesis"] = B2i.term.map(INTER)
    B1q.to_csv(os.path.join(ctx.tab_dir, "battery_quadratic.csv"), index=False)
    B1i.to_csv(os.path.join(ctx.tab_dir, "battery_interactions_round1.csv"), index=False)
    B2i.to_csv(os.path.join(ctx.tab_dir, "battery_interactions_round2.csv"), index=False)
    cub = battery(FORM["M6"], ["I(exper**3)", "I(ten**3)", "I(edu**3)"], S)
    R["battery"] = dict(quad=B1q.to_dict(orient="records"), inter1=B1i.to_dict(orient="records"),
                        inter2=B2i.to_dict(orient="records"), cubic=cub.to_dict(orient="records"))
    R["sq_tests"] = {k: dict(b=float(ctx.ROB["M6"].params[k]), p=float(ctx.ROB["M6"].pvalues[k]))
                     for k in ["edu2", "exper2", "ten2"]}

    # alternative functional forms: is the remaining RESET signal removable?
    S["edu_cat"] = pd.cut(S.edu, [-1, 9, 11, 12, 15, 20], labels=["<=9", "10-11", "12", "13-15", "16+"]).astype(str)
    S["univ16"] = (S.edu >= 16).astype(int)
    S["hs12"] = (S.edu >= 12).astype(int)
    rest = "exper + exper2 + ten + ten2 + " + PERS + " + " + LOC + " + " + JOB + " + female:married"
    R["alt_forms"] = {
        "M6: education linear + squared (preferred)": reset_pair(FORM["M6"], S),
        "Education as categorical (<=9, 10-11, 12, 13-15, 16+)": reset_pair("lsal ~ C(edu_cat) + " + rest, S),
        "Education linear + high-school and degree step dummies": reset_pair("lsal ~ edu + hs12 + univ16 + " + rest, S),
        "M6 + cubic experience": reset_pair(FORM["M6"] + " + I(exper**3)", S),
        "M6 + cubic and quartic experience": reset_pair(FORM["M6"] + " + I(exper**3) + I(exper**4)", S),
        "M6 + edu x exper + edu x ten": reset_pair(FORM["M6"] + " + edu:exper + edu:ten", S),
    }
    rj = smf.ols(FORM["M6"] + " + edu:exper + edu:ten", S).fit(cov_type="HC3", use_t=True)
    wj = rj.wald_test("edu:exper = 0, edu:ten = 0", use_f=True, scalar=True)
    R["alt_forms_joint_edu_inter"] = dict(F=float(wj.statistic), p=float(wj.pvalue))
