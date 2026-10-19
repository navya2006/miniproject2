"""Stage D - nested OLS models M1-M6 on ln(hourly salary), HC3-robust inference.

Also: nested-model tests, block Wald tests, the bootstrap mediation test for education,
and the family-background test on the smaller sample.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from .config import (SEED, FORM, LABEL, stars, PERS, LOC, JOB, REGION, FUNC, DIV, FAM)


def fit(formula, data):
    """Classical OLS fit and HC3-robust fit (t-distribution) of the same formula."""
    ols = smf.ols(formula, data).fit()
    rob = smf.ols(formula, data).fit(cov_type="HC3", use_t=True)
    return ols, rob


def coef_table(ols, rob):
    ci = rob.conf_int()
    t = pd.DataFrame({"term": rob.params.index, "b": rob.params.values, "se_hc3": rob.bse.values,
                      "t_hc3": rob.tvalues.values, "p_hc3": rob.pvalues.values, "lo": ci[0].values,
                      "hi": ci[1].values, "se_ols": ols.bse.values, "p_ols": ols.pvalues.values})
    t["label"] = t.term.map(LABEL)
    t["stars"] = t.p_hc3.map(stars)
    return t


def cv_rmse(formula, data, k=10, seed=SEED):
    m = smf.ols(formula, data).fit()
    y, X = m.model.endog, m.model.exog
    idx = np.random.default_rng(seed).permutation(len(y))
    se = []
    for f in np.array_split(idx, k):
        tr = np.setdiff1d(idx, f)
        b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
        se.append((y[f] - X[f] @ b) ** 2)
    return float(np.sqrt(np.concatenate(se).mean()))


def wald(rob, names):
    """Heteroskedasticity-robust joint Wald (F) test that all `names` are zero."""
    w = rob.wald_test(", ".join(f"{n} = 0" for n in names), use_f=True, scalar=True)
    return float(w.statistic), float(w.pvalue), len(names)


def _edu_pair(data):
    base = ["edu", "exper", "exper2", "ten", "ten2", "female", "nonprev", "married", "dep", "metro", "norcen", "south", "west"]
    y = data.lsal.values
    X3 = sm.add_constant(data[base]).values
    X4 = sm.add_constant(data[base + FUNC + DIV + ["wrkhm"]]).values
    return np.linalg.lstsq(X3, y, rcond=None)[0][1], np.linalg.lstsq(X4, y, rcond=None)[0][1]


def run(ctx):
    S, S4, R, N = ctx.S, ctx.S4, ctx.R, ctx.N
    MOD, ROB, TBL, FIT = ctx.MOD, ctx.ROB, ctx.TBL, ctx.FIT
    for k, f in FORM.items():
        o, r = fit(f, S)
        MOD[k], ROB[k], TBL[k] = o, r, coef_table(o, r)
        TBL[k].to_csv(os.path.join(ctx.tab_dir, f"coef_{k}.csv"), index=False)
        FIT[k] = dict(n=int(o.nobs), k=int(o.df_model), r2=float(o.rsquared), adj_r2=float(o.rsquared_adj),
                      aic=float(o.aic), bic=float(o.bic), rmse=float(np.sqrt(o.mse_resid)), cv_rmse=cv_rmse(f, S),
                      F=float(o.fvalue), F_p=float(o.f_pvalue),
                      F_rob=float(r.fvalue) if np.ndim(r.fvalue) == 0 else float(np.squeeze(r.fvalue)),
                      F_rob_p=float(r.f_pvalue))
    R["fit"] = FIT
    R["coef"] = {k: TBL[k].to_dict(orient="records") for k in TBL}

    # nested comparisons: classical F and robust Wald
    steps = {"M1->M2 (personal characteristics)": ("M1", "M2", PERS.split(" + ")),
             "M2->M3 (location)": ("M2", "M3", LOC.split(" + ")),
             "M3->M4 (job assignment)": ("M3", "M4", JOB.split(" + ")),
             "M4->M5 (education squared)": ("M4", "M5", ["edu2"]),
             "M5->M6 (female x married)": ("M5", "M6", ["female:married"])}
    R["nested"] = {}
    for lab, (a, b, names) in steps.items():
        fc = MOD[b].compare_f_test(MOD[a])
        wr = wald(ROB[b], names)
        R["nested"][lab] = dict(F_classic=float(fc[0]), p_classic=float(fc[1]), df=int(fc[2]), F_robust=wr[0],
                                p_robust=wr[1], d_adj_r2=FIT[b]["adj_r2"] - FIT[a]["adj_r2"])

    blocks = {"Region dummies (norcen, south, west)": REGION, "IT function dummies (6)": FUNC,
              "IT division dummies (R&D, tech support)": DIV,
              "All job-assignment variables (9)": FUNC + DIV + ["wrkhm"],
              "Experience: exper and exper2": ["exper", "exper2"], "Tenure: ten and ten2": ["ten", "ten2"],
              "Education: edu and edu2": ["edu", "edu2"], "Female and Female x Married": ["female", "female:married"],
              "Married and Female x Married": ["married", "female:married"]}
    R["blocks"] = {k: dict(zip(["F", "p", "df"], wald(ROB["M6"], v))) for k, v in blocks.items()}

    # subsidiary (mediated) influence of education: coefficient with vs without job-assignment variables
    b3, b4 = _edu_pair(S)
    bs = np.array([_edu_pair(S.iloc[ctx.rng.integers(0, N, N)]) for _ in range(2000)])
    diff = bs[:, 0] - bs[:, 1]
    share = diff / bs[:, 0]
    R["mediation"] = dict(b_M3=float(b3), b_M4=float(b4), diff=float(b3 - b4),
                          diff_lo=float(np.percentile(diff, 2.5)), diff_hi=float(np.percentile(diff, 97.5)),
                          share=float((b3 - b4) / b3), share_lo=float(np.percentile(share, 2.5)),
                          share_hi=float(np.percentile(share, 97.5)))
    chan = {}
    for j in ["itrd", "ittech", "itarch", "itai", "itclient", "itprog", "itqual", "itreq", "wrkhm"]:
        m = smf.ols(f"{j} ~ edu + exper + female", S).fit(cov_type="HC3", use_t=True)
        chan[j] = dict(b=float(m.params["edu"]), p=float(m.pvalues["edu"]))
    R["edu_channels"] = chan

    # family background (sample S4)
    o4f, r4f = fit(FORM["M4"], S4)
    o5f, r5f = fit(FORM["M4"] + " + momed + daded + sib", S4)
    ctx.TB5F = coef_table(o5f, r5f)
    ctx.TB5F.to_csv(os.path.join(ctx.tab_dir, "coef_M4_family.csv"), index=False)
    R["family"] = dict(n=int(o5f.nobs), joint=dict(zip(["F", "p", "df"], wald(r5f, FAM))),
                       F_classic=float(o5f.compare_f_test(o4f)[0]), p_classic=float(o5f.compare_f_test(o4f)[1]),
                       coef=ctx.TB5F[ctx.TB5F.term.isin(FAM)].to_dict(orient="records"),
                       adj_r2_without=float(o4f.rsquared_adj), adj_r2_with=float(o5f.rsquared_adj),
                       bic_without=float(o4f.bic), bic_with=float(o5f.bic),
                       edu_b_without=float(r4f.params["edu"]), edu_b_with=float(r5f.params["edu"]))
    mp = smf.ols("edu ~ momed + daded + sib", S4).fit(cov_type="HC3", use_t=True)
    mpw = mp.wald_test("momed = 0, daded = 0, sib = 0", use_f=True, scalar=True)
    R["family"]["edu_on_parents"] = dict(coef=mp.params.to_dict(), p=mp.pvalues.to_dict(), r2=float(mp.rsquared),
                                         F_p=float(mp.f_pvalue) if not np.isnan(mp.f_pvalue) else None,
                                         joint_p=float(mpw.pvalue))
