"""Stage G - robustness of the preferred model M6.

Seven alternative specifications, Box-Cox dependent variable, 2000-resample pairs bootstrap,
and the drop-block R-squared decomposition.
"""
import re
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from .config import FORM, LABEL, KEYT, FUNC, DIV, REGION
from .models import fit


def run(ctx):
    S, R, N, ROB, m6 = ctx.S, ctx.R, ctx.N, ctx.ROB, ctx.MOD["M6"]
    D = R["diag"]
    specs = {}

    def add(name, params, pvals):
        specs[name] = {t: dict(b=float(params[t]), p=float(pvals[t])) for t in KEYT}

    add("Baseline M6 (HC3)", ROB["M6"].params, ROB["M6"].pvalues)
    add("Classical OLS SEs", m6.params, m6.pvalues)
    Sx = S[S.ten_gt_exper == 0]
    _, rx = fit(FORM["M6"], Sx); add(f"Excl. tenure>experience (n={len(Sx)})", rx.params, rx.pvalues)
    Si = S[S.cook <= 4 / N]
    _, ri = fit(FORM["M6"], Si); add(f"Excl. Cook's D>4/n (n={len(Si)})", ri.params, ri.pvalues)
    Sr = S[np.abs(S.sres) <= 3]
    _, rr = fit(FORM["M6"], Sr); add(f"Excl. |stud. resid|>3 (n={len(Sr)})", rr.params, rr.pvalues)
    qr = smf.quantreg(FORM["M6"], S).fit(q=0.5, max_iter=5000); add("Median (quantile) regression", qr.params, qr.pvalues)
    rl = smf.rlm(FORM["M6"], S, M=sm.robust.norms.HuberT()).fit(); add("Huber robust regression", rl.params, rl.pvalues)
    onr, rnr = fit(FORM["M6"].replace(" + ten2 +", " +"), S)
    p_, pv_ = rnr.params.copy(), rnr.pvalues.copy()
    p_["ten2"], pv_["ten2"] = np.nan, np.nan
    add("Without tenure squared", p_, pv_)

    lam = D["boxcox_lambda"]
    S["y_bc"] = (S.sal ** lam - 1) / lam
    bc = smf.ols(FORM["M6"].replace("lsal", "y_bc"), S).fit(cov_type="HC3", use_t=True)
    agree = {t: bool(np.sign(bc.params[t]) == np.sign(ROB["M6"].params[t]) and (bc.pvalues[t] < 0.05) == (ROB["M6"].pvalues[t] < 0.05))
             for t in KEYT}
    R["boxcox_model"] = dict(lam=float(lam), p={t: float(bc.pvalues[t]) for t in KEYT}, b={t: float(bc.params[t]) for t in KEYT},
                             agree=agree, n_agree=int(sum(agree.values())), n_terms=len(agree), r2=float(bc.rsquared))
    R["robust"] = dict(specs=specs, keyt=KEYT, label={t: LABEL[t] for t in KEYT},
                       fit_no_ten2=dict(bic=float(onr.bic), aic=float(onr.aic), bic_M6=float(m6.bic), aic_M6=float(m6.aic)))

    # pairs bootstrap, 2000 resamples
    X, y = m6.model.exog, S.lsal.values
    bt = np.empty((2000, X.shape[1]))
    for i in range(2000):
        ii = ctx.rng.integers(0, N, N)
        bt[i] = np.linalg.lstsq(X[ii], y[ii], rcond=None)[0]
    names = m6.model.exog_names
    boot = {n: dict(lo=float(np.percentile(bt[:, j], 2.5)), hi=float(np.percentile(bt[:, j], 97.5))) for j, n in enumerate(names)}
    excl = {n: bool(not (np.percentile(bt[:, j], 2.5) <= 0 <= np.percentile(bt[:, j], 97.5))) for j, n in enumerate(names)}
    boot["_excl_zero"] = excl
    boot["_agree"] = int(sum(excl[n] == (ROB["M6"].pvalues[n] < 0.05) for n in names))
    boot["_n"] = len(names)
    R["bootstrap"] = boot

    # drop-block R-squared
    def r2_without(drop):
        f = FORM["M6"]
        for d_ in drop:
            f = re.sub(r"\s\+\s" + re.escape(d_) + r"(?![\w:])", "", f)
        return smf.ols(f, S).fit().rsquared
    blocks = {"Human capital (edu, edu2, exper, exper2, ten, ten2)": ["edu", "edu2", "exper", "exper2", "ten", "ten2"],
              "Gender and marital status (female, married, female x married)": ["female:married", "female", "married"],
              "Job assignment (function, division, wrkhm)": FUNC + DIV + ["wrkhm"],
              "Location (metro, regions)": ["metro"] + REGION, "Non-privileged and dependents": ["nonprev", "dep"]}
    R["r2_drop"] = {k: float(m6.rsquared - r2_without(v)) for k, v in blocks.items()}
    R["r2_M6"] = float(m6.rsquared)
