"""Stage F - diagnostics for the preferred model M6.

Normality, heteroskedasticity, functional form (RESET), the log-vs-level comparison, influence,
multicollinearity (VIF, condition index, mean-centring) and autocorrelation.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset, het_goldfeldquandt
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from .config import FORM


def _vif_table(X, names):
    return {n: float(variance_inflation_factor(X, i)) for i, n in enumerate(names) if n != "Intercept"}


def _cond_index(X):
    Xs = X / np.linalg.norm(X, axis=0)
    ev = np.linalg.eigvalsh(Xs.T @ Xs)
    return float(np.sqrt(ev.max() / ev.min()))


def _residual_profile(m):
    r = m.resid.values
    return dict(skew=float(stats.skew(r)), kurt=float(stats.kurtosis(r, fisher=False)),
                jb=float(stats.jarque_bera(r)[0]), bp_p=float(het_breuschpagan(r, m.model.exog)[1]),
                bpfit_p=float(het_breuschpagan(r, sm.add_constant(m.fittedvalues.values))[1]),
                reset_p=float(linear_reset(m, power=[2, 3], use_f=True).pvalue), r2=float(m.rsquared))


def run(ctx):
    S, R, N, MOD = ctx.S, ctx.R, ctx.N, ctx.MOD
    m6 = MOD["M6"]
    res, fitted, X6 = m6.resid.values, m6.fittedvalues.values, m6.model.exog
    D = {}

    # --- normality
    D["jb"] = dict(zip(["stat", "p"], map(float, stats.jarque_bera(res))))
    D["shapiro"] = dict(zip(["stat", "p"], map(float, stats.shapiro(res))))
    D["skew"], D["kurt"] = float(stats.skew(res)), float(stats.kurtosis(res, fisher=False))

    # --- heteroskedasticity
    bp = het_breuschpagan(res, X6, robust=True)
    D["bp"] = dict(lm=float(bp[0]), p=float(bp[1]), df=int(X6.shape[1] - 1))
    Zf = sm.add_constant(np.column_stack([fitted, fitted ** 2]))
    bpf = het_breuschpagan(res, Zf, robust=True)
    D["bp_fitted"] = dict(lm=float(bpf[0]), p=float(bpf[1]))
    bpn = het_breuschpagan(res, Zf, robust=False)          # assumes normal errors (not valid here)
    D["bp_fitted_nonrobust"] = dict(lm=float(bpn[0]), p=float(bpn[1]))
    bpa = het_breuschpagan(res, X6, robust=False)
    D["bp_nonrobust"] = dict(lm=float(bpa[0]), p=float(bpa[1]))
    srt = np.lexsort((S.id.values, S.exper.values))        # order by experience; ties broken by employee id
    gq = het_goldfeldquandt(S.lsal.values[srt], X6[srt], split=0.5, drop=0.2, result_object=False)
    D["gq_exper"] = dict(F=float(gq[0]), p=float(gq[1]))
    D["abs_res_fitted_corr"] = float(np.corrcoef(np.abs(res), fitted)[0, 1])
    S["res"] = res
    lev = {g: float(stats.levene(*[x.res.values for _, x in S.groupby(g)]).pvalue)
           for g in ["female", "married", "metro", "itrd", "itarch", "itai", "wrkhm"]}
    lev["edu_group"] = float(stats.levene(*[x.res.values for _, x in S.groupby(pd.cut(S.edu, [-1, 11, 12, 15, 20]), observed=True)]).pvalue)
    lev["exper_group"] = float(stats.levene(*[x.res.values for _, x in S.groupby(pd.cut(S.exper, [0, 5, 15, 30, 60]), observed=True)]).pvalue)
    D["levene"] = lev

    # --- functional form
    rs = linear_reset(m6, power=[2, 3], use_f=True)
    D["reset"] = dict(F=float(rs.fvalue), p=float(rs.pvalue))
    rs2 = linear_reset(m6, power=[2], use_f=True)
    D["reset2"] = dict(F=float(rs2.fvalue), p=float(rs2.pvalue))
    S["yh2"], S["yh3"] = fitted ** 2, fitted ** 3
    rr = smf.ols(FORM["M6"] + " + yh2 + yh3", S).fit(cov_type="HC3", use_t=True)
    w = rr.wald_test("yh2 = 0, yh3 = 0", use_f=True, scalar=True)
    D["reset_robust"] = dict(F=float(w.statistic), p=float(w.pvalue))
    D["reset_by_model"] = {k: float(linear_reset(MOD[k], power=[2, 3], use_f=True).pvalue) for k in MOD}
    D["dw"] = float(sm.stats.stattools.durbin_watson(S.sort_values("id").pipe(lambda d_: m6.resid.loc[d_.index]).values))

    # --- level vs log
    lvl = smf.ols(FORM["M4"].replace("lsal", "sal"), S).fit()
    D["level_vs_log"] = dict(level=_residual_profile(lvl), log=_residual_profile(MOD["M4"]))
    lam = stats.boxcox_normmax(S.sal, method="mle")
    D["boxcox_lambda"] = float(lam)
    _, lam_ci = stats.boxcox(S.sal, alpha=0.05)[1:]
    D["boxcox_ci"] = [float(lam_ci[0]), float(lam_ci[1])]

    # --- influence
    infl = OLSInfluence(m6)
    cd, hat, sr = infl.cooks_distance[0], infl.hat_matrix_diag, infl.resid_studentized_external
    S["cook"], S["hat"], S["sres"] = cd, hat, sr
    kpar = X6.shape[1]
    D["infl"] = dict(n_cook_4n=int((cd > 4 / N).sum()), n_cook_1=int((cd > 1).sum()), max_cook=float(cd.max()),
                     n_lev=int((hat > 2 * kpar / N).sum()), lev_thr=float(2 * kpar / N), n_sres3=int((np.abs(sr) > 3).sum()),
                     top=S.sort_values("cook", ascending=False).head(6)[
                         ["id", "sal", "edu", "exper", "ten", "female", "cook", "hat", "sres"]].to_dict(orient="records"))

    # --- multicollinearity: raw vs mean-centred polynomial terms
    D["vif_raw"] = _vif_table(X6, m6.model.exog_names)
    Sc = S.copy()
    for v in ["edu", "exper", "ten"]:
        Sc[v + "_c"] = Sc[v] - Sc[v].mean()
        Sc[v + "_c2"] = Sc[v + "_c"] ** 2
    Fc = FORM["M6"].replace("edu2", "edu_c2").replace("exper2", "exper_c2").replace("ten2", "ten_c2")
    for v in ["edu", "exper", "ten"]:
        Fc = Fc.replace(f"{v} +", f"{v}_c +", 1) if f"{v} +" in Fc else Fc
    mc = smf.ols(Fc, Sc).fit()
    D["centered_formula"] = Fc
    D["vif_cent"] = _vif_table(mc.model.exog, mc.model.exog_names)
    D["centered_same_fit"] = dict(r2_raw=float(m6.rsquared), r2_cent=float(mc.rsquared),
                                  max_abs_fitted_diff=float(np.max(np.abs(mc.fittedvalues.values - fitted))))
    D["cond_raw"], D["cond_cent"] = _cond_index(X6), _cond_index(mc.model.exog)
    D["rank_ok"] = bool(np.linalg.matrix_rank(X6) == X6.shape[1])
    D["vif_M4"] = _vif_table(MOD["M4"].model.exog, MOD["M4"].model.exog_names)
    D["corr_exper_ten"] = float(S.exper.corr(S.ten))
    R["diag"] = D
