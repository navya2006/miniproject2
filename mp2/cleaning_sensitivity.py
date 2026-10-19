"""Stage G2 - how much do the M6 results depend on the analyst-chosen cleaning thresholds?

Five scenarios re-clean the data differently and refit M6 (HC3). The last scenario applies illustrative
digit-slip repairs for gauging sensitivity only; it is not adopted.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from .config import FORM, KEYT, CORE_VARS

SCENARIOS = {
    "Baseline rules (sal <= 100)": dict(),
    "Keep salaries up to 500": dict(sal_keep=500),
    "Keep every positive salary": dict(sal_keep=1e9),
    "Drop salaries above 30": dict(sal_max=30),
    "Illustrative digit-slip repairs (not adopted)": dict(sal_fix={44: 11.11, 103: 4.3567, 360: 3.33334, 503: 2.45},
                                                          edu_fix={53: 12, 481: 12}),
}


def _sample(ctx, rawnum, sal_keep=None, sal_fix=None, edu_fix=None, sal_max=None):
    d2 = ctx.df.copy()
    if sal_keep is not None:                                  # restore raw salaries removed by the range rule
        m = rawnum["sal"].notna() & (rawnum["sal"] > 0) & (rawnum["sal"] <= sal_keep) & d2["sal"].isna()
        d2.loc[m, "sal"] = rawnum["sal"][m]
    for k, v in (sal_fix or {}).items():
        d2.loc[d2.id == k, "sal"] = v
    for k, v in (edu_fix or {}).items():
        d2.loc[d2.id == k, "edu"] = v
    if sal_max is not None:
        d2.loc[d2.sal > sal_max, "sal"] = np.nan
    d2["lsal"] = np.log(d2["sal"].where(d2["sal"] > 0))
    d2["edu2"] = d2.edu ** 2
    return d2[d2[CORE_VARS].notna().all(axis=1) & d2.lsal.notna()].copy()


def run(ctx):
    rawnum = {c: pd.to_numeric(ctx.raw[c], errors="coerce") for c in ["sal", "edu"]}
    sens = {}
    for name, kw in SCENARIOS.items():
        Sd = _sample(ctx, rawnum, **kw)
        rr = smf.ols(FORM["M6"], Sd).fit(cov_type="HC3", use_t=True)
        sens[name] = dict(n=int(len(Sd)), terms={t: dict(b=float(rr.params[t]), p=float(rr.pvalues[t])) for t in KEYT},
                          r2=float(rr.rsquared))
    assert sens["Baseline rules (sal <= 100)"]["n"] == ctx.N
    ctx.R["clean_sens"] = sens
