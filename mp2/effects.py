"""Stage - derived quantities from the preferred model M6 (delta method, HC3 covariance).

Percentage effects, marginal effects at chosen values, turning points of the quadratics,
planned contrasts and the 14-line hypothesis summary.
"""
import numpy as np
from scipy import stats


def run(ctx):
    S, R, ROB = ctx.S, ctx.R, ctx.ROB
    r6 = ROB["M6"]
    V = r6.cov_params()
    pnames = list(r6.params.index)

    def lincomb(w):
        """Linear combination of M6 coefficients with HC3 delta-method inference and % effect."""
        v = np.zeros(len(pnames))
        for k, c in w.items():
            v[pnames.index(k)] = c
        est = float(v @ r6.params.values)
        se = float(np.sqrt(v @ V.values @ v))
        dfree = r6.df_resid
        tq = stats.t.ppf(0.975, dfree)
        p = float(2 * (1 - stats.t.cdf(abs(est / se), dfree)))
        return dict(b=est, se=se, lo=est - tq * se, hi=est + tq * se, p=p, pct=100 * (np.exp(est) - 1),
                    pct_lo=100 * (np.exp(est - tq * se) - 1), pct_hi=100 * (np.exp(est + tq * se) - 1))

    lin = {"gender_gap_unmarried": lincomb({"female": 1}),
           "gender_gap_married": lincomb({"female": 1, "female:married": 1}),
           "marriage_effect_men": lincomb({"married": 1}),
           "marriage_effect_women": lincomb({"married": 1, "female:married": 1})}
    for k in ["nonprev", "metro", "norcen", "south", "west", "itreq", "itprog", "itqual", "itarch", "itai",
              "itclient", "itrd", "ittech", "wrkhm", "dep"]:
        lin[k] = lincomb({k: 1})
    for e in (8, 10, 12, 14, 16, 18):
        lin[f"edu_marg_{e}"] = lincomb({"edu": 1, "edu2": 2 * e})
    for x in (0, 5, 10, 20, 30, 40):
        lin[f"exper_marg_{x}"] = lincomb({"exper": 1, "exper2": 2 * x})
    for t in (0, 5, 10, 20, 30):
        lin[f"ten_marg_{t}"] = lincomb({"ten": 1, "ten2": 2 * t})
    R["lin"] = lin
    R["func_contrasts_ai"] = {g: lincomb({"itai": 1, g: -1}) for g in ["itreq", "itprog", "itqual", "itarch", "itclient"]}
    R["func_contrasts_ai"]["none_of_six"] = lincomb({"itai": 1})

    def turning(lin_, quad, xmin, xmax):
        b1, b2 = r6.params[lin_], r6.params[quad]
        tp = -b1 / (2 * b2)
        g = np.zeros(len(pnames))
        g[pnames.index(lin_)] = -1 / (2 * b2)
        g[pnames.index(quad)] = b1 / (2 * b2 ** 2)
        se = float(np.sqrt(g @ V.values @ g))
        return dict(tp=float(tp), se=se, lo=float(tp - 1.96 * se), hi=float(tp + 1.96 * se), within=bool(xmin <= tp <= xmax))

    R["turning"] = {"exper": turning("exper", "exper2", S.exper.min(), S.exper.max()),
                    "ten": turning("ten", "ten2", S.ten.min(), S.ten.max()),
                    "edu": turning("edu", "edu2", S.edu.min(), S.edu.max())}
    R["sample_ranges"] = {v: [float(S[v].min()), float(S[v].max()), float(S[v].mean())] for v in ["edu", "exper", "ten"]}
    R["share_past_exper_peak"] = float((S.exper > R["turning"]["exper"]["tp"]).mean())

    # hypothesis summary (two-sided p; one-sided where a sign was predicted)
    H = []

    def add(code, label, term, pred, spec="M6"):
        r = ROB[spec]
        b, se, tt, p = r.params[term], r.bse[term], r.tvalues[term], r.pvalues[term]
        sign = +1 if pred == "+" else -1
        p_one = float(1 - stats.t.cdf(sign * tt, r.df_resid)) if pred in ("+", "-") else None
        H.append(dict(code=code, label=label, term=term, pred=pred, b=float(b), se=float(se), t=float(tt),
                      p=float(p), p_one=p_one, spec=spec))

    add("H1", "Education: beta_edu > 0", "edu", "+", spec="M4")
    add("H2a", "Experience: beta_exper > 0", "exper", "+")
    add("H2b", "Experience squared: beta_exper2 < 0", "exper2", "-")
    add("H3a", "Tenure: beta_ten > 0", "ten", "+")
    add("H3b", "Tenure squared: beta_ten2 < 0", "ten2", "-")
    add("H4", "Gender (unmarried): beta_female < 0", "female", "-")
    add("H5", "Non-privileged: beta_nonprev < 0", "nonprev", "-")
    add("H6", "Education squared: beta_edu2 != 0", "edu2", "2s")
    add("H7a", "Metro: beta_metro > 0", "metro", "+")
    add("H7b", "R&D division: beta_itrd > 0", "itrd", "+")
    add("H7c", "AI applications: beta_itai > 0", "itai", "+")
    add("H7d", "Works from home: beta_wrkhm != 0", "wrkhm", "2s")
    add("H8a", "Marriage (men): beta_married > 0", "married", "+")
    add("H8b", "Female x Married != 0", "female:married", "2s")
    add("H9", "# dependents: beta_dep != 0", "dep", "2s")
    R["hyp"] = H


def hypothesis_verdict(h):
    """Verdict for one hypothesis-summary row: directional hypotheses use the one-sided p, others the two-sided p.

    Returns (supported: bool, p_used: float, text: str). A directional hypothesis whose estimate is significant in
    the opposite direction is labelled "Contradicted". The female x married interaction is flagged exploratory
    because it does not survive correction for the 12-term interaction search.
    """
    p_used = h["p_one"] if h["pred"] in ("+", "-") else h["p"]
    ok = bool(p_used < 0.05)
    if ok:
        text = "Supported (exploratory)" if h["code"] == "H8b" else "Supported"
    elif h["pred"] in ("+", "-") and h["p"] < 0.05:
        text = "Contradicted (significant, opposite sign)"
    else:
        text = "Not supported"
    return ok, float(p_used), text
