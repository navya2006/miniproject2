"""Turn the results in ctx.R into presentation-ready DataFrames (all numbers already formatted as text).

Every builder returns a Table(key, title, df, note, align, highlight). The same tables feed the PNG renderer and
the Excel workbook, so the two outputs can never disagree.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import LABEL, FORM, KEYT, MODEL_NAMES, stars
from .effects import hypothesis_verdict
from .style import fmt_p


@dataclass
class Table:
    key: str
    title: str
    df: pd.DataFrame
    note: str = ""
    left_cols: tuple = (0,)             # columns aligned left (others centred)
    bold_rows: tuple = ()               # row indexes drawn in bold (e.g. totals)
    group: int = 1                      # rows per zebra stripe (2 for coefficient / s.e. pairs)


def _f(x, d=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def _p(p):
    return fmt_p(p)


def t_cleaning(ctx):
    C = ctx.R["clean"]
    rows = [("Raw records", C["n_raw"]), ("Records with at least one flagged cell", C["n_rows_flagged"]),
            ("Distinct problem cells set to missing", C["n_flag_cells"]), ("Blank dummies recovered by block logic", C["n_recovered"])]
    rows += [(f"  {k}", v) for k, v in C["by_cat"].items()]
    rows += [("Tenure > experience (flagged, kept)", C["n_ten_gt_exper"]), ("Main analysis sample (N)", C["n_final"]),
             ("Family-background sample", C["n_final_fam"])]
    return Table("cleaning_summary", "Data cleaning summary", pd.DataFrame(rows, columns=["Item", "Count"]),
                 "Every action is itemised in cleaning_log.csv. The Excel source file is never modified.", bold_rows=(len(rows) - 2,))


def t_descriptives(ctx):
    R = ctx.R
    nm = {"sal": "Hourly salary", "lsal": "ln(hourly salary)", "edu": "Education (years)", "exper": "Experience (years)",
          "ten": "Tenure (years)", "dep": "# dependents", "momed": "Mother's education", "daded": "Father's education", "sib": "# siblings"}
    rows = []
    for r in R["desc_cont"] + R["desc_fam"]:
        rows.append([nm[r["var"]], int(r["count"]), _f(r["mean"], 2), _f(r["std"], 2), _f(r["min"], 2), _f(r["median"], 2), _f(r["max"], 2)])
    return Table("descriptives_continuous", "Descriptive statistics: continuous variables",
                 pd.DataFrame(rows, columns=["Variable", "N", "Mean", "Std. dev.", "Min", "Median", "Max"]),
                 f"Main sample N = {ctx.N}; parental education and siblings use the family-background sample (N = {len(ctx.S4)}).")


def t_shares(ctx):
    rows = [[LABEL.get(c, c), int(v["n1"]), f"{100 * v['share']:.1f}%"] for c, v in ctx.R["desc_bin"].items()]
    return Table("descriptives_binary", "Descriptive statistics: 0/1 variables",
                 pd.DataFrame(rows, columns=["Variable", "Employees with value 1", "Share of sample"]),
                 f"N = {ctx.N}.")


def t_regression(ctx):
    """Side-by-side M1-M6 table: coefficient with stars, HC3 standard error beneath."""
    terms = [t for t in ctx.TBL["M6"].term if t != "Intercept"] + ["Intercept"]
    rows = []
    for t in terms:
        coef, se = [LABEL.get(t, t)], [""]
        for k in FORM:
            tb = ctx.TBL[k].set_index("term")
            if t in tb.index:
                r = tb.loc[t]
                coef.append(f"{r.b:.4f}{stars(r.p_hc3)}"); se.append(f"({r.se_hc3:.4f})")
            else:
                coef.append(""); se.append("")
        rows += [coef, se]
    for lab, fn in [("Observations", lambda k: f"{ctx.FIT[k]['n']}"), ("R-squared", lambda k: _f(ctx.FIT[k]["r2"])),
                    ("Adjusted R-squared", lambda k: _f(ctx.FIT[k]["adj_r2"])), ("AIC", lambda k: _f(ctx.FIT[k]["aic"], 1)),
                    ("BIC", lambda k: _f(ctx.FIT[k]["bic"], 1)), ("10-fold CV RMSE", lambda k: _f(ctx.FIT[k]["cv_rmse"]))]:
        rows.append([lab] + [fn(k) for k in FORM])
    cols = ["Dependent variable: ln(hourly salary)"] + [f"{k}\n{MODEL_NAMES[k]}" for k in FORM]
    return Table("regression_M1_M6", "Regression results, nested models M1-M6", pd.DataFrame(rows, columns=cols),
                 "HC3-robust standard errors in parentheses. * p<0.10, ** p<0.05, *** p<0.01. M6 is the preferred model.", group=2)


def t_nested(ctx):
    rows = []
    for lab, r in ctx.R["nested"].items():
        rows.append([lab, f"{r['F_classic']:.2f}", _p(r["p_classic"]), f"{r['F_robust']:.2f}", _p(r["p_robust"]), f"{r['d_adj_r2']:+.4f}"])
    return Table("nested_tests", "Does each block of variables improve the model?",
                 pd.DataFrame(rows, columns=["Step", "F (classical)", "p (classical)", "F (robust Wald)", "p (robust)", "Change in adj. R2"]),
                 "Robust Wald tests use the HC3 covariance matrix.")


def t_blocks(ctx):
    rows = [[k, f"{v['F']:.2f}", int(v["df"]), _p(v["p"])] for k, v in ctx.R["blocks"].items()]
    return Table("block_tests", "Joint significance of variable blocks in M6",
                 pd.DataFrame(rows, columns=["Block (H0: all coefficients zero)", "F (robust)", "df", "p-value"]), "HC3-robust Wald F tests.")


def t_hypotheses(ctx):
    rows = []
    for h in ctx.R["hyp"]:
        ok, p_used, text = hypothesis_verdict(h)
        rows.append([h["code"], h["label"].replace("beta_", "β ").replace("!=", "≠"), {"+": "positive", "-": "negative", "2s": "non-zero"}[h["pred"]],
                     f"{h['b']:+.4f}", f"{h['se']:.4f}", _p(p_used) + (" (1-sided)" if h["pred"] in ("+", "-") else ""),
                     ("✓ " if ok else "– ") + text])
    return Table("hypotheses", "Hypothesis tests", pd.DataFrame(rows, columns=["#", "Hypothesis", "Predicted", "Estimate", "HC3 s.e.", "p-value", "Verdict"]),
                 "H1 is tested in M4 (linear education); all others in M6. H8b is exploratory: Holm-adjusted p = "
                 f"{next(r['p_holm'] for r in ctx.R['battery']['inter1'] if r['term'] == 'female:married'):.3f} across the interaction search.",
                 left_cols=(0, 1, 6))


def t_effects(ctx):
    from .plots_models import EFFECT_GROUPS
    L = ctx.R["lin"]
    rows = []
    for g, items in EFFECT_GROUPS:
        for lab, k in items:
            d = L[k]
            rows.append([g, lab, f"{d['pct']:+.1f}%", f"[{d['pct_lo']:+.1f}%, {d['pct_hi']:+.1f}%]", _p(d["p"]) + stars(d["p"])])
    return Table("effects_percent", "Percentage pay differences implied by M6", pd.DataFrame(rows, columns=["Group", "Characteristic", "Effect", "95% CI", "p-value"]),
                 "100*(exp(b)-1), delta-method CI with HC3 covariance. * p<0.10, ** p<0.05, *** p<0.01.", left_cols=(0, 1))


def t_marginal(ctx):
    L = ctx.R["lin"]
    rows = []
    for v, nm, xs in [("edu", "Education", [8, 10, 12, 14, 16, 18]), ("exper", "Experience", [0, 5, 10, 20, 30, 40]), ("ten", "Tenure", [0, 5, 10, 20, 30])]:
        for x in xs:
            d = L[f"{v}_marg_{x}"]
            rows.append([nm, x, f"{d['pct']:+.2f}%", f"[{d['pct_lo']:+.2f}%, {d['pct_hi']:+.2f}%]", _p(d["p"]) + stars(d["p"])])
    T = ctx.R["turning"]
    return Table("marginal_effects", "Pay gain from one more year at different starting points", pd.DataFrame(rows, columns=["Variable", "At (years)", "% gain per year", "95% CI", "p-value"]),
                 f"Turning points: experience {T['exper']['tp']:.1f} years (95% CI {T['exper']['lo']:.1f} to {T['exper']['hi']:.1f}); "
                 f"tenure {T['ten']['tp']:.1f}; education {T['edu']['tp']:.1f} ({'inside' if T['edu']['within'] else 'outside'} the observed range).", left_cols=(0,))


def t_battery(ctx):
    B = ctx.R["battery"]
    rows = [[r["hypothesis"], f"{r['b']:+.4f}", _p(r["p"]), _p(r["p_holm"]), _p(r["p_bh"])] for r in B["inter1"]]
    for r in B["quad"]:
        rows.append([{"edu2": "Education squared", "dep2": "Dependents squared"}[r["term"]], f"{r['b']:+.4f}", _p(r["p"]), _p(r["p_holm"]), _p(r["p_bh"])])
    return Table("interaction_search", "Quadratic and interaction search (added one at a time to M4)",
                 pd.DataFrame(rows, columns=["Candidate term", "Coefficient", "p (HC3)", "p (Holm)", "p (BH)"]),
                 "Holm controls the family-wise error rate; BH controls the false discovery rate. Each family is adjusted separately.", left_cols=(0,))


def _reject(p, yes, no, borderline=None):
    if p < 0.05:
        return yes
    if borderline and p < 0.10:
        return borderline
    return no


def t_diagnostics(ctx):
    D = ctx.R["diag"]
    vr, vc = max(D["vif_raw"].values()), max(D["vif_cent"].values())
    rows = [
        ["Normality", "Jarque-Bera", f"{D['jb']['stat']:.1f}", _p(D["jb"]["p"]), _reject(D["jb"]["p"], "Rejected: non-normal errors; HC3 and bootstrap used", "Not rejected")],
        ["Normality", "Shapiro-Wilk", f"{D['shapiro']['stat']:.3f}", _p(D["shapiro"]["p"]), _reject(D["shapiro"]["p"], "Rejected", "Not rejected")],
        ["Heteroskedasticity", "Breusch-Pagan (robust, all regressors)", f"{D['bp']['lm']:.1f}", _p(D["bp"]["p"]), _reject(D["bp"]["p"], "Heteroskedasticity detected", "No evidence")],
        ["Heteroskedasticity", "Breusch-Pagan (robust, fitted values)", f"{D['bp_fitted']['lm']:.2f}", _p(D["bp_fitted"]["p"]), _reject(D["bp_fitted"]["p"], "Heteroskedasticity detected", "No evidence")],
        ["Heteroskedasticity", "Goldfeld-Quandt (sorted by experience)", f"{D['gq_exper']['F']:.3f}", _p(D["gq_exper"]["p"]), _reject(D["gq_exper"]["p"], "Heteroskedasticity detected", "No evidence")],
        ["Functional form", "RESET (classical, squares and cubes)", f"{D['reset']['F']:.2f}", _p(D["reset"]["p"]), _reject(D["reset"]["p"], "Rejected", "Not rejected", "Borderline (p < 0.10)")],
        ["Functional form", "RESET (HC3-robust)", f"{D['reset_robust']['F']:.2f}", _p(D["reset_robust"]["p"]), _reject(D["reset_robust"]["p"], "Rejected: claims kept associational", "Not rejected")],
        ["Autocorrelation", "Durbin-Watson (ordered by id)", f"{D['dw']:.2f}", "", "Informal only: the data have no time order"],
        ["Multicollinearity", "Max VIF, raw / centred", f"{vr:.1f} / {vc:.1f}", "", "Centring removes the squared-term inflation" if vc < 5 < vr else "See figure 8"],
        ["Multicollinearity", "Condition index, raw / centred", f"{D['cond_raw']:.0f} / {D['cond_cent']:.0f}", "", "Centred value below 30" if D["cond_cent"] < 30 else "Centred value above 30"],
        ["Influence", "Obs. with Cook's D > 4/n", f"{D['infl']['n_cook_4n']}", "", f"Max Cook's D = {D['infl']['max_cook']:.2f}" + ("; none exceed 1" if D["infl"]["n_cook_1"] == 0 else "")],
        ["Influence", "Obs. with |studentised residual| > 3", f"{D['infl']['n_sres3']}", "", "Excluded in a robustness check"],
    ]
    return Table("diagnostics", "Diagnostic tests for the preferred model M6",
                 pd.DataFrame(rows, columns=["Area", "Test", "Statistic", "p-value", "Reading"]),
                 "Breusch-Pagan rows use the Koenker (robust) version. Classical BP on the fitted values: "
                 f"p = {_p(D['bp_fitted_nonrobust']['p'])}; it assumes normal errors, which fail here.", left_cols=(0, 1, 4))


def t_robustness(ctx):
    R = ctx.R["robust"]
    names = list(R["specs"])
    rows = []
    for t in KEYT:
        row = [LABEL[t]]
        for n in names:
            d = R["specs"][n][t]
            row.append("n/a" if d["b"] is None or np.isnan(d["b"]) else f"{d['b']:+.4f}{stars(d['p'])}")
        rows.append(row)
    cols = ["Term"] + [n.replace(" (", "\n(") for n in names]
    return Table("robustness", "Robustness of the key M6 coefficients", pd.DataFrame(rows, columns=cols),
                 "Coefficients with stars. Baseline = HC3. * p<0.10, ** p<0.05, *** p<0.01.")


def t_clean_sens(ctx):
    C = ctx.R["clean_sens"]
    names = list(C)
    rows = []
    for t in KEYT:
        rows.append([LABEL[t]] + [f"{C[n]['terms'][t]['b']:+.4f}{stars(C[n]['terms'][t]['p'])}" for n in names])
    rows.append(["Observations"] + [str(C[n]["n"]) for n in names])
    return Table("cleaning_sensitivity", "M6 under alternative cleaning thresholds", pd.DataFrame(rows, columns=["Term"] + [n.replace(" (", "\n(") for n in names]),
                 "The last scenario is illustrative only and is not adopted.")


def t_family(ctx):
    F = ctx.R["family"]
    rows = [[LABEL[r["term"]], f"{r['b']:+.4f}", f"{r['se_hc3']:.4f}", _p(r["p_hc3"]) + stars(r["p_hc3"])] for r in F["coef"]]
    rows.append(["Joint test (all three = 0)", f"F = {F['joint']['F']:.2f}", f"df = {int(F['joint']['df'])}", _p(F["joint"]["p"])])
    return Table("family_background", f"Family background added to M4 (N = {F['n']})", pd.DataFrame(rows, columns=["Variable", "Coefficient", "HC3 s.e.", "p-value"]),
                 f"Adjusted R2 {F['adj_r2_without']:.3f} without vs {F['adj_r2_with']:.3f} with the family variables; BIC {F['bic_without']:.1f} vs {F['bic_with']:.1f}.")


def build_all(ctx):
    return [f(ctx) for f in (t_cleaning, t_descriptives, t_shares, t_regression, t_nested, t_blocks, t_hypotheses, t_effects,
                             t_marginal, t_battery, t_family, t_diagnostics, t_robustness, t_clean_sens)]
