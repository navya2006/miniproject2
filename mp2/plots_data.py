"""Figures about the data: distributions, human-capital profiles, group comparisons, correlations, cleaning."""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from .style import (BLUE, ORANGE, GREY, DGREY, LGREY, INK2, SURFACE, DIVERGING, apply_theme, headline, save)


def fig_salary_distribution(ctx):
    S, R = ctx.S, ctx.R
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, v, ttl, sk in [(ax[0], "sal", "Hourly salary", R["skew"]["sal"]), (ax[1], "lsal", "Log hourly salary, ln(sal)", R["skew"]["lsal"])]:
        a.hist(S[v], bins=40, color=BLUE, edgecolor="white", linewidth=0.5)
        a.axvline(S[v].median(), color=ORANGE, lw=2)
        a.set_title(ttl); a.set_xlabel("Hourly salary" if v == "sal" else "ln(sal)"); a.set_ylabel("Employees" if v == "sal" else "")
        a.text(0.97, 0.95, f"skewness {sk:.2f}", transform=a.transAxes, ha="right", va="top", color=INK2, fontsize=10)
        a.text(S[v].median(), a.get_ylim()[1] * 0.80, "  median", color=INK2, fontsize=9, va="top")
    top = headline(fig, "Taking logs removes the right skew in pay", f"{ctx.N} employees; the orange line marks the median")
    fig.tight_layout(rect=[0, 0, 1, top - 0.02]); save(ctx, fig, "fig1_salary_distribution.png")


def fig_hc_profiles(ctx):
    S, N = ctx.S, ctx.N
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True)
    for a, v, lab in zip(ax, ["edu", "exper", "ten"], ["Years of education", "Years of experience", "Years of tenure"]):
        x = S[v].values + (ctx.jitter[v] if v in ctx.jitter else 0)
        a.scatter(x, S.lsal, s=14, color=BLUE, alpha=0.28, linewidths=0)
        lo = lowess(S.lsal, S[v], frac=0.45)
        a.plot(lo[:, 0], lo[:, 1], color=ORANGE, lw=2.6)
        a.set_xlabel(lab)
    ax[0].set_ylabel("ln(sal)")
    top = headline(fig, "Pay rises with education, experience and tenure, at different speeds",
                   "Each dot is an employee (education and tenure jittered); the orange line is a LOWESS smoother")
    fig.tight_layout(rect=[0, 0, 1, top - 0.02]); save(ctx, fig, "fig2_human_capital_profiles.png")


def fig_gender_marital(ctx):
    gm = pd.DataFrame(ctx.R["gender_marital_means"])
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for f, col, lab in [(0, BLUE, "Male"), (1, ORANGE, "Female")]:
        g = gm[gm.female == f]
        ax.errorbar([0, 1], g["mean"], yerr=g["ci"], color=col, marker="o", ms=8, lw=2.6, capsize=4, elinewidth=1.3)
        ax.text(1.07, g["mean"].iloc[1], lab, color=DGREY, va="center", fontsize=11, fontweight="bold")
        for xx, (m, n_) in enumerate(zip(g["mean"], g["count"])):
            ax.text(xx - 0.06, m + (0.05 if f == 0 else -0.06), f"n = {n_}", color=INK2, ha="right", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Unmarried", "Married"]); ax.set_xlim(-0.4, 1.5)
    ax.set_ylabel("Mean ln(sal) with 95% CI")
    top = headline(fig, "Marriage and the gender pay gap", "Mean log salary by gender and marital status", top=0.9)
    fig.tight_layout(rect=[0, 0, 1, top - 0.02]); save(ctx, fig, "fig3_gender_marital.png")


def fig_job_groups(ctx):
    S, R = ctx.S, ctx.R
    sections = [("IT function group", "function"), ("Division", "division"), ("Work location", "wfh")]
    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    ypos, ylabels, y = [], [], 0
    for title, col in sections:
        g = pd.DataFrame(R["group_means"][col])
        ax.text(1.105, y, title.upper(), fontsize=9, fontweight="bold", color=INK2, va="center", ha="left")
        ax.axhline(y + 0.5, color=LGREY, lw=1)
        y += 1
        for _, r_ in g.iloc[::-1].iterrows():
            ax.errorbar(r_["mean"], y, xerr=r_["ci"], fmt="o", color=BLUE, ms=7, elinewidth=1.7, capsize=3)
            ypos.append(y); ylabels.append(f"{r_[col]}  (n={int(r_['count'])})"); y += 1
        y += 0.6
    ax.axvline(S.lsal.mean(), color=GREY, lw=1, ls="--")
    ax.text(S.lsal.mean(), -0.9, " overall mean", color=INK2, fontsize=9, ha="left", va="center")
    ax.set_yticks(ypos); ax.set_yticklabels(ylabels, fontsize=10); ax.invert_yaxis()
    ax.grid(axis="y", visible=False); ax.set_xlabel("Mean ln(sal) with 95% CI"); ax.set_xlim(1.1, 2.1)
    top = headline(fig, "Pay differs by job assignment", "Within each block, groups run from highest to lowest mean pay", top=0.92)
    fig.tight_layout(rect=[0, 0, 1, top - 0.02]); save(ctx, fig, "fig4_job_groups.png")


def fig_correlations(ctx):
    cm = pd.DataFrame(ctx.R["corr_fam"])
    cm = cm.loc[cm.columns, cm.columns]
    fig, ax = plt.subplots(figsize=(6.8, 5.9))
    im = ax.imshow(cm.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm))); ax.set_yticklabels(cm.columns)
    for i in range(len(cm)):
        for j in range(len(cm)):
            v = cm.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if abs(v) > 0.6 else DGREY)
    ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02).outline.set_visible(False)
    top = headline(fig, "Correlations among the continuous variables",
                   f"Pearson r, family-background sample (n = {len(ctx.S4)}); blue positive, red negative", top=0.92)
    fig.tight_layout(rect=[0, 0, 1, top - 0.02]); save(ctx, fig, "fig5_correlations.png")


def fig_cleaning(ctx):
    """Two panels: how 526 records become the 502-employee sample, and which cells were repaired or dropped."""
    C, lg = ctx.R["clean"], ctx.log
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    a = ax[0]
    short = ["Salary missing / invalid", "Education missing", "Experience missing", "Metro / region dummies missing",
             "IT function / division dummies missing"]
    steps = [("Raw records", C["n_raw"], None)]
    cum = C["n_raw"]
    for lab, (_, k) in zip(short, C["waterfall"]):
        cum -= k
        steps.append((f"minus {lab.lower()}" if False else lab, cum, k))
    steps.append(("Analysis sample", cum, None))
    yy = np.arange(len(steps))[::-1]
    a.barh(yy, [s_[1] for s_ in steps], color=[BLUE if s_[2] is None else "#86b6ef" for s_ in steps], height=0.62)
    xr = C["n_raw"] * 1.04
    for y_, (lab, n_, k) in zip(yy, steps):
        a.text(n_ - 8, y_, f"{n_}", va="center", ha="right", fontsize=10.5, fontweight="bold", color="white" if k is None else DGREY)
        a.text(xr, y_, f"-{k}" if k else "", va="center", fontsize=10.5, color=ORANGE, fontweight="bold")
    a.text(xr, yy[0] + 0.62, "dropped", fontsize=8.5, color=INK2, va="bottom")
    a.set_yticks(yy); a.set_yticklabels([s_[0] for s_ in steps], fontsize=9.5); a.set_xlim(0, C["n_raw"] * 1.14)
    a.set_xlabel("Employees remaining after each exclusion step"); a.grid(axis="y", visible=False)
    a.set_title("(a) Records remaining")
    a = ax[1]
    bad = lg[lg.cat != "recovered via block logic"]
    pt = bad.groupby(["variable", "cat"]).size().unstack(fill_value=0)
    pt = pt.loc[pt.sum(axis=1).sort_values().index]
    cats = list(pt.columns); cols = [BLUE, ORANGE, "#1baf7a", "#eda100"]
    left = np.zeros(len(pt))
    for c, col in zip(cats, cols):
        a.barh(pt.index, pt[c], left=left, color=col, label=c, height=0.62, edgecolor=SURFACE, linewidth=1.5)
        left += pt[c].values
    for y_, t in enumerate(left):
        a.text(t + 0.3, y_, f"{int(t)}", va="center", fontsize=9.5, color=DGREY)
    a.legend(loc="lower right", fontsize=9)
    a.set_xlabel("Problem cells set to missing"); a.grid(axis="y", visible=False)
    a.set_title(f"(b) {C['n_flag_cells']} problem cells by variable and type")
    top = headline(fig, "Data cleaning at a glance",
                   f"{C['n_rows_flagged']} of {C['n_raw']} records had a flagged cell; {C['n_recovered']} blank dummies were recovered by block logic; every action is in cleaning_log.csv")
    fig.tight_layout(rect=[0, 0, 1, top - 0.02]); save(ctx, fig, "fig11_data_cleaning.png")


def run(ctx):
    apply_theme()
    for f in (fig_salary_distribution, fig_hc_profiles, fig_gender_marital, fig_job_groups, fig_correlations, fig_cleaning):
        f(ctx)
