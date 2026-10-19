"""Figures about the estimated models: fit comparison, effect sizes, returns to human capital, hypotheses."""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from .effects import hypothesis_verdict
from .style import (BLUE, ORANGE, GREY, DGREY, INK, INK2, SURFACE, apply_theme, headline, save, fmt_p, p_label)

BLUE_SOFT = "#9ec5f4"


EFFECT_GROUPS = [("Gender and marital status", [("Gender gap, unmarried (female vs male)", "gender_gap_unmarried"),
                                         ("Gender gap, married (female vs male)", "gender_gap_married"),
                                         ("Marriage effect, men", "marriage_effect_men"),
                                         ("Marriage effect, women", "marriage_effect_women")]),
          ("Background", [("Non-privileged", "nonprev")]),
          ("Location (vs East region)", [("Metro area", "metro"), ("North-central", "norcen"), ("South", "south"), ("West", "west")]),
          ("Job assignment", [("IT requirements", "itreq"), ("IT programming", "itprog"), ("IT quality assurance", "itqual"),
                              ("IT architecture", "itarch"), ("IT AI applications", "itai"), ("IT client-only", "itclient"),
                              ("IT R&D division", "itrd"), ("IT tech support", "ittech"), ("Works from home", "wrkhm")])]


def fig_dummy_effects(ctx):
    L = ctx.R["lin"]
    groups = EFFECT_GROUPS
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    y = 0
    yt, yl = [], []
    xmax = max(L[k]["pct_hi"] for _, it in groups for _, k in it)
    xmin = min(L[k]["pct_lo"] for _, it in groups for _, k in it)
    for gi, (gname, items) in enumerate(groups):
        ax.text(xmin - 0.02 * (xmax - xmin), y, gname.upper(), fontsize=8.5, fontweight="bold", color=INK2, ha="left", va="center")
        y += 1
        for lab, k in items:
            d = L[k]; col = BLUE if d["p"] < 0.05 else GREY
            ax.plot([d["pct_lo"], d["pct_hi"]], [y, y], color=col, lw=2.2, solid_capstyle="round")
            ax.scatter(d["pct"], y, color=col, s=55, zorder=3)
            ax.text(xmax + 0.04 * (xmax - xmin), y, f"{d['pct']:+.1f}%   {p_label(d['p'])}", va="center", fontsize=9,
                    color=INK if d["p"] < 0.05 else INK2, fontweight="bold" if d["p"] < 0.05 else "normal")
            yt.append(y); yl.append(lab); y += 1
        y += 0.5
    ax.axvline(0, color=DGREY, lw=1)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=10); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlim(xmin - 0.02 * (xmax - xmin), xmax + 0.04 * (xmax - xmin))
    ax.set_xlabel("Difference in hourly salary (%), 95% CI (HC3-robust)")
    top = headline(fig, "How much do these characteristics move pay?",
                   "Model M6, other characteristics held fixed. Blue = significant at 5%, grey = not distinguishable from zero", top=0.93)
    fig.tight_layout(rect=[0, 0, 0.84, top - 0.02]); save(ctx, fig, "fig9_dummy_effects.png")


def _pred(ctx, var, grid):
    S, r6 = ctx.S, ctx.ROB["M6"]
    base = S.mean(numeric_only=True).to_dict()
    rows = []
    for g in grid:
        d_ = dict(base); d_[var] = g
        for v in ("edu", "exper", "ten", "dep"):
            d_[v + "2"] = d_[v] ** 2
        rows.append(d_)
    return r6.get_prediction(pd.DataFrame(rows)).summary_frame(alpha=0.05)


def fig_predicted_profiles(ctx):
    R, S = ctx.R, ctx.S
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    specs = [("edu", np.linspace(6, 18, 60), "Years of education", "(a) Education: convex returns"),
             ("exper", np.linspace(1, 51, 80), "Years of experience", "(b) Experience: concave profile"),
             ("ten", np.linspace(0, 30, 60), "Years of tenure", "(c) Tenure: gains level off")]
    for a, (v, g, xl, ttl) in zip(ax, specs):
        gp = _pred(ctx, v, g)
        a.fill_between(g, gp["mean_ci_lower"], gp["mean_ci_upper"], color=BLUE, alpha=0.16, lw=0)
        a.plot(g, gp["mean"], color=BLUE, lw=2.8)
        a.set_xlabel(xl); a.set_title(ttl)
        tp = R["turning"].get(v)
        if tp and v in ("exper", "ten") and g.min() <= tp["tp"] <= g.max():
            a.axvline(tp["tp"], color=ORANGE, lw=1.6, ls="--")
            a.text(tp["tp"] + (g.max() - g.min()) * 0.02, a.get_ylim()[0] + 0.03, f"peak ~ {tp['tp']:.0f} yrs", color=INK2, fontsize=9.5)
    ax[0].set_ylabel("Predicted ln(sal), other variables at their means")
    top = headline(fig, "What the fitted profiles look like", "Model M6 predictions with 95% confidence bands")
    fig.tight_layout(rect=[0, 0, 1, top - 0.02]); save(ctx, fig, "fig10_predicted_profiles.png")


def fig_model_comparison(ctx):
    F, R = ctx.FIT, ctx.R
    ks = list(F)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.3]))
    a = ax[0]
    vals = [F[k]["adj_r2"] for k in ks]
    a.bar(ks, vals, color=[BLUE_SOFT] * (len(ks) - 1) + [BLUE], width=0.66)
    for i, v in enumerate(vals):
        a.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9, color=DGREY, fontweight="bold" if i == len(ks) - 1 else "normal")
    a.set_ylim(0, max(vals) * 1.15); a.set_title("(a) Adjusted R-squared"); a.grid(axis="x", visible=False)
    a.set_ylabel("Share of ln(sal) variance explained")
    a = ax[1]
    cv = [F[k]["cv_rmse"] for k in ks]
    a.plot(ks, cv, color=BLUE, lw=2.4, marker="o", ms=8)
    for i, v in enumerate(cv):
        a.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(9, 9), ha="center", fontsize=9, color=DGREY,
                   bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))
    a.set_ylim(min(cv) - (max(cv) - min(cv)) * 0.2, max(cv) + (max(cv) - min(cv)) * 0.25)
    a.set_title("(b) Out-of-sample error (10-fold CV RMSE)"); a.set_ylabel("Lower is better"); a.grid(axis="x", visible=False)
    a = ax[2]
    items = sorted(R["r2_drop"].items(), key=lambda kv: kv[1])
    labs = [k.split(" (")[0] for k, _ in items]
    a.barh(labs, [v for _, v in items], color=BLUE, height=0.6)
    for i, (_, v) in enumerate(items):
        a.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9.5, color=DGREY)
    a.set_xlim(0, max(v for _, v in items) * 1.18); a.grid(axis="y", visible=False)
    a.set_title("(c) R-squared lost when a block is removed from M6")
    top = headline(fig, "Model fit: each block of variables earns its place",
                   "M1 human capital, M2 + personal, M3 + location, M4 + job assignment, M5 + education squared, M6 + female x married")
    fig.tight_layout(rect=[0, 0, 1, top - 0.02]); save(ctx, fig, "fig12_model_comparison.png")


def fig_marginal_returns(ctx):
    L = ctx.R["lin"]
    panels = [("edu", [8, 10, 12, 14, 16, 18], "Education: extra salary from one more year", "Years of education"),
              ("exper", [0, 5, 10, 20, 30, 40], "Experience: extra salary from one more year", "Years of experience"),
              ("ten", [0, 5, 10, 20, 30], "Tenure: extra salary from one more year", "Years of tenure")]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.8))
    for a, (v, xs, ttl, xl) in zip(ax, panels):
        d = [L[f"{v}_marg_{x}"] for x in xs]
        a.fill_between(xs, [i["pct_lo"] for i in d], [i["pct_hi"] for i in d], color=BLUE, alpha=0.15, lw=0)
        a.plot(xs, [i["pct"] for i in d], color=BLUE, lw=2.6)
        for x, i in zip(xs, d):
            a.scatter(x, i["pct"], s=60, zorder=3, facecolor=BLUE if i["p"] < 0.05 else SURFACE, edgecolor=BLUE, linewidth=1.8)
            a.annotate(f"{i['pct']:+.1f}%", (x, i["pct"]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color=DGREY)
        a.axhline(0, color=DGREY, lw=1)
        a.set_title(ttl.split(":")[0]); a.set_xlabel(xl); a.set_xticks(xs); a.margins(x=0.08)
    ax[0].set_ylabel("% change in hourly salary per extra year")
    top = headline(fig, "Returns to a year of human capital depend on where you start",
                   "Model M6 marginal effects with 95% bands. Filled dot = significant at 5%, hollow = not significant")
    fig.tight_layout(rect=[0, 0, 1, top - 0.02]); save(ctx, fig, "fig13_marginal_returns.png")


def fig_hypotheses(ctx):
    H = ctx.R["hyp"]
    n = len(H)
    fig, ax = plt.subplots(figsize=(12.5, 0.52 * n + 1.9))
    ax.set_xlim(0, 100); ax.set_ylim(n + 0.6, -1.1); ax.axis("off")
    cols = {"code": 1, "label": 7, "pred": 47, "b": 56, "p": 66, "verdict": 76}
    for k, t in [("code", "#"), ("label", "Hypothesis"), ("pred", "Predicted"), ("b", "Estimate"), ("p", "p-value"), ("verdict", "Verdict")]:
        ax.text(cols[k], -0.7, t, fontsize=10, fontweight="bold", color=INK2, va="center")
    ax.plot([0, 100], [-0.3, -0.3], color=GREY, lw=1)
    nsup = 0
    for i, h in enumerate(H):
        ok, p_used, text = hypothesis_verdict(h)
        nsup += ok
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f4f3ef", lw=0)
        ax.text(cols["code"], i, h["code"], fontsize=10, fontweight="bold", va="center")
        ax.text(cols["label"], i, h["label"].replace("beta_", "β ").replace("!=", "≠"), fontsize=10, va="center")
        ax.text(cols["pred"], i, {"+": "positive", "-": "negative", "2s": "non-zero"}[h["pred"]], fontsize=10, va="center", color=INK2)
        ax.text(cols["b"], i, f"{h['b']:+.4f}", fontsize=10, va="center", family="DejaVu Sans Mono")
        ax.text(cols["p"], i, f"{fmt_p(p_used)}" + ("  (1-sided)" if h["pred"] in ("+", "-") else ""), fontsize=10, va="center")
        if ok:
            ax.scatter(cols["verdict"] + 1, i, s=150, color=BLUE, zorder=3); ax.text(cols["verdict"] + 1, i, "✓", color="white", ha="center", va="center", fontsize=9, fontweight="bold")
        else:
            ax.scatter(cols["verdict"] + 1, i, s=150, facecolor=SURFACE, edgecolor=GREY, linewidth=1.8, zorder=3); ax.text(cols["verdict"] + 1, i, "–", color=GREY, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(cols["verdict"] + 3.5, i, text, fontsize=9.5, va="center", color=INK if ok else INK2)
    inter = {r["term"]: r for r in ctx.R["battery"]["inter1"]}["female:married"]
    ax.text(1, n + 0.05, f"H8b is exploratory: p = {fmt_p(next(h['p'] for h in H if h['code'] == 'H8b'))} in M6, but Holm / Benjamini-Hochberg-adjusted "
            f"p = {inter['p_holm']:.3f} across the {len(ctx.R['battery']['inter1'])} interaction terms searched.", fontsize=9, color=INK2, va="center")
    top = headline(fig, f"Hypothesis scorecard: {nsup} of {n} supported",
                   "Directional hypotheses use one-sided p-values, the rest two-sided; HC3-robust, model M6 (H1 uses M4)", top=0.93)
    fig.subplots_adjust(left=0.01, right=0.99, top=top - 0.03, bottom=0.02); save(ctx, fig, "fig14_hypotheses.png")


def run(ctx):
    apply_theme()
    for f in (fig_dummy_effects, fig_predicted_profiles, fig_model_comparison, fig_marginal_returns, fig_hypotheses):
        f(ctx)
