"""Diagnostic figures for the preferred model M6: residual panel, partial residuals, multicollinearity."""
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from .config import LABEL
from .style import (BLUE, ORANGE, GREY, DGREY, INK2, apply_theme, headline, save, fmt_p)


def fig_residual_panel(ctx):
    S, m6, D, N = ctx.S, ctx.MOD["M6"], ctx.R["diag"], ctx.N
    res, fitted, hat, sr, cd = m6.resid.values, m6.fittedvalues.values, S.hat.values, S.sres.values, S.cook.values
    kpar = m6.model.exog.shape[1]
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 9.0))
    a = ax[0, 0]
    a.scatter(fitted, res, s=16, color=BLUE, alpha=0.4, linewidths=0)
    lo = lowess(res, fitted, frac=0.5); a.plot(lo[:, 0], lo[:, 1], color=ORANGE, lw=2.6)
    a.axhline(0, color=GREY, lw=1); a.set_xlabel("Fitted ln(sal)"); a.set_ylabel("Residual")
    a.set_title("(a) Residuals vs fitted: any pattern means mis-specification")
    a = ax[0, 1]
    (osm, osr), (slope, icpt, _) = stats.probplot(res, dist="norm")
    a.scatter(osm, osr, s=16, color=BLUE, alpha=0.5, linewidths=0); a.plot(osm, slope * osm + icpt, color=ORANGE, lw=2.6)
    a.set_xlabel("Theoretical normal quantiles"); a.set_ylabel("Residual quantiles")
    a.set_title("(b) Normal Q-Q: heavy tails (Jarque-Bera " + ("p<0.001" if D["jb"]["p"] < 0.001 else f"p={D['jb']['p']:.3f}") + ")")
    a = ax[1, 0]
    std = res / np.sqrt(m6.mse_resid * (1 - hat)); y_ = np.sqrt(np.abs(std))
    a.scatter(fitted, y_, s=16, color=BLUE, alpha=0.4, linewidths=0)
    lo = lowess(y_, fitted, frac=0.5); a.plot(lo[:, 0], lo[:, 1], color=ORANGE, lw=2.6)
    a.set_xlabel("Fitted ln(sal)"); a.set_ylabel("sqrt |standardised residual|")
    a.set_title("(c) Scale-location: a slope means non-constant variance")
    a = ax[1, 1]
    a.scatter(hat, sr, s=14 + 1400 * cd, color=BLUE, alpha=0.35, linewidths=0)
    a.axhline(0, color=GREY, lw=1); a.axvline(2 * kpar / N, color=GREY, lw=1, ls="--")
    for (_, r_), off in zip(S.sort_values("cook", ascending=False).head(4).iterrows(), [(10, 4), (-12, 12), (10, 4), (10, -3)]):
        a.annotate(f"id {int(r_.id)}", (r_.hat, r_.sres), textcoords="offset points", xytext=off, fontsize=9, color=DGREY, ha="center" if off[0] < 0 else "left")
    a.set_xlabel("Leverage (dashed: 2k/n)"); a.set_ylabel("Studentised residual")
    a.set_title("(d) Influence: bubble size = Cook's distance")
    top = headline(fig, "Residual diagnostics for the preferred model (M6)",
                   f"Heteroskedasticity-robust (HC3) standard errors are used throughout; classical Breusch-Pagan on the fitted values p = {fmt_p(D['bp_fitted_nonrobust']['p'])}", top=0.95)
    fig.tight_layout(rect=[0, 0, 1, top - 0.015]); save(ctx, fig, "fig6_diagnostics.png")


def fig_partial_residuals(ctx):
    S, m6 = ctx.S, ctx.MOD["M6"]
    res, pn = S.res.values, m6.params
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    for a, v, v2, lab in zip(ax, ["edu", "exper", "ten"], ["edu2", "exper2", "ten2"], ["Education", "Experience", "Tenure"]):
        cpr = res + pn[v] * S[v] + pn[v2] * S[v2]
        a.scatter(S[v], cpr, s=14, color=BLUE, alpha=0.3, linewidths=0)
        lo = lowess(cpr, S[v], frac=0.5); a.plot(lo[:, 0], lo[:, 1], color=ORANGE, lw=2.6, label="LOWESS smoother")
        xs = np.sort(S[v].unique())
        a.plot(xs, pn[v] * xs + pn[v2] * xs ** 2 + np.mean(cpr - pn[v] * S[v] - pn[v2] * S[v2]), color=DGREY, lw=1.8, ls="--",
               label="fitted quadratic")
        a.set_xlabel(lab); a.set_ylabel("Partial residual" if v == "edu" else "")
    ax[0].legend(loc="upper left", fontsize=9)
    top = headline(fig, "Is a quadratic the right shape?", "Partial-residual plots: the dashed fitted quadratic should follow the orange smoother")
    fig.tight_layout(rect=[0, 0, 1, top - 0.02]); save(ctx, fig, "fig7_partial_residuals.png")


def fig_vif(ctx):
    D = ctx.R["diag"]
    terms = list(D["vif_raw"])
    cent_map = {"edu": "edu_c", "edu2": "edu_c2", "exper": "exper_c", "exper2": "exper_c2", "ten": "ten_c", "ten2": "ten_c2"}
    fig, ax = plt.subplots(figsize=(9.2, 8.0))
    yy = np.arange(len(terms))[::-1]
    for y_, t in zip(yy, terms):
        r, c = D["vif_raw"][t], D["vif_cent"][cent_map.get(t, t)]
        ax.plot([r, c], [y_, y_], color="#c9c8c2", lw=1.6, zorder=1)
    ax.scatter([D["vif_raw"][t] for t in terms], yy, color=ORANGE, s=48, zorder=3, label="Raw (uncentred) model")
    ax.scatter([D["vif_cent"][cent_map.get(t, t)] for t in terms], yy, color=BLUE, s=48, zorder=3, label="Mean-centred model")
    for x_, lab in ((5, "5"), (10, "10: concern")):
        ax.axvline(x_, color=GREY, lw=1, ls="--"); ax.text(x_, len(terms) - 0.2, " " + lab, color=INK2, fontsize=9, va="bottom")
    ax.set_yticks(yy); ax.set_yticklabels([LABEL.get(t, t) for t in terms], fontsize=9.5)
    ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 30, 100]); ax.set_xticklabels(["1", "2", "5", "10", "30", "100"])
    ax.set_xlabel("Variance inflation factor (log scale)"); ax.grid(axis="y", visible=False)
    ax.legend(loc="center right")
    top = headline(fig, "Multicollinearity: squared terms inflate VIF, centring fixes it",
                   f"Condition index {D['cond_raw']:.0f} (raw) vs {D['cond_cent']:.0f} (centred); centring leaves the fit unchanged", top=0.94)
    fig.tight_layout(rect=[0, 0, 1, top - 0.02]); save(ctx, fig, "fig8_vif.png")


def run(ctx):
    apply_theme()
    for f in (fig_residual_panel, fig_partial_residuals, fig_vif):
        f(ctx)
