"""One-page summary dashboard: headline numbers plus the four charts that tell the story."""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from .effects import hypothesis_verdict
from .plots_models import EFFECT_GROUPS, BLUE_SOFT
from .style import BLUE, DGREY, INK, INK2, LGREY, apply_theme, headline, footer, save


def _tile(ax, value, label, note=""):
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.04), 0.96, 0.92, boxstyle="round,pad=0,rounding_size=0.06", fc="white", ec=LGREY, lw=1.2,
                                transform=ax.transAxes))
    ax.add_patch(FancyBboxPatch((0.02, 0.04), 0.022, 0.92, boxstyle="square,pad=0", fc=BLUE, ec="none", transform=ax.transAxes))
    ax.text(0.10, 0.62, value, fontsize=26, fontweight="bold", color=INK, transform=ax.transAxes, va="center")
    ax.text(0.10, 0.28, label, fontsize=10.5, color=INK, transform=ax.transAxes, va="center")
    if note:
        ax.text(0.10, 0.11, note, fontsize=8.5, color=INK2, transform=ax.transAxes, va="center")


def run(ctx):
    apply_theme()
    R, L, F = ctx.R, ctx.R["lin"], ctx.FIT
    nsup = sum(hypothesis_verdict(h)[0] for h in R["hyp"])
    fig = plt.figure(figsize=(16, 10.2))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.35, 1, 1], hspace=0.42, wspace=0.32,
                  left=0.17, right=0.985, top=0.66, bottom=0.095)
    gt = GridSpec(1, 5, figure=fig, wspace=0.08, left=0.03, right=0.985, top=0.87, bottom=0.72)
    tiles = [(f"{ctx.N}", "employees analysed", f"of {R['clean']['n_raw']} raw records"),
             (f"{F['M6']['adj_r2']:.2f}", "adjusted R-squared", f"model M6, R-squared {F['M6']['r2']:.3f}"),
             (f"{L['metro']['pct']:+.0f}%", "metro-area pay premium", f"95% CI {L['metro']['pct_lo']:+.0f}% to {L['metro']['pct_hi']:+.0f}%"),
             (f"{L['gender_gap_unmarried']['pct']:+.0f}%", "gender gap, unmarried", f"married: {L['gender_gap_married']['pct']:+.0f}%"),
             (f"{nsup} / {len(R['hyp'])}", "hypotheses supported", "see fig14 for the full scorecard")]
    for i, t in enumerate(tiles):
        _tile(fig.add_subplot(gt[0, i]), *t)

    # (a) significant effects only
    a = fig.add_subplot(gs[:, 0])
    items = [(lab, L[k]) for _, grp in EFFECT_GROUPS for lab, k in grp if L[k]["p"] < 0.05]
    items.sort(key=lambda x: x[1]["pct"])
    yy = np.arange(len(items))
    for y, (lab, d) in zip(yy, items):
        a.plot([d["pct_lo"], d["pct_hi"]], [y, y], color=BLUE, lw=2.4, solid_capstyle="round")
        a.scatter(d["pct"], y, color=BLUE, s=60, zorder=3)
        if d["pct"] > 0:
            a.text(d["pct_hi"] + 1.2, y, f"{d['pct']:+.0f}%", va="center", ha="left", fontsize=9.5, color=DGREY)
        else:
            a.text(d["pct_lo"] - 1.2, y, f"{d['pct']:+.0f}%", va="center", ha="right", fontsize=9.5, color=DGREY)
    a.axvline(0, color=DGREY, lw=1)
    a.set_yticks(yy); a.set_yticklabels([i[0] for i in items], fontsize=9.5); a.grid(axis="y", visible=False)
    a.set_xlabel("Difference in hourly salary (%), 95% CI"); a.set_title("(a) Significant pay differences", loc="left")
    a.set_xlim(min(i[1]["pct_lo"] for i in items) - 9, max(i[1]["pct_hi"] for i in items) + 8)

    # (b) fit progression
    b = fig.add_subplot(gs[0, 1])
    ks = list(F); vals = [F[k]["adj_r2"] for k in ks]
    b.bar(ks, vals, color=[BLUE_SOFT] * (len(ks) - 1) + [BLUE], width=0.66)
    for i, v in enumerate(vals):
        b.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9, color=DGREY)
    b.set_ylim(0, max(vals) * 1.15); b.grid(axis="x", visible=False); b.set_title("(b) Adjusted R-squared by model")

    # (c)-(e) returns to human capital
    for pos, (v, xs, ttl) in zip([(0, 2), (1, 1), (1, 2)], [("edu", [8, 10, 12, 14, 16, 18], "(c) Education"),
                                                           ("exper", [0, 5, 10, 20, 30, 40], "(d) Experience"),
                                                           ("ten", [0, 5, 10, 20, 30], "(e) Tenure")]):
        c = fig.add_subplot(gs[pos])
        d = [L[f"{v}_marg_{x}"] for x in xs]
        c.fill_between(xs, [i["pct_lo"] for i in d], [i["pct_hi"] for i in d], color=BLUE, alpha=0.15, lw=0)
        c.plot(xs, [i["pct"] for i in d], color=BLUE, lw=2.4, marker="o", ms=6)
        c.axhline(0, color=DGREY, lw=1); c.set_title(ttl + ": % pay gain per extra year"); c.set_xlabel("Years")
    top = headline(fig, "Salary determinants at HCL Technologies: the picture in one page",
                   "OLS on ln(hourly salary), HC3-robust inference. Model M6 unless stated. Associations, not causal effects", top=0.9)
    footer(fig, "Source: ITEmp.xlsx after rule-based cleaning (see fig11 and cleaning_log.csv). RESET still rejects the functional form, so claims are associational.")
    save(ctx, fig, "00_summary_dashboard.png")
