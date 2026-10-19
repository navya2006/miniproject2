"""Robustness figures: do the conclusions survive other estimators, other samples, other cleaning rules?

Heat maps use one encoding throughout: sign(b) * -log10(p), blue = significantly positive, red = significantly
negative, grey = no evidence. The cell text always carries the coefficient and stars, so colour is never alone.
"""
import numpy as np
from matplotlib import pyplot as plt
from .config import LABEL, KEYT, stars
from .style import (BLUE, ORANGE, GREY, DGREY, INK2, SURFACE, DIVERGING, apply_theme, headline, save,
                    signed_log_p)


def _evidence_heatmap(ax, rows, cols, cell, col_labels, row_labels):
    """cell(row, col) -> (b, p) or None."""
    Z = np.full((len(rows), len(cols)), np.nan)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = cell(r, c)
            if v is not None and not np.isnan(v[0]):
                Z[i, j] = signed_log_p(*v)
    ax.imshow(np.ma.masked_invalid(Z), cmap=DIVERGING, vmin=-3, vmax=3, aspect="auto")
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = cell(r, c)
            if v is None or np.isnan(v[0]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8.5, color=INK2)
                continue
            b, p = v
            dark = abs(Z[i, j]) > 1.9
            ax.text(j, i, f"{b:+.4f}{stars(p)}", ha="center", va="center", fontsize=8.8, color="white" if dark else DGREY,
                    fontweight="bold" if p < 0.05 else "normal")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(col_labels, fontsize=9.2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(row_labels, fontsize=10)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True); ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2); ax.grid(which="major", visible=False)
    for s_ in ax.spines.values():
        s_.set_visible(False)


def _wrap(s, w=18):
    import textwrap
    return "\n".join(textwrap.wrap(s, w, break_long_words=False))


def fig_robustness(ctx):
    R = ctx.R["robust"]
    names = list(R["specs"])
    fig, ax = plt.subplots(figsize=(14.5, 7.8))
    _evidence_heatmap(ax, KEYT, names, lambda t, n: (R["specs"][n][t]["b"], R["specs"][n][t]["p"]) if not np.isnan(R["specs"][n][t]["b"] or np.nan) else None,
                      [_wrap(n) for n in names], [LABEL[t] for t in KEYT])
    agree = sum(all((R["specs"][n][t]["p"] < 0.05) == (R["specs"]["Baseline M6 (HC3)"][t]["p"] < 0.05)
                    for n in names if not np.isnan(R["specs"][n][t]["b"] or np.nan)) for t in KEYT)
    top = headline(fig, "Robustness: how stable are the M6 coefficients?",
                   f"{agree} of {len(KEYT)} key terms keep the same 5% verdict in every specification. Stars: * p<0.10, ** p<0.05, *** p<0.01. Blue positive, red negative, grey no evidence",
                   top=0.9)
    fig.tight_layout(rect=[0, 0, 1, top - 0.02]); save(ctx, fig, "fig15_robustness.png")


def fig_cleaning_sensitivity(ctx):
    C = ctx.R["clean_sens"]
    names = list(C)
    fig, ax = plt.subplots(figsize=(11.5, 7.6))
    _evidence_heatmap(ax, KEYT, names, lambda t, n: (C[n]["terms"][t]["b"], C[n]["terms"][t]["p"]),
                      [_wrap(f"{n} (n={C[n]['n']})", 20) for n in names], [LABEL[t] for t in KEYT])
    top = headline(fig, "Sensitivity to the cleaning thresholds",
                   "M6 refitted after re-cleaning the data differently; the last column gauges sensitivity only and is not adopted", top=0.9)
    fig.tight_layout(rect=[0, 0, 1, top - 0.02]); save(ctx, fig, "fig16_cleaning_sensitivity.png")


def fig_interaction_search(ctx):
    B = ctx.R["battery"]
    rows = [dict(label=r["hypothesis"], p=r["p"], holm=r["p_holm"], bh=r["p_bh"]) for r in B["inter1"]]
    rows += [dict(label="Education squared (curvature)", p=r["p"], holm=r["p_holm"], bh=r["p_bh"]) for r in B["quad"] if r["term"] == "edu2"]
    rows += [dict(label="Dependents squared (curvature)", p=r["p"], holm=r["p_holm"], bh=r["p_bh"]) for r in B["quad"] if r["term"] == "dep2"]
    rows.sort(key=lambda r: -r["p"])
    fig, ax = plt.subplots(figsize=(11, 7.4))
    yy = np.arange(len(rows))
    lp = lambda p: -np.log10(np.clip(p, 1e-4, 1))
    for y, r in zip(yy, rows):
        ax.plot([lp(r["p"]), lp(r["holm"])], [y, y], color="#c9c8c2", lw=1.6, zorder=1)
    ax.scatter([lp(r["p"]) for r in rows], yy, color=BLUE, s=58, zorder=3, label="Raw p (HC3)")
    ax.scatter([lp(r["holm"]) for r in rows], yy, facecolor=SURFACE, edgecolor=ORANGE, linewidth=2, s=58, zorder=3,
               label="Holm-adjusted p")
    ax.axvline(lp(0.05), color=GREY, lw=1, ls="--"); ax.text(lp(0.05), len(rows) - 0.2, " p = 0.05", color=INK2, fontsize=9.5)
    ax.set_yticks(yy); ax.set_yticklabels([r["label"] for r in rows], fontsize=10); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Evidence against 'no effect':  -log10(p)   (further right = stronger)")
    ax.legend(loc="lower right")
    surv = [r["label"].split(" (")[0].lower() for r in rows if r["holm"] < 0.05]
    verdict = ("Only " + " and ".join(surv) + (" survives" if len(surv) == 1 else " survive") + " the multiple-testing correction") if surv \
        else "No term survives the multiple-testing correction"
    top = headline(fig, "Searching for curvature and interactions",
                   f"Each candidate term added one at a time to M4. {verdict}", top=0.93)
    fig.tight_layout(rect=[0, 0, 1, top - 0.02]); save(ctx, fig, "fig17_interaction_search.png")


def run(ctx):
    apply_theme()
    for f in (fig_robustness, fig_cleaning_sensitivity, fig_interaction_search):
        f(ctx)
