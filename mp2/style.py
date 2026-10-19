"""Plotting theme shared by every figure.

Palette (validated with the dataviz palette checker, light surface):
  BLUE   primary series / "significant"          ORANGE  contrast series / highlight
  GREY   de-emphasised / "not significant"       RED     negative pole of the diverging scale
Diverging scale: blue <-> red around a neutral grey midpoint. Text always wears text colours, never series colours.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
GREY, DGREY, LGREY, MID = "#8a8984", "#3b3a37", "#e5e4df", "#f0efec"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE_LIGHT = "#cde2fb"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#9ec5f4", "#2a78d6", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div_blue_red", [RED, MID, BLUE])


def apply_theme():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": GREY, "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": LGREY,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlesize": 11.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "legend.frameon": False, "figure.dpi": 100})


def headline(fig, title, subtitle=None, top=0.93):
    """Left-aligned figure title and an optional one-line subtitle; callers leave room above `top`."""
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.012, 0.985 - 0.045 * (6.0 / fig.get_figheight()), subtitle, ha="left", va="top", fontsize=10, color=INK2)
    return top


def footer(fig, text):
    fig.text(0.012, 0.008, text, ha="left", va="bottom", fontsize=8, color=INK2)


def save(ctx, fig, name):
    fig.savefig(os.path.join(ctx.fig_dir, name), dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def sig_color(p, alpha=0.05):
    return BLUE if p < alpha else GREY


def signed_log_p(b, p, cap=3.0):
    """sign(b) * -log10(p), capped: the diverging encoding of 'direction and strength of evidence'."""
    import numpy as np
    return float(np.sign(b) * min(-np.log10(max(p, 1e-12)), cap))


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def p_label(p):
    """'p<0.001' or 'p=0.025'."""
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"
