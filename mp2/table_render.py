"""Render every results table as a clean PNG image (the 'presentable format' for slides and documents)."""
import os
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .style import BLUE, INK, INK2, GREY, SURFACE, apply_theme
from .table_builders import build_all

ROW_H, HEAD_PAD, PAD, CHAR_W, FONT = 0.34, 0.16, 0.22, 0.092, 10.5
ZEBRA = "#f4f3ef"


def _col_widths(df, fonts=1.0):
    widths = []
    for c in df.columns:
        longest = max([max((len(x) for x in str(c).split("\n")), default=0)] + [len(str(v)) for v in df[c]])
        widths.append(min(max(longest * CHAR_W * fonts + 2 * PAD, 0.8), 6.4))
    return widths


def render_table(tab, path):
    df = tab.df
    widths = _col_widths(df)
    W = sum(widths) + 0.5
    head_lines = max(str(c).count("\n") + 1 for c in df.columns)
    head_h = ROW_H * head_lines + HEAD_PAD
    title_h = 0.62
    note_lines = textwrap.wrap(tab.note, int(W / 0.075)) if tab.note else []
    note_h = 0.14 + 0.22 * len(note_lines)
    H = title_h + head_h + ROW_H * len(df) + note_h + 0.2
    fig = plt.figure(figsize=(W, H), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    x0 = 0.25
    ax.text(x0, 0.34, tab.title, fontsize=14.5, fontweight="bold", color=INK, va="center")
    y = title_h
    ax.add_patch(Rectangle((x0, y), sum(widths), head_h, fc=BLUE, ec="none"))
    x = x0
    for c, w in zip(df.columns, widths):
        j = list(df.columns).index(c)
        left = j in tab.left_cols
        ax.text(x + (PAD * 0.7 if left else w / 2), y + head_h / 2, str(c), color="white", fontsize=FONT, fontweight="bold",
                va="center", ha="left" if left else "center", linespacing=1.25)
        x += w
    y += head_h
    for i, row in enumerate(df.itertuples(index=False)):
        if (i // tab.group) % 2 == 1:
            ax.add_patch(Rectangle((x0, y), sum(widths), ROW_H, fc=ZEBRA, ec="none"))
        x = x0
        se_row = tab.group == 2 and str(row[0]) == ""
        for j, (v, w) in enumerate(zip(row, widths)):
            v = str(v)
            left = j in tab.left_cols
            color, weight = INK, "bold" if i in tab.bold_rows else "normal"
            if se_row:
                color = INK2
            if v.startswith("✓"):
                color, weight = BLUE, "bold"
            elif v.startswith("– "):
                color = INK2
            ax.text(x + (PAD * 0.7 if left else w / 2), y + ROW_H / 2 - (0.03 if not se_row else 0.06), v, color=color, fontsize=FONT - (1 if se_row else 0),
                    fontweight=weight, va="center", ha="left" if left else "center")
            x += w
        y += ROW_H
    ax.plot([x0, x0 + sum(widths)], [y, y], color=GREY, lw=1.2)
    for k, ln in enumerate(note_lines):
        ax.text(x0, y + 0.2 + 0.22 * k, ln, fontsize=9, color=INK2, va="center", style="italic")
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def run(ctx):
    apply_theme()
    ctx.tables = build_all(ctx)
    for tab in ctx.tables:
        render_table(tab, os.path.join(ctx.img_tab_dir, f"table_{tab.key}.png"))
