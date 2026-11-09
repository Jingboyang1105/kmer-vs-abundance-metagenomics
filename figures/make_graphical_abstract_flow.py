"""Flowchart-style graphical abstract (vector; PDF/EPS/SVG/PNG).

Text matches the manuscript's conclusions: the Wilcoxon count is shown with
its dependence-corrected count, and IBD is shown as directional, not robust.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "graphical_abstract"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "font.family": "DejaVu Sans"})

NAVY = "#1B3A5C"
MID_BLUE = "#4A90D9"
BAND_BLUE = "#23629F"
DARK = "#14304A"
ARROW = "#1F5F8B"
NEUTRAL_BG, NEUTRAL_FG = "#FCEFC7", "#8A5A00"
RED_BG, RED_FG = "#FBE3DC", "#A12A1E"


def box(ax, x, y, w, h, fc, title, sub, tc="white", sc="white", tfs=15, sfs=12.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.018",
                                fc=fc, ec="none"))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
            fontsize=tfs, fontweight="bold", color=tc)
    ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
            fontsize=sfs, color=sc, linespacing=1.25)


def arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16,
                                 color=ARROW, lw=1.6, shrinkA=0, shrinkB=0))


def main():
    # MBEC GA box: 32.93 mm wide x 37.63 mm high (portrait, w/h = 0.875).
    fig = plt.figure(figsize=(5.6, 6.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.30, 0.875, 0.40, 0.105, NAVY, "Public cohorts", "HMP2 · cMD3", tfs=15, sfs=12.5)
    box(ax, 0.04, 0.705, 0.42, 0.105, MID_BLUE, "k-mer", "8-mer frequencies", tfs=15, sfs=12.5)
    box(ax, 0.54, 0.705, 0.42, 0.105, MID_BLUE, "Abundance", "CLR + diversity", tfs=15, sfs=12.5)
    box(ax, 0.04, 0.525, 0.92, 0.115, BAND_BLUE, "Matched comparison",
        "k-mer ahead · robust only in geography", tfs=15, sfs=12.5)
    box(ax, 0.02, 0.305, 0.30, 0.145, NEUTRAL_BG, "IBD", "ahead ·\nnot significant",
        tc=NEUTRAL_FG, sc=NEUTRAL_FG, tfs=14, sfs=12)
    box(ax, 0.35, 0.305, 0.30, 0.145, RED_BG, "Obesity", "batch ·\n≈ chance",
        tc=RED_FG, sc=RED_FG, tfs=14, sfs=12)
    box(ax, 0.68, 0.305, 0.30, 0.145, RED_BG, "Geography", "study\nconfound",
        tc=RED_FG, sc=RED_FG, tfs=14, sfs=12)
    box(ax, 0.04, 0.03, 0.92, 0.19, DARK, "Random-split edge\n≠ transferable",
        "stress-test before clinical use", sc="#C9D3DE", tfs=15, sfs=12.5)

    arrow(ax, (0.44, 0.875), (0.30, 0.81))
    arrow(ax, (0.56, 0.875), (0.70, 0.81))
    arrow(ax, (0.25, 0.705), (0.25, 0.64))
    arrow(ax, (0.75, 0.705), (0.75, 0.64))
    for x in (0.17, 0.50, 0.83):
        arrow(ax, (x, 0.525), (x, 0.45))
        arrow(ax, (x, 0.305), (x, 0.22))

    for ext in ("pdf", "eps", "png"):
        kw = {"dpi": 300} if ext == "png" else {}
        fig.savefig(OUT / f"Graphical_Abstract_Patterson.{ext}", **kw)
    plt.close(fig)


if __name__ == "__main__":
    main()
