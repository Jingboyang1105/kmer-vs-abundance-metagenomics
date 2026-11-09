from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle, Polygon


OUT = Path("graphical_abstract")
OUT.mkdir(exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)


def box(ax, xy, w, h, fc, ec, title, body=None, title_color="#222222", fs=10.5):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        fc=fc,
        ec=ec,
        lw=1.4,
        zorder=2,
    )
    ax.add_patch(patch)
    x, y = xy
    ax.text(
        x + w / 2,
        y + h - 0.018,
        title,
        ha="center",
        va="top",
        fontsize=fs,
        fontweight="bold",
        color=title_color,
        zorder=3,
    )
    if body:
        ax.text(
            x + w / 2,
            y + h - 0.058,
            body,
            ha="center",
            va="top",
            fontsize=fs - 2.4,
            color=title_color,
            linespacing=1.05,
            zorder=3,
        )
    return patch


def arrow(ax, start, end, color="#666666", lw=1.6, rad=0.0, ms=12):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=ms,
            lw=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
        )
    )


def dna_icon(ax, x, y, scale=1.0, color="#0F766E"):
    # Compact read pile plus canonical 8-mer cue.
    for i, off in enumerate([0.00, 0.022, 0.044]):
        ax.add_patch(Rectangle((x, y + off * scale), 0.13 * scale, 0.010 * scale, fc=color, ec="none", alpha=0.85))
        for j in range(4):
            ax.add_patch(
                Rectangle(
                    (x + (0.018 + j * 0.028) * scale, y + off * scale),
                    0.006 * scale,
                    0.010 * scale,
                    fc="#FFFFFF",
                    ec="none",
                    alpha=0.85,
                )
            )
    ax.text(x + 0.066 * scale, y - 0.015 * scale, "8-mer", ha="center", va="top", fontsize=7.4, color=color)


def abundance_icon(ax, x, y, scale=1.0, color="#9A3412"):
    heights = [0.035, 0.065, 0.045, 0.085, 0.055]
    for i, h in enumerate(heights):
        ax.add_patch(
            Rectangle(
                (x + i * 0.024 * scale, y),
                0.014 * scale,
                h * scale,
                fc=color,
                ec="none",
                alpha=0.78,
            )
        )
    ax.plot([x - 0.004 * scale, x + 0.12 * scale], [y, y], color=color, lw=1)
    ax.text(x + 0.058 * scale, y - 0.016 * scale, "species", ha="center", va="top", fontsize=7.4, color=color)


def shield_icon(ax, x, y, scale=1.0):
    pts = [
        (x, y + 0.070 * scale),
        (x + 0.055 * scale, y + 0.090 * scale),
        (x + 0.110 * scale, y + 0.070 * scale),
        (x + 0.098 * scale, y + 0.020 * scale),
        (x + 0.055 * scale, y),
        (x + 0.012 * scale, y + 0.020 * scale),
    ]
    ax.add_patch(Polygon(pts, closed=True, fc="#FEF3C7", ec="#B45309", lw=1.3))
    ax.text(x + 0.055 * scale, y + 0.047 * scale, "!", ha="center", va="center", fontsize=13, fontweight="bold", color="#92400E")


def plus_icon(ax, x, y, scale=1.0, color="#4338CA"):
    ax.add_patch(Circle((x, y), 0.035 * scale, fc="#EEF2FF", ec=color, lw=1.4))
    ax.plot([x - 0.017 * scale, x + 0.017 * scale], [y, y], color=color, lw=2)
    ax.plot([x, x], [y - 0.017 * scale, y + 0.017 * scale], color=color, lw=2)


def main():
    # Springer full-width friendly: 174 mm x 96 mm.
    fig_w = 174 / 25.4
    fig_h = 96 / 25.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=600)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    teal = "#0F766E"
    teal_bg = "#DDF7F0"
    rust = "#9A3412"
    rust_bg = "#FDEBE4"
    indigo = "#4338CA"
    indigo_bg = "#EEF2FF"
    gray = "#374151"
    gray_bg = "#F3F4F6"
    amber = "#B45309"
    amber_bg = "#FEF3C7"
    green = "#166534"
    green_bg = "#DCFCE7"

    ax.text(
        0.5,
        0.965,
        "Alignment-free k-mer vs. species-abundance profiling for metagenomic classification",
        ha="center",
        va="center",
        fontsize=11.6,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.5,
        0.925,
        "Matched-sample comparison across three tasks: is an apparent k-mer advantage robust, or a cohort artifact?",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#4B5563",
    )

    box(
        ax,
        (0.030, 0.680),
        0.220,
        0.150,
        gray_bg,
        "#9CA3AF",
        "Public cohorts",
        "iHMP/HMP2\ncuratedMetagenomicData v3",
        gray,
        fs=9.9,
    )

    box(ax, (0.320, 0.755), 0.200, 0.125, teal_bg, teal, "K-mer layer", "canonical 8-mer\nrelative frequencies", teal, fs=9.4)
    box(ax, (0.320, 0.605), 0.200, 0.135, rust_bg, rust, "Abundance layer", "CLR species +\ndiversity features", rust, fs=9.4)

    box(
        ax,
        (0.600, 0.630),
        0.200,
        0.178,
        indigo_bg,
        indigo,
        "Fusion",
        "best AUC: 0.731 / 0.710 / 0.971\n(IBD / obesity / geography)",
        indigo,
        fs=9.0,
    )

    ax.text(
        0.5,
        0.560,
        "Matched samples: 21/48 significant by Wilcoxon; 10/48 (all geographic origin) after correcting for overlapping splits",
        ha="center",
        va="center",
        fontsize=7.3,
        color="#1F2937",
        fontweight="bold",
    )

    box(ax, (0.030, 0.290), 0.280, 0.170, green_bg, green, "IBD", "k-mer AUC +0.02 to +0.08,\nnot significant; shrinks or\nreverses at matched dimension", green, fs=9.4)
    box(ax, (0.360, 0.290), 0.280, 0.170, amber_bg, amber, "Obesity", "Random-split gain reverses\nwithin 4 of 6 studies;\nLOSO AUC ≈ 0.55", amber, fs=9.4)
    box(ax, (0.690, 0.290), 0.280, 0.170, teal_bg, teal, "Geography", "Classifier-dependent;\ncohort-structure probe\n(Cramér's V = 0.91)", teal, fs=9.4)

    box(
        ax,
        (0.080, 0.075),
        0.840,
        0.155,
        "#FFFFFF",
        "#6B7280",
        "Take-home message",
        "k-mers: a candidate complementary layer, not demonstrably superior;\nrandom-split evaluation can overstate cross-cohort transferability",
        "#1F2937",
        fs=9.4,
    )

    arrow(ax, (0.250, 0.800), (0.320, 0.805), color="#6B7280")
    arrow(ax, (0.250, 0.720), (0.320, 0.665), color="#6B7280")
    arrow(ax, (0.520, 0.805), (0.600, 0.770), color=teal, rad=-0.10)
    arrow(ax, (0.520, 0.665), (0.600, 0.680), color=rust, rad=0.08)
    arrow(ax, (0.660, 0.663), (0.220, 0.460), color="#6B7280", rad=0.08)
    arrow(ax, (0.700, 0.663), (0.500, 0.460), color="#6B7280", rad=0.0)
    arrow(ax, (0.740, 0.663), (0.780, 0.460), color="#6B7280", rad=-0.08)
    arrow(ax, (0.500, 0.290), (0.500, 0.235), color="#6B7280")

    for ext in ["pdf", "eps", "svg", "png", "tiff"]:
        path = OUT / f"graphical_abstract_MBEC.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
