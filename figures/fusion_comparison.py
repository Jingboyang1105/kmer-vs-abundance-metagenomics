"""
Three-task Fusion comparison figure.
Abundance vs K-mer vs Fusion across IBD, Obesity, Geography.
Output: figures/fusion_comparison_three_tasks.png
        figures/fusion_comparison_three_tasks.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

ROOT        = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

COLORS = {
    "Abundance": "#0072b2",
    "K-mer":     "#e69f00",
    "Fusion":    "#009e73",
}
CLASSIFIERS = ["LASSO-LR", "Random Forest", "SVM (RBF)", "XGBoost"]
CLF_SHORT   = ["LASSO", "RF", "SVM", "XGB"]


def load_task(paths):
    """Load (abundance, kmer, fusion) DataFrames for one task, each with 'classifier' and 'AUC'."""
    out = {}
    for name, (path, pipe_col) in paths.items():
        df = pd.read_csv(path)
        if pipe_col is not None:
            df = df[df["pipeline"] == pipe_col].copy()
        out[name] = df[["classifier", "AUC"]].copy()
    return out


TASKS = {
    "IBD": {
        "Abundance": (
            os.path.join(ROOT, "../ibd/results/ibd_abundance_repeated.csv"),
            "Abundance (Line1)"),
        "K-mer": (
            os.path.join(ROOT, "../ibd/results/ibd_kmer_repeated.csv"),
            "K-mer (Line2)"),
        "Fusion": (
            os.path.join(ROOT, "../ibd/results/fusion_ibd_all_rounds.csv"),
            "Fusion (Abundance+K-mer)"),
    },
    "Obesity": {
        "Abundance": (
            os.path.join(ROOT, "../obesity/results/abundance_all_rounds.csv"),
            "abundance (Line1-style)"),
        "K-mer": (
            os.path.join(ROOT, "../obesity/results/kmer_all_rounds.csv"),
            "k-mer (Line2-style)"),
        "Fusion": (
            os.path.join(ROOT, "../obesity/results/fusion_all_rounds.csv"),
            "Fusion (Abundance+K-mer)"),
    },
    "Geography\n(30-country)": {
        "Abundance": (
            os.path.join(ROOT, "../geography/results/abundance_all_rounds.csv"),
            "abundance (Line1-style)"),
        "K-mer": (
            os.path.join(ROOT, "../geography/results/kmer_all_rounds.csv"),
            "k-mer (Line2-style)"),
        "Fusion": (
            os.path.join(ROOT, "../geography/results/fusion_geo_all_rounds.csv"),
            "Fusion (Abundance+K-mer)"),
    },
}


def plot_task(ax, dfs, title):
    x      = np.arange(len(CLASSIFIERS))
    width  = 0.26
    offsets = {"Abundance": -width, "K-mer": 0, "Fusion": width}

    for pipe_name, offset in offsets.items():
        df  = dfs[pipe_name]
        col = COLORS[pipe_name]
        for i, clf in enumerate(CLASSIFIERS):
            vals = df[df["classifier"] == clf]["AUC"].values
            if len(vals) == 0:
                continue
            xpos = x[i] + offset
            vp = ax.violinplot(vals, positions=[xpos], widths=width * 0.82,
                               showmedians=False, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(col)
                body.set_alpha(0.30)
            ax.boxplot(vals, positions=[xpos], widths=width * 0.42,
                       patch_artist=True, notch=False,
                       medianprops=dict(color="black", lw=1.8),
                       boxprops=dict(facecolor=col, alpha=0.80),
                       whiskerprops=dict(color=col, lw=1.1),
                       capprops=dict(color=col, lw=1.1),
                       flierprops=dict(marker="o", markersize=2.5,
                                       markerfacecolor=col, alpha=0.5))

    ax.set_xticks(x)
    ax.set_xticklabels(CLF_SHORT)
    ax.set_ylabel("AUC")
    ax.set_title(title)
    ax.axhline(0.5, color="gray", ls="--", lw=0.7, alpha=0.5)
    vals_all = np.concatenate([
        df[df["classifier"].isin(CLASSIFIERS)]["AUC"].values
        for df in dfs.values()
    ])
    ymin = max(0.45, float(np.nanmin(vals_all)) - 0.04)
    ymax = min(1.0, float(np.nanmax(vals_all)) + 0.04)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6))

    for ax, (task_name, paths) in zip(axes, TASKS.items()):
        dfs = load_task(paths)
        plot_task(ax, dfs, task_name)

    legend_handles = [Patch(facecolor=COLORS[p], label=p) for p in COLORS]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.03), frameon=False)
    plt.tight_layout()

    out = os.path.join(FIGURES_DIR, "fusion_comparison_three_tasks.png")
    out_pdf = os.path.join(FIGURES_DIR, "fusion_comparison_three_tasks.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")
    print(f"Saved → {out_pdf}")

    # Print summary table
    print(f"\n{'='*75}")
    print(f"{'Task':<16} {'Pipe':<12} {'Classifier':<16} {'AUC mean':>9} {'AUC std':>8}")
    print(f"{'='*75}")
    for task_name, paths in TASKS.items():
        dfs = load_task(paths)
        for pipe_name in ["Abundance", "K-mer", "Fusion"]:
            df = dfs[pipe_name]
            for clf in CLASSIFIERS:
                vals = df[df["classifier"] == clf]["AUC"].values
                if len(vals):
                    print(f"{task_name.replace(chr(10), ' '):<16} {pipe_name:<12} "
                          f"{clf:<16} {vals.mean():>9.3f} {vals.std(ddof=1):>8.3f}")


if __name__ == "__main__":
    main()
