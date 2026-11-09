"""
Extra figures for paper:
  1. stability_boxplot.png  — 10-round AUC violin+box, IBD & Obesity
  2. ibd_tsne.png           — t-SNE side-by-side: abundance vs k-mer feature spaces
  3. round_trajectory.png   — per-round AUC line plot for IBD (variance story)
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

ROOT        = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

COLORS = {"Abundance (Line1)": "#0072b2", "K-mer (Line2)": "#e69f00"}
GROUP_COLORS = {1: "#d55e00", 0: "#0072b2"}
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


# ── Figure 1: Stability boxplot ───────────────────────────────────────────────

def fig_stability():
    tasks = {
        "IBD": {
            "abund": os.path.join(ROOT, "../ibd/results/ibd_abundance_repeated.csv"),
            "kmer":  os.path.join(ROOT, "../ibd/results/ibd_kmer_repeated.csv"),
        },
        "Obesity": {
            "abund": os.path.join(ROOT, "../obesity/results/abundance_all_rounds.csv"),
            "kmer":  os.path.join(ROOT, "../obesity/results/kmer_all_rounds.csv"),
        },
    }

    classifiers = ["LASSO-LR", "Random Forest", "SVM (RBF)", "XGBoost"]
    clf_short   = ["LASSO", "RF", "SVM", "XGB"]
    fig, axes   = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=False)

    for ax, (task, paths) in zip(axes, tasks.items()):
        ab = pd.read_csv(paths["abund"])
        km = pd.read_csv(paths["kmer"])

        x      = np.arange(len(classifiers))
        width  = 0.32
        offset = {"Abundance (Line1)": -width / 2, "K-mer (Line2)": width / 2}

        for pipe, df in [("Abundance (Line1)", ab), ("K-mer (Line2)", km)]:
            col = COLORS[pipe]
            for i, clf in enumerate(classifiers):
                vals = df[df["classifier"] == clf]["AUC"].values
                xpos = x[i] + offset[pipe]
                vp   = ax.violinplot(vals, positions=[xpos], widths=width * 0.85,
                                     showmedians=False, showextrema=False)
                for body in vp["bodies"]:
                    body.set_facecolor(col)
                    body.set_alpha(0.35)
                bp = ax.boxplot(vals, positions=[xpos], widths=width * 0.45,
                                patch_artist=True, notch=False,
                                medianprops=dict(color="black", lw=1.8),
                                boxprops=dict(facecolor=col, alpha=0.75),
                                whiskerprops=dict(color=col, lw=1.2),
                                capprops=dict(color=col, lw=1.2),
                                flierprops=dict(marker="o", markersize=3,
                                                markerfacecolor=col, alpha=0.5))

        ax.set_xticks(x)
        ax.set_xticklabels(clf_short)
        ax.set_ylabel("AUC")
        ax.set_title(task)
        ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
        vals_all = np.concatenate([ab["AUC"].values, km["AUC"].values])
        ax.set_ylim(max(0.45, float(np.nanmin(vals_all)) - 0.04),
                    min(1.0, float(np.nanmax(vals_all)) + 0.04))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_handles = [Patch(facecolor=COLORS[p], label=p) for p in COLORS]
    fig.legend(handles=legend_handles, labels=["Abundance", "K-mer"], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.03), frameon=False)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "stability_boxplot.png")
    out_pdf = os.path.join(FIGURES_DIR, "stability_boxplot.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Figure 1 → {out}")
    print(f"Figure 1 → {out_pdf}")


# ── Figure 2: IBD t-SNE side-by-side ─────────────────────────────────────────

def fig_ibd_tsne():
    # ── Load abundance features ──
    data_dir = os.path.join(ROOT, "../ibd/data/raw")
    meta_path = os.path.join(data_dir, "hmp2_metadata_2018-08-20.csv")
    taxon_path = os.path.join(data_dir, "taxonomic_profiles_3.tsv.gz")

    meta = pd.read_csv(meta_path, low_memory=False)
    meta = meta[meta["data_type"] == "metagenomics"]
    meta = meta[meta["diagnosis"].isin(["CD", "UC", "nonIBD"])]
    meta["diagnosis"] = meta["diagnosis"].replace({"CD": "IBD", "UC": "IBD"})
    meta = meta[["External ID", "diagnosis"]].drop_duplicates("External ID")
    meta.rename(columns={"External ID": "sample_id"}, inplace=True)
    meta = meta.set_index("sample_id")

    print("  Loading taxonomic profiles…")
    taxon = pd.read_csv(taxon_path, sep="\t", index_col=0)
    sp_mask = taxon.index.map(
        lambda x: x.count("|") == 6 and "s__" in x and "t__" not in x)
    species = taxon[sp_mask].copy()
    species.index = species.index.map(lambda x: x.split("|")[-1])
    species = species.T
    species.index = species.index.str.replace("_profile$", "", regex=True)
    species.index.name = "sample_id"
    species = species.astype(float)

    merged_ab = meta.join(species, how="inner")
    y_ab      = (merged_ab["diagnosis"] == "IBD").astype(int).values
    feat_ab   = merged_ab.drop(columns=["diagnosis"])
    prev      = (feat_ab > 0).mean()
    feat_ab   = feat_ab.loc[:, prev >= 0.10]

    X_ab = feat_ab.values + 1e-6
    log_X = np.log(X_ab)
    X_ab_clr = log_X - log_X.mean(axis=1, keepdims=True)
    X_ab_clr = StandardScaler().fit_transform(X_ab_clr)

    # ── Load k-mer features ──
    kmer_dir = os.path.join(ROOT, "../ibd/kmer_counts")
    X_km      = np.load(os.path.join(kmer_dir, "kmer_matrix_k8_X.npy"))
    labels_km = np.load(os.path.join(kmer_dir, "kmer_matrix_k8_labels.npy"),
                        allow_pickle=True)
    y_km = (labels_km == "IBD").astype(int)

    # Top-500 by variance for t-SNE speed
    var_idx = np.argsort(X_km.var(axis=0))[::-1][:500]
    X_km_sel = StandardScaler().fit_transform(X_km[:, var_idx])

    # ── t-SNE ──
    print("  Running t-SNE on abundance features…")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsne_ab = TSNE(n_components=2, perplexity=40, random_state=42,
                       max_iter=1000, learning_rate="auto", init="pca")
        Z_ab = tsne_ab.fit_transform(X_ab_clr)

    print("  Running t-SNE on k-mer features…")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsne_km = TSNE(n_components=2, perplexity=40, random_state=42,
                       max_iter=1000, learning_rate="auto", init="pca")
        Z_km = tsne_km.fit_transform(X_km_sel)

    # ── Plot ──
    pal = GROUP_COLORS
    labels_str = {1: "IBD", 0: "nonIBD"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, Z, y, title in [
        (axes[0], Z_ab, y_ab, "Abundance (CLR-transformed species)"),
        (axes[1], Z_km, y_km, "K-mer (8-mer frequency, top-500 by variance)"),
    ]:
        for cls in [0, 1]:
            mask = y == cls
            ax.scatter(Z[mask, 0], Z[mask, 1],
                       c=pal[cls], label=labels_str[cls],
                       s=12, alpha=0.55, linewidths=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.legend(markerscale=2, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("IBD vs nonIBD — t-SNE feature space comparison (HMP2)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "ibd_tsne.png")
    out_pdf = os.path.join(FIGURES_DIR, "ibd_tsne.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Figure 2 → {out}")
    print(f"Figure 2 → {out_pdf}")


# ── Figure 3: Round trajectory line plot (IBD) ───────────────────────────────

def fig_round_trajectory():
    ab = pd.read_csv(os.path.join(ROOT, "../ibd/results/ibd_abundance_repeated.csv"))
    km = pd.read_csv(os.path.join(ROOT, "../ibd/results/ibd_kmer_repeated.csv"))

    classifiers = ["LASSO-LR", "Random Forest", "SVM (RBF)", "XGBoost"]
    clf_short   = {"LASSO-LR": "LASSO", "Random Forest": "RF",
                   "SVM (RBF)": "SVM", "XGBoost": "XGB"}

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    rounds = sorted(ab["round"].unique())

    for ax, clf in zip(axes, classifiers):
        ab_vals = ab[ab["classifier"] == clf].sort_values("round")["AUC"].values
        km_vals = km[km["classifier"] == clf].sort_values("round")["AUC"].values

        ax.plot(rounds, ab_vals, "o-", color=COLORS["Abundance (Line1)"],
                lw=1.5, ms=5, label="Abundance", alpha=0.85)
        ax.plot(rounds, km_vals, "s-", color=COLORS["K-mer (Line2)"],
                lw=1.5, ms=5, label="K-mer", alpha=0.85)

        ax.axhline(ab_vals.mean(), color=COLORS["Abundance (Line1)"],
                   ls="--", lw=1.0, alpha=0.5)
        ax.axhline(km_vals.mean(), color=COLORS["K-mer (Line2)"],
                   ls="--", lw=1.0, alpha=0.5)
        ax.axhline(0.5, color="gray", ls=":", lw=0.7, alpha=0.5)

        ax.set_title(clf_short[clf], fontsize=11, fontweight="bold")
        ax.set_xlabel("Round", fontsize=9)
        ax.set_xticks(rounds)
        ax.set_xticklabels(rounds, fontsize=7)
        if ax == axes[0]:
            ax.set_ylabel("AUC", fontsize=10)

    handles = [
        plt.Line2D([0], [0], color=COLORS["Abundance (Line1)"], marker="o", lw=1.5, label="Abundance"),
        plt.Line2D([0], [0], color=COLORS["K-mer (Line2)"],     marker="s", lw=1.5, label="K-mer"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.08), fontsize=10)
    fig.suptitle("IBD: per-round AUC across 10 repeated splits\n"
                 "(dashed = mean; narrower spread = more stable)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "round_trajectory_ibd.png")
    out_pdf = os.path.join(FIGURES_DIR, "round_trajectory_ibd.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Figure 3 → {out}")
    print(f"Figure 3 → {out_pdf}")


if __name__ == "__main__":
    print("=== Figure 1: Stability boxplot ===")
    fig_stability()

    print("\n=== Figure 2: IBD t-SNE ===")
    fig_ibd_tsne()

    print("\n=== Figure 3: Round trajectory (IBD) ===")
    fig_round_trajectory()

    print("\nDone.")
