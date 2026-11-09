"""
K-mer Shannon entropy vs abundance alpha diversity correlation (IBD dataset).
Output: figures/kmer_diversity_correlation.png
        figures/kmer_diversity_correlation.pdf
        kmer_diversity_stats.txt
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

ROOT        = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def shannon_entropy(X):
    """Row-wise Shannon entropy on frequency vectors."""
    X = np.array(X, dtype=float)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    p = X / row_sums
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.sum(np.where(p > 0, p * np.log(p), 0), axis=1)
    return h


def main():
    # ── Load k-mer matrix ──────────────────────────────────────────────
    kmer_dir = os.path.join(ROOT, "../ibd/kmer_counts")
    X_km      = np.load(os.path.join(kmer_dir, "kmer_matrix_k8_X.npy"))
    sample_ids_km = np.load(os.path.join(kmer_dir, "kmer_matrix_k8_sample_ids.npy"),
                            allow_pickle=True)
    labels_km = np.load(os.path.join(kmer_dir, "kmer_matrix_k8_labels.npy"),
                        allow_pickle=True)

    kmer_entropy = shannon_entropy(X_km)
    df_km = pd.DataFrame({
        "sample_id":    sample_ids_km,
        "kmer_entropy": kmer_entropy,
        "label":        labels_km,
    }).drop_duplicates("sample_id").set_index("sample_id")

    # ── Load abundance / compute alpha diversity ───────────────────────
    data_dir   = os.path.join(ROOT, "../ibd/data/raw")
    meta_path  = os.path.join(data_dir, "hmp2_metadata_2018-08-20.csv")
    taxon_path = os.path.join(data_dir, "taxonomic_profiles_3.tsv.gz")

    meta = pd.read_csv(meta_path, low_memory=False)
    meta = meta[meta["data_type"] == "metagenomics"]
    meta = meta[meta["diagnosis"].isin(["CD", "UC", "nonIBD"])]
    meta["diagnosis"] = meta["diagnosis"].replace({"CD": "IBD", "UC": "IBD"})
    meta = (meta[["External ID", "diagnosis"]]
            .drop_duplicates("External ID")
            .rename(columns={"External ID": "sample_id"})
            .set_index("sample_id"))

    print("Loading taxonomic profiles…")
    taxon = pd.read_csv(taxon_path, sep="\t", index_col=0)
    sp_mask = taxon.index.map(
        lambda x: x.count("|") == 6 and "s__" in x and "t__" not in x)
    species = taxon[sp_mask].copy()
    species.index = species.index.map(lambda x: x.split("|")[-1])
    species = species.T
    species.index = species.index.str.replace("_profile$", "", regex=True)
    species.index.name = "sample_id"
    species = species.astype(float)

    merged = meta.join(species, how="inner")
    feat   = merged.drop(columns=["diagnosis"])
    abund_entropy = shannon_entropy(feat.values)
    df_ab = pd.DataFrame({
        "abund_entropy": abund_entropy,
        "label":         merged["diagnosis"].values,
    }, index=merged.index)

    # ── Merge on shared samples ────────────────────────────────────────
    shared = df_km.index.intersection(df_ab.index)
    df = pd.DataFrame({
        "kmer_entropy":  df_km.loc[shared, "kmer_entropy"].values,
        "abund_entropy": df_ab.loc[shared, "abund_entropy"].values,
        "label":         df_km.loc[shared, "label"].values,
    }, index=shared)
    print(f"Shared samples: {len(df)}")

    # ── Correlation ────────────────────────────────────────────────────
    r_p, p_p = stats.pearsonr(df["kmer_entropy"], df["abund_entropy"])
    r_s, p_s = stats.spearmanr(df["kmer_entropy"], df["abund_entropy"])

    stats_txt = (
        f"N = {len(df)} shared samples\n"
        f"Pearson  r = {r_p:.3f}  p = {p_p:.2e}\n"
        f"Spearman r = {r_s:.3f}  p = {p_s:.2e}\n"
    )
    print(stats_txt)

    out_txt = os.path.join(ROOT, "kmer_diversity_stats.txt")
    with open(out_txt, "w") as f:
        f.write(stats_txt)
    print(f"Stats → {out_txt}")

    # ── Plot ───────────────────────────────────────────────────────────
    pal   = {"IBD": "#d55e00", "nonIBD": "#0072b2"}
    fig, axes = plt.subplots(
        1, 2, figsize=(8.6, 3.7),
        gridspec_kw={"width_ratios": [1.25, 0.75]}
    )
    ax = axes[0]

    for grp, sub in df.groupby("label"):
        ax.scatter(sub["abund_entropy"], sub["kmer_entropy"],
                   c=pal[grp], label=grp, s=11, alpha=0.42, linewidths=0)

    # regression line
    x_line = np.linspace(df["abund_entropy"].min(), df["abund_entropy"].max(), 200)
    slope, intercept, *_ = stats.linregress(df["abund_entropy"], df["kmer_entropy"])
    ax.plot(x_line, slope * x_line + intercept, "k--", lw=1.2, alpha=0.7)

    ax.set_xlabel("Species Shannon entropy")
    ax.set_ylabel("K-mer Shannon entropy")
    ax.set_title("A", loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Diagnosis-specific k-mer entropy distribution. This supports the
    # biological interpretation that low k-mer entropy is concentrated in IBD.
    ax_b = axes[1]
    order = ["nonIBD", "IBD"]
    positions = np.arange(len(order))
    data = [df.loc[df["label"] == grp, "kmer_entropy"].values for grp in order]
    vp = ax_b.violinplot(data, positions=positions, widths=0.72,
                         showmedians=False, showextrema=False)
    for body, grp in zip(vp["bodies"], order):
        body.set_facecolor(pal[grp])
        body.set_alpha(0.28)
        body.set_edgecolor("none")
    bp = ax_b.boxplot(data, positions=positions, widths=0.32, patch_artist=True,
                      medianprops=dict(color="black", lw=1.4),
                      boxprops=dict(facecolor="white", edgecolor="black", lw=0.9),
                      whiskerprops=dict(color="black", lw=0.8),
                      capprops=dict(color="black", lw=0.8),
                      flierprops=dict(marker="o", markersize=2.0,
                                      markerfacecolor="#777777", markeredgecolor="none", alpha=0.35))
    for patch, grp in zip(bp["boxes"], order):
        patch.set_facecolor(pal[grp])
        patch.set_alpha(0.60)
    ax_b.set_xticks(positions)
    ax_b.set_xticklabels(order)
    ax_b.set_ylabel("K-mer Shannon entropy")
    ax_b.set_title("B", loc="left", fontweight="bold")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    ax.legend(loc="lower right", frameon=False, markerscale=1.6)
    plt.tight_layout()

    out_fig = os.path.join(FIGURES_DIR, "kmer_diversity_correlation.png")
    out_pdf = os.path.join(FIGURES_DIR, "kmer_diversity_correlation.pdf")
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out_fig}")
    print(f"Figure → {out_pdf}")


if __name__ == "__main__":
    main()
