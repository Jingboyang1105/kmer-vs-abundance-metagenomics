"""Generate publication-style manuscript figures without internal headline titles.

The manuscript captions carry the explanatory text; figures should contain only
panel labels, axes, legends, and data annotations needed to interpret the plot.
"""

from pathlib import Path
import os
import shutil
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("MBEC_DATA_ROOT", str(REPO_ROOT)))
OUT = DATA_ROOT / "output" / "manuscript_figures"
OUT.mkdir(parents=True, exist_ok=True)
TIFF_OUT = DATA_ROOT / "output" / "manuscript_figures_tiff"
TIFF_OUT.mkdir(parents=True, exist_ok=True)
VECTOR_OUT = OUT.parent / "manuscript_figures_vector"
VECTOR_OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
SPRINGER_FULL_WIDTH = 174 * MM

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

COLORS = {
    "Abundance": "#0072b2",
    "K-mer": "#e69f00",
    "Fusion": "#009e73",
}
FUSION_COLORS = {
    "Abundance": "#003366",
    "K-mer": "#E6960A",
    "Fusion": "#1B7A6E",
}
CLASSIFIERS = ["LASSO-LR", "Random Forest", "SVM (RBF)", "XGBoost"]
CLF_SHORT = ["LASSO", "RF", "SVM", "XGB"]
METRICS = ["AUC", "AUPRC", "F1_macro", "MCC"]
METRIC_LABELS = {"AUC": "AUC", "AUPRC": "AUPRC", "F1_macro": "F1-macro", "MCC": "MCC"}


def _flatten_alpha(fig):
    """Replace semi-transparent colours with their opaque equivalent on white.

    EPS cannot store transparency, so without this a light violin body would be
    exported fully saturated and hide the box plot drawn on top of it.
    """
    from matplotlib.collections import Collection
    from matplotlib.colors import to_rgba, to_rgba_array
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    def blend(rgba):
        rgba = np.array(rgba, dtype=float).reshape(-1, 4)
        a = rgba[:, 3:4]
        out = rgba.copy()
        out[:, :3] = rgba[:, :3] * a + (1 - a)
        out[:, 3] = 1.0
        return out

    for art in fig.findobj():
        if isinstance(art, Patch):
            fc, ec = art.get_facecolor(), art.get_edgecolor()
            if fc[3] > 0:
                art.set_facecolor(tuple(blend(fc)[0]))
            if ec[3] > 0:
                art.set_edgecolor(tuple(blend(ec)[0]))
            art.set_alpha(None)
        elif isinstance(art, Line2D):
            alpha = art.get_alpha()
            if alpha is not None and alpha < 1:
                art.set_color(tuple(blend(to_rgba(art.get_color(), alpha))[0]))
                for get, put in ((art.get_markerfacecolor, art.set_markerfacecolor),
                                 (art.get_markeredgecolor, art.set_markeredgecolor)):
                    c = to_rgba(get(), alpha)
                    if c[3] > 0:
                        put(tuple(blend(c)[0]))
                art.set_alpha(None)
        elif isinstance(art, Collection) and art.get_array() is None:
            # Colormapped collections (heatmap cells, colorbar) are opaque already.
            fcs, ecs = art.get_facecolors(), art.get_edgecolors()
            art.set_alpha(None)
            if len(fcs):
                art.set_facecolors(blend(to_rgba_array(fcs)))
            if len(ecs):
                art.set_edgecolors(blend(to_rgba_array(ecs)))


def save(fig, name, dpi=600):
    _flatten_alpha(fig)
    path = OUT / name
    fig.savefig(path)
    # Editable vector copies (EPS for production, SVG for editing).
    for ext in ("eps", "svg"):
        vec_path = VECTOR_OUT / f"{Path(name).stem}.{ext}"
        fig.savefig(vec_path, format=ext)
        print(f"Saved {vec_path}")
    tiff_path = TIFF_OUT / f"{Path(name).stem}.tiff"
    fig.savefig(tiff_path, dpi=dpi, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"Saved {path}")
    print(f"Saved {tiff_path} at {dpi} dpi")


SUPP_DIRS = [
    DATA_ROOT / "output" / "supplementary_figures",
]


def save_supp(fig, name):
    for d in SUPP_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(d / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}.pdf/.png to {len(SUPP_DIRS)} supplementary-figure locations")


def fig2_stats_heatmap():
    df = pd.read_csv(DATA_ROOT / "data" / "supplementary_tables" / "Table_S4_Wilcoxon_tests.csv")
    task_order = ["Ibd", "Obesity", "Geography"]
    task_labels = {"Ibd": "IBD", "Obesity": "Obesity", "Geography": "Geography"}
    rows, mat, pmat = [], [], []
    for task in task_order:
        for clf in CLASSIFIERS:
            sub = df[(df["task"] == task) & (df["classifier"] == clf)].set_index("metric")
            rows.append(f"{task_labels[task]} | {clf.replace('Random Forest','RF').replace('SVM (RBF)','SVM').replace('XGBoost','XGB')}")
            mat.append([sub.loc[m, "delta"] for m in METRICS])
            pmat.append([sub.loc[m, "p_adj_2s"] for m in METRICS])
    mat = np.array(mat, dtype=float)
    pmat = np.array(pmat, dtype=float)
    cmap = LinearSegmentedColormap.from_list("delta", ["#0072b2", "#f7f7f7", "#e69f00"], N=256)
    vmax = max(np.nanmax(np.abs(mat)), 0.02)

    fig, ax = plt.subplots(figsize=(SPRINGER_FULL_WIDTH, 134 * MM))
    # pcolormesh (not imshow) so every cell is a vector rectangle, not an embedded bitmap.
    im = ax.pcolormesh(
        np.arange(mat.shape[1] + 1) - 0.5, np.arange(mat.shape[0] + 1) - 0.5, mat,
        cmap=cmap, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), rasterized=False,
    )
    ax.set_xlim(-0.5, mat.shape[1] - 0.5)
    ax.set_ylim(mat.shape[0] - 0.5, -0.5)
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.tick_params(length=0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            p = pmat[i, j]
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            color = "white" if abs(mat[i, j]) > vmax * 0.58 else "black"
            ax.text(j, i, f"{mat[i, j]:+.2f}{star}", ha="center", va="center", fontsize=8, color=color)
    for sep in [3.5, 7.5]:
        ax.axhline(sep, color="white", lw=2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.solids.set_rasterized(False)
    cbar.solids.set_edgecolor("face")
    cbar.set_label("Delta performance (k-mer minus abundance)")
    fig.text(0.01, 0.01, "* adjusted p<0.05; ** adjusted p<0.01; *** adjusted p<0.001; BH correction across 48 comparisons",
             fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "Fig1_statistical_tests_heatmap.pdf", dpi=600)


def _load_rounds(task):
    # All three tasks now use the matched abundance/k-mer sample intersection as
    # the primary comparison (Table 2 for IBD; Table S2/S3 for obesity/geography).
    md = DATA_ROOT / "data" / "matched_data"
    if task == "IBD":
        ab = pd.read_csv(md / "ibd_abundance_matched_repeated.csv")
        km = pd.read_csv(md / "ibd_kmer_matched_repeated.csv")
    elif task == "Obesity":
        ab = pd.read_csv(md / "obesity_abundance_matched_repeated.csv")
        km = pd.read_csv(md / "obesity_kmer_matched_repeated.csv")
    else:
        ab = pd.read_csv(md / "geography_abundance_matched_repeated.csv")
        km = pd.read_csv(md / "geography_kmer_matched_repeated.csv")
    return ab, km


def fig3_stability():
    fig, axes = plt.subplots(1, 2, figsize=(SPRINGER_FULL_WIDTH, 73 * MM))
    for ax, task in zip(axes, ["IBD", "Obesity"]):
        ab, km = _load_rounds(task)
        x = np.arange(len(CLASSIFIERS))
        width = 0.32
        for pipe, df, offset, color in [
            ("Abundance", ab, -width/2, COLORS["Abundance"]),
            ("K-mer", km, width/2, COLORS["K-mer"]),
        ]:
            for i, clf in enumerate(CLASSIFIERS):
                vals = df[df["classifier"] == clf]["AUC"].values
                xpos = x[i] + offset
                vp = ax.violinplot(vals, positions=[xpos], widths=width*0.85,
                                   showmedians=False, showextrema=False)
                for body in vp["bodies"]:
                    body.set_facecolor(color)
                    body.set_alpha(0.32)
                ax.boxplot(vals, positions=[xpos], widths=width*0.45, patch_artist=True,
                           medianprops=dict(color="black", lw=1.5),
                           boxprops=dict(facecolor=color, alpha=0.75),
                           whiskerprops=dict(color=color, lw=1.0),
                           capprops=dict(color=color, lw=1.0),
                           flierprops=dict(marker="o", markersize=2.3, markerfacecolor=color,
                                           markeredgecolor="none", alpha=0.45))
        ax.set_title(task)
        ax.set_xticks(x)
        ax.set_xticklabels(CLF_SHORT)
        ax.set_ylabel("AUC")
        ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.55)
        vals_all = np.r_[ab["AUC"].values, km["AUC"].values]
        ax.set_ylim(max(0.45, vals_all.min()-0.04), min(1.0, vals_all.max()+0.04))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.legend([Patch(facecolor=COLORS["Abundance"]), Patch(facecolor=COLORS["K-mer"])],
               ["Abundance", "K-mer"], loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save(fig, "Fig2_stability_boxplot.pdf", dpi=600)


def fig4_global_summary():
    tables = {
        "IBD": "Table_S1_IBD_full_metrics.csv",
        "Obesity": "Table_S2_Obesity_full_metrics.csv",
        "Geography": "Table_S3_Geography_full_metrics.csv",
    }
    fig, axes = plt.subplots(1, 3, figsize=(SPRINGER_FULL_WIDTH, 66 * MM), sharey=True)
    for ax, (task, fname) in zip(axes, tables.items()):
        df = pd.read_csv(DATA_ROOT / "data" / "supplementary_tables" / fname)
        df = df[df["pipeline"].isin(["Abundance", "K-mer"])]
        x = np.arange(len(CLASSIFIERS))
        width = 0.34
        for i, pipe in enumerate(["Abundance", "K-mer"]):
            sub = df[df["pipeline"] == pipe].set_index("classifier")
            means = [sub.loc[c, "AUC_mean"] for c in CLASSIFIERS]
            stds = [sub.loc[c, "AUC_sd"] for c in CLASSIFIERS]
            ax.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=2.5,
                   color=COLORS[pipe], alpha=0.9, edgecolor="white", lw=0.4)
        ax.set_title(task)
        ax.set_xticks(x)
        ax.set_xticklabels(CLF_SHORT)
        ax.set_ylim(0.45, 1.0)
        ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("AUC (mean +/- SD)")
    fig.legend([Patch(facecolor=COLORS["Abundance"]), Patch(facecolor=COLORS["K-mer"])],
               ["Abundance", "K-mer"], loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save_supp(fig, "FigS8_global_summary")


def fig5_fusion():
    paths = {
        "IBD": {
            "Abundance": DATA_ROOT / "data" / "matched_data" / "ibd_abundance_matched_repeated.csv",
            "K-mer": DATA_ROOT / "data" / "matched_data" / "ibd_kmer_matched_repeated.csv",
            "Fusion": DATA_ROOT / "data" / "fusion_results" / "fusion_ibd_all_rounds.csv",
        },
        "Obesity": {
            "Abundance": DATA_ROOT / "data" / "matched_data" / "obesity_abundance_matched_repeated.csv",
            "K-mer": DATA_ROOT / "data" / "matched_data" / "obesity_kmer_matched_repeated.csv",
            "Fusion": DATA_ROOT / "data" / "fusion_results" / "fusion_obesity_all_rounds.csv",
        },
        "Geography": {
            "Abundance": DATA_ROOT / "data" / "matched_data" / "geography_abundance_matched_repeated.csv",
            "K-mer": DATA_ROOT / "data" / "matched_data" / "geography_kmer_matched_repeated.csv",
            "Fusion": DATA_ROOT / "data" / "fusion_results" / "fusion_geography_all_rounds.csv",
        },
    }
    fig, axes = plt.subplots(1, 3, figsize=(SPRINGER_FULL_WIDTH, 73 * MM))
    width = 0.26
    offsets = {"Abundance": -width, "K-mer": 0, "Fusion": width}
    for ax, (task, pmap) in zip(axes, paths.items()):
        dfs = {k: pd.read_csv(v) for k, v in pmap.items()}
        x = np.arange(len(CLASSIFIERS))
        for pipe in ["Abundance", "K-mer", "Fusion"]:
            df = dfs[pipe]
            for i, clf in enumerate(CLASSIFIERS):
                vals = df[df["classifier"] == clf]["AUC"].values
                xpos = x[i] + offsets[pipe]
                color = FUSION_COLORS[pipe]
                vp = ax.violinplot(vals, positions=[xpos], widths=width*0.82,
                                   showmedians=False, showextrema=False)
                for body in vp["bodies"]:
                    body.set_facecolor(color)
                    body.set_alpha(0.28)
                ax.boxplot(vals, positions=[xpos], widths=width*0.42, patch_artist=True,
                           medianprops=dict(color="black", lw=1.4),
                           boxprops=dict(facecolor=color, alpha=0.75),
                           whiskerprops=dict(color=color, lw=1.0),
                           capprops=dict(color=color, lw=1.0),
                           flierprops=dict(marker="o", markersize=2.2,
                                           markerfacecolor=color, markeredgecolor="none", alpha=0.45))
        vals_all = np.concatenate([d["AUC"].values for d in dfs.values()])
        ax.set_title(task)
        ax.set_xticks(x)
        ax.set_xticklabels(CLF_SHORT)
        ax.set_ylabel("AUC")
        ax.set_ylim(max(0.45, vals_all.min()-0.04), min(1.0, vals_all.max()+0.04))
        ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.legend([Patch(facecolor=FUSION_COLORS[p]) for p in ["Abundance", "K-mer", "Fusion"]],
               ["Abundance", "K-mer", "Fusion"], loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save(fig, "Fig5_fusion_comparison_three_tasks.pdf", dpi=600)


def shannon_entropy(x):
    x = np.asarray(x, dtype=float)
    row_sums = x.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    p = x / row_sums
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.sum(np.where(p > 0, p * np.log(p), 0), axis=1)


def fig6_diversity():
    kdir = DATA_ROOT / "ibd" / "kmer_matrix"
    x_km = np.load(kdir / "kmer_matrix_k8_X.npy")
    ids = np.load(kdir / "kmer_matrix_k8_sample_ids.npy", allow_pickle=True)
    labels = np.load(kdir / "kmer_matrix_k8_labels.npy", allow_pickle=True)
    df_km = pd.DataFrame({"sample_id": ids, "kmer_entropy": shannon_entropy(x_km), "label": labels}).drop_duplicates("sample_id").set_index("sample_id")

    data_dir = DATA_ROOT / "ibd" / "data" / "raw"
    meta = pd.read_csv(data_dir / "hmp2_metadata_2018-08-20.csv", low_memory=False)
    meta = meta[(meta["data_type"] == "metagenomics") & (meta["diagnosis"].isin(["CD", "UC", "nonIBD"]))]
    meta["diagnosis"] = meta["diagnosis"].replace({"CD": "IBD", "UC": "IBD"})
    meta = meta[["External ID", "diagnosis"]].drop_duplicates("External ID").rename(columns={"External ID": "sample_id"}).set_index("sample_id")
    taxon = pd.read_csv(data_dir / "taxonomic_profiles_3.tsv.gz", sep="\t", index_col=0)
    sp_mask = taxon.index.map(lambda s: s.count("|") == 6 and "s__" in s and "t__" not in s)
    species = taxon[sp_mask].copy()
    species.index = species.index.map(lambda s: s.split("|")[-1])
    species = species.T
    species.index = species.index.str.replace("_profile$", "", regex=True)
    species.index.name = "sample_id"
    merged = meta.join(species.astype(float), how="inner")
    df_ab = pd.DataFrame({"abund_entropy": shannon_entropy(merged.drop(columns=["diagnosis"]).values),
                          "label": merged["diagnosis"].values}, index=merged.index)
    shared = df_km.index.intersection(df_ab.index)
    df = pd.DataFrame({
        "kmer_entropy": df_km.loc[shared, "kmer_entropy"].values,
        "abund_entropy": df_ab.loc[shared, "abund_entropy"].values,
        "label": df_km.loc[shared, "label"].values,
    }, index=shared)
    rp, pp = stats.pearsonr(df["kmer_entropy"], df["abund_entropy"])
    rs, ps = stats.spearmanr(df["kmer_entropy"], df["abund_entropy"])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(SPRINGER_FULL_WIDTH, 73 * MM),
        gridspec_kw={"width_ratios": [1.25, 0.75]}
    )
    pal = {"IBD": "#d55e00", "nonIBD": "#0072b2"}
    for grp, sub in df.groupby("label"):
        ax1.scatter(sub["abund_entropy"], sub["kmer_entropy"], c=pal[grp], label=grp, s=10, alpha=0.42, linewidths=0)
    xline = np.linspace(df["abund_entropy"].min(), df["abund_entropy"].max(), 200)
    slope, intercept, *_ = stats.linregress(df["abund_entropy"], df["kmer_entropy"])
    ax1.plot(xline, slope*xline + intercept, "k--", lw=1.1, alpha=0.7)
    ax1.set_title("A", loc="left", fontweight="bold")
    ax1.set_xlabel("Species Shannon entropy")
    ax1.set_ylabel("K-mer Shannon entropy")
    ax1.legend(loc="lower right", frameon=False, markerscale=1.5)

    order = ["nonIBD", "IBD"]
    data = [df[df["label"] == grp]["kmer_entropy"].values for grp in order]
    pos = np.arange(len(order))
    vp = ax2.violinplot(data, positions=pos, widths=0.72, showmedians=False, showextrema=False)
    for body, grp in zip(vp["bodies"], order):
        body.set_facecolor(pal[grp])
        body.set_alpha(0.28)
        body.set_edgecolor("none")
    bp = ax2.boxplot(data, positions=pos, widths=0.32, patch_artist=True,
                     medianprops=dict(color="black", lw=1.3),
                     boxprops=dict(facecolor="white", edgecolor="black", lw=0.8),
                     whiskerprops=dict(color="black", lw=0.8),
                     capprops=dict(color="black", lw=0.8),
                     flierprops=dict(marker="o", markersize=2, markerfacecolor="#777", markeredgecolor="none", alpha=0.35))
    for patch, grp in zip(bp["boxes"], order):
        patch.set_facecolor(pal[grp])
        patch.set_alpha(0.6)
    ax2.set_title("B", loc="left", fontweight="bold")
    ax2.set_xticks(pos)
    ax2.set_xticklabels(order)
    ax2.set_ylabel("K-mer Shannon entropy")
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_supp(fig, "FigS9_kmer_diversity_correlation")
    print(f"FigS9: n={len(df)}, Pearson r={rp:.3f}, Spearman r={rs:.3f}")


def fig7_attribution():
    kdir = DATA_ROOT / "ibd" / "kmer_matrix"
    x_km = np.load(kdir / "kmer_matrix_k8_X.npy")
    kmer_index = np.load(kdir / "kmer_matrix_k8_kmer_index.npy")
    sample_ids = np.load(kdir / "kmer_matrix_k8_sample_ids.npy", allow_pickle=True)
    labels = np.load(kdir / "kmer_matrix_k8_labels.npy", allow_pickle=True)

    row_sums = x_km.sum(axis=1, keepdims=True)
    x_km = x_km / np.where(row_sums == 0, 1, row_sums)
    _, uniq_idx = np.unique(sample_ids, return_index=True)
    x_km = x_km[uniq_idx]
    sample_ids = sample_ids[uniq_idx]
    labels = labels[uniq_idx]
    y = (labels == "IBD").astype(int)

    data_dir = DATA_ROOT / "ibd" / "data" / "raw"
    taxon = pd.read_csv(data_dir / "taxonomic_profiles_3.tsv.gz", sep="\t", index_col=0)
    sp_mask = taxon.index.map(lambda s: s.count("|") == 6 and "s__" in s and "t__" not in s)
    species = taxon[sp_mask].copy()
    species.index = species.index.map(lambda s: s.split("|")[-1])
    species = species.T
    species.index = species.index.str.replace("_profile$", "", regex=True)
    species.index.name = "sample_id"
    species = species.astype(float)
    species = species.loc[:, (species > 0).mean() >= 0.10]

    km_df = pd.DataFrame(x_km, index=sample_ids)
    shared = km_df.index.intersection(species.index)
    km_shared = km_df.loc[shared].values
    ab_shared = species.loc[shared].values
    sp_names = species.columns.tolist()
    y_shared = pd.Series(y, index=sample_ids).loc[shared].values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pvals = stats.ttest_ind(
            km_shared[y_shared == 0], km_shared[y_shared == 1],
            equal_var=False, axis=0
        )
    pvals = np.nan_to_num(pvals, nan=1.0)
    _, p_adj, _, _ = multipletests(pvals, method="fdr_bh")
    delta = km_shared[y_shared == 1].mean(axis=0) - km_shared[y_shared == 0].mean(axis=0)
    top_idx = np.argsort(np.where(delta > 0, p_adj, 1.0))[:50]
    bot_idx = np.argsort(np.where(delta < 0, p_adj, 1.0))[:50]

    def decode_kmer(idx, k=8):
        bases = []
        for _ in range(k):
            bases.append("ACGT"[idx & 3])
            idx >>= 2
        return "".join(reversed(bases))

    def revcomp(seq):
        return "".join({"A": "T", "T": "A", "C": "G", "G": "C"}[b] for b in reversed(seq))

    def idx_to_seq(col_idx):
        seq = decode_kmer(int(kmer_index[col_idx]))
        rc = revcomp(seq)
        return seq if seq <= rc else rc

    top_seqs = [idx_to_seq(i) for i in top_idx]
    bot_seqs = [idx_to_seq(i) for i in bot_idx]

    def compute_corr_matrix(indices, seqs):
        rows = []
        for ci in indices:
            kv = km_shared[:, ci]
            corrs = []
            for si in range(ab_shared.shape[1]):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r, _ = stats.spearmanr(kv, ab_shared[:, si])
                corrs.append(0.0 if np.isnan(r) else r)
            rows.append(corrs)
        return pd.DataFrame(rows, index=seqs, columns=sp_names)

    corr_top = compute_corr_matrix(top_idx, top_seqs)
    corr_bot = compute_corr_matrix(bot_idx, bot_seqs)
    mean_corr_top = corr_top.mean(axis=0).sort_values(ascending=False)
    mean_corr_bot = corr_bot.mean(axis=0).sort_values()
    sp_show_top = mean_corr_top.head(8).index.tolist() + mean_corr_top.tail(4).index.tolist()
    sp_show_bot = mean_corr_bot.head(4).index.tolist() + mean_corr_bot.tail(8).index.tolist()
    sp_show = list(dict.fromkeys(sp_show_top + sp_show_bot))[:16]

    n_show = 16
    corr_combined = pd.concat([corr_top.iloc[:n_show], corr_bot.iloc[:n_show]])[sp_show]

    def short(s):
        s = s.replace("s__", "")
        parts = s.split("_")
        if len(parts) >= 2:
            return f"{parts[0][0]}. {' '.join(parts[1:])}"
        return s

    mat = corr_combined.values
    row_labels = [f"up {s}" for s in top_seqs[:n_show]] + [f"down {s}" for s in bot_seqs[:n_show]]
    fig, ax = plt.subplots(figsize=(SPRINGER_FULL_WIDTH, 145 * MM))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-0.45, vmax=0.45)
    cbar = fig.colorbar(im, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_label("Spearman r")
    ax.set_xticks(np.arange(len(sp_show)))
    ax.set_xticklabels([short(s) for s in sp_show], rotation=45, ha="right", fontstyle="italic")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i, label in enumerate(ax.get_yticklabels()):
        label.set_color("#d55e00" if i < n_show else "#0072b2")
    ax.axhline(n_show - 0.5, color="black", lw=1.2)
    ax.set_xlabel("Species")
    ax.set_ylabel("Differential 8-mer")
    fig.tight_layout()
    save_supp(fig, "FigS10_kmer_attribution_heatmap")


def fig_cohort_stress():
    """Cross-cohort stress tests: IBD LOSiteO, obesity LOSO, geography study-held-out."""
    fig, axes = plt.subplots(1, 3, figsize=(SPRINGER_FULL_WIDTH, 68 * MM))
    ax1, ax2, ax3 = axes

    # Panel A: IBD leave-one-site-out (Table S15), aggregated across the 4 classifiers.
    s15 = pd.read_csv(DATA_ROOT / "data" / "supplementary_tables" / "Table_S15_IBD_leave_one_site_out_summary.csv")
    pipe_map = {
        "Abundance (4 matched sites)": "Abundance",
        "K-mer": "K-mer",
        "Fusion": "Fusion",
    }
    rows = []
    for raw, label in pipe_map.items():
        sub = s15[s15["Pipeline"] == raw]
        rows.append((label, sub["AUC_mean"].mean(), sub["AUC_mean"].std()))
    labels_a = [r[0] for r in rows]
    means_a = [r[1] for r in rows]
    sds_a = [r[2] for r in rows]
    colors_a = [COLORS[l] for l in labels_a]
    ax1.bar(labels_a, means_a, yerr=sds_a, capsize=3, color=colors_a, alpha=0.9, edgecolor="white")
    ax1.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax1.set_ylabel("AUC")
    ax1.set_ylim(0.3, 0.85)
    ax1.set_title("a", loc="left", fontweight="bold")
    ax1.set_xlabel("IBD: leave-one-site-out\n(4 matched sites, mean of 4 classifiers)", fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: obesity leave-one-study-out (Table S5, XGBoost only -- the classifier used for this check).
    s5 = pd.read_csv(DATA_ROOT / "data" / "supplementary_tables" / "Table_S5_LOSO_obesity_summary.csv")
    order_b = ["Abundance", "K-mer", "Fusion"]
    s5 = s5.set_index("pipeline").loc[order_b].reset_index()
    colors_b = [COLORS[l] for l in order_b]
    ax2.bar(order_b, s5["AUC_mean"], yerr=s5["AUC_std"], capsize=3, color=colors_b, alpha=0.9, edgecolor="white")
    ax2.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax2.set_ylabel("AUC")
    ax2.set_ylim(0.3, 0.85)
    ax2.set_title("b", loc="left", fontweight="bold")
    ax2.set_xlabel("Obesity: leave-one-study-out\n(XGBoost, mean of 29 held-out folds)", fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel C: geography study-held-out, AUC vs F1-macro dissociation (Table S18).
    s18 = pd.read_csv(DATA_ROOT / "data" / "supplementary_tables" / "Table_S18_Geography_study_held_out_abundance.csv")
    pipe_map_c = {
        "Abundance (full cohort, n=22,397)": "Abundance",
        "K-mer (pilot pool, n=1,453/1,587)": "K-mer",
    }
    x = np.arange(2)
    width = 0.32
    auc_vals, auc_sds, f1_vals, f1_sds = [], [], [], []
    for raw, label in pipe_map_c.items():
        sub = s18[s18["pipeline"] == raw]
        auc_vals.append(sub["AUC_mean"].mean())
        auc_sds.append(sub["AUC_mean"].std())
        f1_vals.append(sub["F1_macro_mean"].mean())
        f1_sds.append(sub["F1_macro_mean"].std())
    ax3.bar(x - width / 2, auc_vals, width, yerr=auc_sds, capsize=3, label="AUC",
            color="#4C72B0", alpha=0.9, edgecolor="white")
    ax3.bar(x + width / 2, f1_vals, width, yerr=f1_sds, capsize=3, label="F1-macro",
            color="#C44E52", alpha=0.9, edgecolor="white")
    ax3.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax3.set_xticks(x)
    ax3.set_xticklabels(list(pipe_map_c.values()))
    ax3.set_ylabel("Score")
    ax3.set_ylim(0, 0.9)
    ax3.set_title("c", loc="left", fontweight="bold")
    ax3.set_xlabel("Geography: study-held-out\n(mean of 4 classifiers)", fontsize=8)
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2, frameon=False, fontsize=8)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "Fig3_cohort_stress_tests.pdf", dpi=600)


def fig_fairness_diagnostics():
    """Fairness diagnostics: obesity within-study reversal, matched-dimension sensitivity."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(SPRINGER_FULL_WIDTH, 68 * MM))

    # Panel A: obesity within-study vs. mixed-pool k-mer-minus-abundance AUC gap (Table S19).
    s19 = pd.read_csv(DATA_ROOT / "data" / "supplementary_tables" / "Table_S19_Obesity_batch_signal_regime_comparison.csv")
    study_order = [
        "All studies (mixed pool)",
        "MetaCardis_2020_a (within-study)",
        "ZeeviD_2015 (within-study)",
        "NielsenHB_2014 (within-study)",
        "AsnicarF_2021 (within-study)",
        "HMP_2012 (within-study)",
        "LeChatelierE_2013 (within-study)",
    ]
    short_labels = ["Mixed\npool", "MetaCardis", "ZeeviD", "NielsenHB", "AsnicarF", "HMP", "LeChatelier"]
    means, sds = [], []
    for s in study_order:
        sub = s19[s19["study"] == s]
        means.append(sub["AUC_gap_kmer_minus_abundance"].mean())
        sds.append(sub["AUC_gap_kmer_minus_abundance"].std())
    bar_colors = ["#6B7280"] + ["#e69f00" if m > 0 else "#0072b2" for m in means[1:]]
    ax1.bar(short_labels, means, yerr=sds, capsize=3, color=bar_colors, alpha=0.9, edgecolor="white")
    ax1.axhline(0, color="black", lw=0.9)
    ax1.set_ylabel("AUC gap (k-mer \u2212 abundance)")
    ax1.set_title("a", loc="left", fontweight="bold")
    ax1.set_xlabel("Obesity: mixed-pool vs. within-study\n(mean of 4 classifiers)", fontsize=8)
    ax1.tick_params(axis="x", rotation=40, labelsize=8)
    for lbl in ax1.get_xticklabels():
        lbl.set_ha("right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: matched-dimension AUC gap at 250 dimensions, all three tasks (Tables S20, S23, S24).
    dim_files = {
        "IBD": ("Table_S20_IBD_matched_dimension_check.csv", "Matched_dimension", "Classifier", "Gap_kmer_minus_abundance"),
        "Obesity": ("Table_S23_Obesity_matched_dimension_check.csv", "matched_dim", "classifier", "gap_kmer_minus_abundance"),
        "Geography": ("Table_S24_Geography_matched_dimension_check.csv", "matched_dim", "classifier", "gap_kmer_minus_abundance"),
    }
    task_means, task_sds = [], []
    for task, (fname, dimcol, clfcol, gapcol) in dim_files.items():
        df = pd.read_csv(DATA_ROOT / "data" / "supplementary_tables" / fname)
        sub = df[df[dimcol] == 250]
        task_means.append(sub[gapcol].mean())
        task_sds.append(sub[gapcol].std())
    bar_colors_b = ["#e69f00" if m > 0 else "#0072b2" for m in task_means]
    ax2.bar(list(dim_files.keys()), task_means, yerr=task_sds, capsize=3, color=bar_colors_b,
            alpha=0.9, edgecolor="white")
    ax2.axhline(0, color="black", lw=0.9)
    ax2.set_ylabel("AUC gap (k-mer \u2212 abundance)")
    ax2.set_title("b", loc="left", fontweight="bold")
    ax2.set_xlabel("Matched-dimension sensitivity\nat 250 dims (mean of 4 classifiers)", fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, "Fig4_fairness_diagnostics.pdf", dpi=600)



def main():
    fig2_stats_heatmap()
    fig3_stability()
    fig_cohort_stress()
    fig_fairness_diagnostics()
    fig5_fusion()
    fig4_global_summary()

    kmer_matrix_dir = DATA_ROOT / "ibd" / "kmer_matrix"
    try:
        fig6_diversity()
    except FileNotFoundError:
        print(
            f"Skipping FigS9/kmer_diversity_correlation: local IBD k-mer matrix not found "
            f"at {kmer_matrix_dir} -- run ibd/extract_kmers.py + ibd/build_kmer_matrix.py "
            f"first (see README)"
        )
    try:
        fig7_attribution()
    except FileNotFoundError:
        print(
            f"Skipping FigS10/kmer_attribution_heatmap: local IBD k-mer matrix not found "
            f"at {kmer_matrix_dir} -- run ibd/extract_kmers.py + ibd/build_kmer_matrix.py "
            f"first (see README)"
        )


if __name__ == "__main__":
    main()
