#!/usr/bin/env python3
"""
Build supplementary figures (Fig S1-S6) for the supplementary robustness analyses,
matching the manuscript's existing figure style (matplotlib, embedded fonts,
consistent classifier colors/order).

Outputs -> $MBEC_DATA_ROOT/output/supplementary_figures/FigS1_*.pdf ... FigS6_*.pdf
(+ .png previews); MBEC_DATA_ROOT defaults to the repo root.
"""
import os
import numpy as np
import pandas as pd
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

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# REPO_ROOT: supplementary_experiments/_common -> up two levels -> repo root.
REPO_ROOT = os.path.dirname(EXP)
DATA_ROOT = os.environ.get("MBEC_DATA_ROOT", REPO_ROOT)
SUPP_TABLES_DIR = os.path.join(DATA_ROOT, "data", "supplementary_tables")
# Single shared supplementary-figure output location (same one
# figures/make_clean_manuscript_figures.py writes FigS8-S10 into).
OUT = os.path.join(DATA_ROOT, "output", "supplementary_figures")
os.makedirs(OUT, exist_ok=True)

CLASSIFIERS = ["LASSO-LR", "Random Forest", "SVM (RBF)", "XGBoost"]
CLF_SHORT = {"LASSO-LR": "LASSO", "Random Forest": "RF", "SVM (RBF)": "SVM", "XGBoost": "XGB"}
CLF_COLORS = {"LASSO-LR": "#4C72B0", "Random Forest": "#DD8452",
              "SVM (RBF)": "#55A868", "XGBoost": "#C44E52"}


def savefig(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {name}.pdf/.png")


# ── Fig S1: IBD leave-one-site-out ──────────────────────────────────────────
def fig_s1():
    df = pd.read_csv(f"{EXP}/01_ibd_loso/results/ibd_loso_summary.csv")
    pipelines = ["Abundance", "K-mer", "Fusion"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(pipelines))
    width = 0.2
    for i, clf in enumerate(CLASSIFIERS):
        means, errs = [], []
        for p in pipelines:
            row = df[(df.pipeline == p) & (df.classifier == clf)]
            means.append(row.AUC_mean.values[0] if len(row) else np.nan)
            errs.append(row.AUC_std.values[0] if len(row) else 0)
        ax.bar(x + (i - 1.5) * width, means, width, yerr=errs, capsize=2,
               label=CLF_SHORT[clf], color=CLF_COLORS[clf])
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pipelines)
    ax.set_ylabel("AUC (mean ± SD, 4 matched held-out sites)")
    ax.set_title("IBD leave-one-site-out validation (matched 4-site comparison)")
    ax.set_ylim(0, 1)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
    savefig(fig, "FigS1_ibd_leave_one_site_out")


# ── Fig S2: Geography study-held-out (abundance, full cohort + k-mer pilot) ─
def fig_s2():
    ab = pd.read_csv(f"{EXP}/02_geography_study_confound/results/geography_study_holdout_summary.csv")
    ab["pipeline"] = "Abundance (n=22,397)"
    km = pd.read_csv(f"{EXP}/02_geography_study_confound/results/geography_kmer_pilot_study_holdout_summary.csv")
    km["pipeline"] = "K-mer pilot (n=1,453/1,587)"
    df = pd.concat([ab, km], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    x = np.arange(len(CLASSIFIERS))
    width = 0.32
    pipelines = [("Abundance (n=22,397)", "#4C72B0"), ("K-mer pilot (n=1,453/1,587)", "#C44E52")]
    for i, (pipe, color) in enumerate(pipelines):
        sub = df[df.pipeline == pipe].set_index("classifier").loc[CLASSIFIERS]
        axes[0].bar(x + (i - 0.5) * width, sub["AUC_mean"], width, yerr=sub["AUC_std"],
                    capsize=3, color=color, label=pipe)
        axes[1].bar(x + (i - 0.5) * width, sub["F1_macro_mean"], width, yerr=sub["F1_macro_std"],
                    capsize=3, color=color, label=pipe)
    axes[0].axhline(0.5, color="gray", ls="--", lw=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([CLF_SHORT[c] for c in CLASSIFIERS])
    axes[0].set_ylabel("AUC (macro OvR)")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Study-held-out AUC")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([CLF_SHORT[c] for c in CLASSIFIERS])
    axes[1].set_ylabel("F1-macro")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Study-held-out F1-macro")
    axes[1].legend(loc="upper center", bbox_to_anchor=(-0.15, -0.18), ncol=2, frameon=False, fontsize=7)
    fig.suptitle("Geography (30-country) study-held-out re-evaluation", y=1.03)
    savefig(fig, "FigS2_geography_study_held_out")


# ── Fig S3: Obesity batch-signal regime comparison (mixed pool + 6 within-study replicates) ──
def fig_s3():
    df = pd.read_csv(os.path.join(SUPP_TABLES_DIR, "Table_S19_Obesity_batch_signal_regime_comparison.csv"))
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    x = np.arange(len(CLASSIFIERS))
    n_regimes = 7
    width = 0.11
    regimes = [
        ("All studies (mixed pool)", "#4C72B0", "Mixed pool\n(all studies)"),
        ("MetaCardis_2020_a (within-study)", "#C44E52", "MetaCardis\n(within-study)"),
        ("ZeeviD_2015 (within-study)", "#DD8452", "ZeeviD_2015\n(within-study)"),
        ("NielsenHB_2014 (within-study)", "#55A868", "NielsenHB_2014\n(within-study)"),
        ("AsnicarF_2021 (within-study)", "#8172B2", "AsnicarF_2021\n(within-study)"),
        ("HMP_2012 (within-study)", "#937860", "HMP_2012\n(within-study)"),
        ("LeChatelierE_2013 (within-study)", "#64B5CD", "LeChatelierE_2013\n(within-study)"),
    ]
    offset0 = -(n_regimes - 1) / 2.0
    for i, (study, color, label) in enumerate(regimes):
        vals = df[df.study == study].set_index("classifier").loc[CLASSIFIERS, "AUC_gap_kmer_minus_abundance"]
        ax.bar(x + (offset0 + i) * width, vals.values, width, label=label, color=color)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([CLF_SHORT[c] for c in CLASSIFIERS])
    ax.set_ylabel("AUC gap (k-mer − abundance)")
    ax.set_title("Obesity: k-mer advantage, mixed pool vs. within-study (6 studies)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=4, frameon=False, fontsize=7)
    savefig(fig, "FigS3_obesity_batch_signal_reversal")


# ── Fig S4: matched-dimension check, all three tasks ────────────────────────
def fig_s4():
    files = {
        "IBD": f"{EXP}/04_matched_dimension_check/results/ibd_matched_dimension_gap.csv",
        "Obesity": f"{EXP}/04_matched_dimension_check/results/obesity_matched_dimension_gap.csv",
        "Geography": f"{EXP}/04_matched_dimension_check/results/geography_matched_dimension_gap.csv",
    }
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
    for ax, (task, path) in zip(axes, files.items()):
        df = pd.read_csv(path)
        for clf in CLASSIFIERS:
            sub = df[df.classifier == clf].sort_values("matched_dim")
            ax.plot(sub.matched_dim, sub.gap_kmer_minus_abundance, marker="o",
                    label=CLF_SHORT[clf], color=CLF_COLORS[clf])
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(250, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("Matched dimension")
        ax.set_title(task)
    axes[0].set_ylabel("AUC gap (k-mer − abundance)")
    axes[1].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    fig.suptitle("Matched-dimension sensitivity (same selection method, same dimension count, both modalities)", y=1.03)
    savefig(fig, "FigS4_ibd_matched_dimension")


# ── Fig S5: HMP2 read-depth sensitivity ──────────────────────────────────────
def fig_s5():
    df = pd.read_csv(f"{EXP}/05_hmp2_read_depth_sensitivity/results/ibd_depth_sensitivity_summary.csv")
    order = {"50000": 0, "100000": 1, "200000": 2, "full": 3}
    labels = ["50k", "100k", "200k\n(cMD3-matched)", "full\n(untruncated)"]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for clf in CLASSIFIERS:
        sub = df[df.classifier == clf].copy()
        sub["_o"] = sub["depth"].astype(str).map(order)
        sub = sub.sort_values("_o")
        ax.errorbar(sub._o, sub.AUC_mean, yerr=sub.AUC_std, marker="o", capsize=3,
                    label=CLF_SHORT[clf], color=CLF_COLORS[clf])
    ax.set_xticks(list(order.values()))
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.4, 0.9)
    ax.set_title("HMP2 k=8 read-depth sensitivity")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False)
    savefig(fig, "FigS5_hmp2_read_depth_sensitivity")


# ── Fig S7: IBD abundance-vs-k-mer complementarity (PCA) ─────────────────────
def fig_s7():
    comp_dir = f"{EXP}/07_complementarity_analysis/results"
    scores = pd.read_csv(f"{comp_dir}/ibd_complementarity_pca_scores.csv")
    corr = pd.read_csv(f"{comp_dir}/ibd_complementarity_pc_correlations.csv")
    var = pd.read_csv(f"{comp_dir}/ibd_pca_variance_explained.csv")

    label_colors = {0: "#4C72B0", 1: "#C44E52"}
    label_names = {0: "non-IBD", 1: "IBD"}

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    for ax, modality, title in [(axes[0], "abundance", "Abundance PCA"), (axes[1], "kmer", "K-mer PCA")]:
        for lab in [0, 1]:
            sub = scores[scores.label == lab]
            v = var[var.modality == modality].set_index("PC")["variance_explained"]
            ax.scatter(sub[f"{modality}_PC1"], sub[f"{modality}_PC2"], s=8, alpha=0.5,
                       color=label_colors[lab], label=label_names[lab])
        ax.set_xlabel(f"PC1 ({v.loc[1]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({v.loc[2]*100:.1f}%)")
        ax.set_title(title)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=8)

    pivot = corr.pivot(index="abundance_PC", columns="kmer_PC", values="pearson_r")
    im = axes[2].imshow(pivot.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    axes[2].set_xticks(range(5)); axes[2].set_xticklabels([f"km PC{i+1}" for i in range(5)], rotation=45, ha="right")
    axes[2].set_yticks(range(5)); axes[2].set_yticklabels([f"ab PC{i+1}" for i in range(5)])
    axes[2].set_title("Cross-modality PC correlation")
    for i in range(5):
        for j in range(5):
            axes[2].text(j, i, f"{pivot.values[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes[2], label="Pearson $r$", fraction=0.046, pad=0.04)

    fig.suptitle("IBD abundance vs. k-mer feature-space complementarity ($n=1{,}338$ matched samples)", y=1.03)
    savefig(fig, "FigS7_ibd_complementarity_pca")


# ── Fig S6: Extended k-mer length ablation ──────────────────────────────────
def fig_s6():
    df = pd.read_csv(f"{EXP}/06_kmer_k_ablation_extended/results/ibd_k_ablation_extended_summary.csv")
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for clf in CLASSIFIERS:
        sub = df[df.classifier == clf].sort_values("k")
        ax.errorbar(sub.k, sub.AUC_mean, yerr=sub.AUC_std, marker="o", capsize=3,
                    label=CLF_SHORT[clf], color=CLF_COLORS[clf])
    ax.axvline(8, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("k-mer length (k)")
    ax.set_ylabel("AUC (mean ± SD, 10 splits)")
    ax.set_title("Extended IBD k-mer length ablation (k = 3–10)")
    ax.set_xticks([3, 6, 7, 8, 9, 10])
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    savefig(fig, "FigS6_ibd_k_ablation_extended")


if __name__ == "__main__":
    for fn in (fig_s1, fig_s2, fig_s3, fig_s4, fig_s5, fig_s6):
        try:
            fn()
        except FileNotFoundError as e:
            print(f"Skipping {fn.__name__}: required results not found ({e}). "
                  f"Run the corresponding supplementary_experiments/<NN>_*/run_*.py "
                  f"script first (see supplementary_experiments/README.md).")
    print("\nSupplementary figures written to", OUT)
