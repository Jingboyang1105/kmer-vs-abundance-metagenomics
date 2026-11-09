"""
Fusion pairwise comparison within same classifier.
Guards against best-classifier selection bias by testing:
  For each classifier, is Fusion significantly better than both Abundance and K-mer?

Output: results/fusion_pairwise_comparison.csv
        supplementary_tables/Table_S8_Fusion_pairwise_same_classifier.csv
        figures/fusion_pairwise_heatmap.png
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings

ROOT        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
SUPPLEMENTARY_DIR = os.path.join(ROOT, "..", "supplementary_tables")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(SUPPLEMENTARY_DIR, exist_ok=True)

METRICS = ["AUC", "AUPRC", "F1_macro", "MCC"]
METRIC_LABELS = {"AUC": "AUC", "AUPRC": "AUPRC", "F1_macro": "F1-macro", "MCC": "MCC"}
CLASSIFIERS = ["LASSO-LR", "Random Forest", "SVM (RBF)", "XGBoost"]
CLASSIFIER_SHORT = {"LASSO-LR": "LASSO", "Random Forest": "RF", "SVM (RBF)": "SVM", "XGBoost": "XGB"}
TASKS = ["IBD", "Obesity", "Geography"]

CMAP_PVAL = LinearSegmentedColormap.from_list(
    "pval_cmap",
    ["#e69f00", "#f7f7f7", "#0072b2"],  # significant -> not significant
    N=256,
)

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 7,
})


def load_task_results(task):
    """Load abundance, k-mer, and fusion results for one task."""
    if task == "IBD":
        abund_path  = os.path.join(ROOT, "../ibd/results/ibd_abundance_repeated.csv")
        kmer_path   = os.path.join(ROOT, "../ibd/results/ibd_kmer_repeated.csv")
        fusion_path = os.path.join(ROOT, "../ibd/results/fusion_ibd_all_rounds.csv")
    else:
        task_lower  = task.lower()
        abund_path  = os.path.join(ROOT, f"../{task_lower}/results/abundance_all_rounds.csv")
        kmer_path   = os.path.join(ROOT, f"../{task_lower}/results/kmer_all_rounds.csv")
        if task == "Geography":
            fusion_path = os.path.join(ROOT, f"../{task_lower}/results/fusion_geo_all_rounds.csv")
        else:
            fusion_path = os.path.join(ROOT, f"../{task_lower}/results/fusion_all_rounds.csv")

    dfs = {}
    for name, path in [("Abundance", abund_path), ("K-mer", kmer_path), ("Fusion", fusion_path)]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs[name] = df
        else:
            print(f"  Warning: Missing {name} for {task} at {path}")
            dfs[name] = pd.DataFrame()

    return dfs


def run_pairwise_tests(task):
    """
    For each classifier, test:
    - Fusion vs Abundance (same classifier)
    - Fusion vs K-mer (same classifier)
    """
    dfs = load_task_results(task)

    if len(dfs) < 3 or any(df.empty for df in dfs.values()):
        print(f"  Skipping {task}: incomplete data")
        return pd.DataFrame()

    rows = []
    for clf in CLASSIFIERS:
        abund_df  = dfs["Abundance"]
        kmer_df   = dfs["K-mer"]
        fusion_df = dfs["Fusion"]

        a_clf = abund_df[abund_df["classifier"] == clf]
        k_clf = kmer_df[kmer_df["classifier"] == clf]
        f_clf = fusion_df[fusion_df["classifier"] == clf]

        if a_clf.empty or k_clf.empty or f_clf.empty:
            continue

        for metric in METRICS:
            paired = (
                f_clf[["round", metric]].rename(columns={metric: "f"})
                .merge(a_clf[["round", metric]].rename(columns={metric: "a"}), on="round")
                .merge(k_clf[["round", metric]].rename(columns={metric: "k"}), on="round")
            )
            if len(paired) < 3:
                continue

            f_vals = paired["f"].values
            a_vals = paired["a"].values
            k_vals = paired["k"].values

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    stat_fa, p_fa = wilcoxon(f_vals, a_vals, alternative="greater")
                    delta_fa = (f_vals - a_vals).mean()
                except (ValueError, RuntimeError):
                    stat_fa, p_fa, delta_fa = float("nan"), 1.0, float("nan")

                try:
                    stat_fk, p_fk = wilcoxon(f_vals, k_vals, alternative="greater")
                    delta_fk = (f_vals - k_vals).mean()
                except (ValueError, RuntimeError):
                    stat_fk, p_fk, delta_fk = float("nan"), 1.0, float("nan")

            rows.append({
                "task":        task,
                "classifier":  clf,
                "metric":      metric,
                "fusion_mean": round(f_vals.mean(), 4),
                "abund_mean":  round(a_vals.mean(), 4),
                "kmer_mean":   round(k_vals.mean(), 4),
                "delta_fa":    round(delta_fa, 4),
                "delta_fk":    round(delta_fk, 4),
                "p_raw_fa":    round(p_fa, 4),
                "p_raw_fk":    round(p_fk, 4),
            })

    return pd.DataFrame(rows)


def plot_pairwise_heatmap(results):
    """Plot two heatmaps: Fusion vs Abundance, Fusion vs K-mer (p-values)."""
    comparisons = [("Fusion vs Abundance", "p_adj_fa"), ("Fusion vs K-mer", "p_adj_fk")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, (title, p_col) in zip(axes, comparisons):
        rows_list = []
        matrix = []

        for task in TASKS:
            task_df = results[results["task"] == task]
            if task_df.empty:
                continue
            for clf in CLASSIFIERS:
                clf_df = task_df[task_df["classifier"] == clf].set_index("metric")
                if clf_df.empty:
                    continue
                rows_list.append(f"{task} | {CLASSIFIER_SHORT[clf]}")
                row_data = [float(clf_df.loc[m, p_col]) if m in clf_df.index else float("nan")
                            for m in METRICS]
                matrix.append(row_data)

        if not matrix:
            continue

        matrix = np.asarray(matrix)
        im = ax.imshow(matrix, cmap=CMAP_PVAL, vmin=0.0, vmax=1.0, aspect="auto")

        ax.set_xticks(np.arange(len(METRICS)))
        ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
        ax.set_yticks(np.arange(len(rows_list)))
        ax.set_yticklabels(rows_list)
        ax.tick_params(length=0)
        ax.set_title(title, loc="left", pad=10)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                p = matrix[i, j]
                if np.isnan(p):
                    continue
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                text_color = "white" if p < 0.5 else "black"
                ax.text(j, i, f"{p:.2f}{star}", ha="center", va="center",
                        fontsize=7, color=text_color, weight="bold" if star else "normal")

        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).set_label("p-value")

    fig.suptitle("Fusion within-classifier pairwise comparison (one-sided: Fusion > single modality)",
                 fontsize=10, y=1.02)
    fig.text(0.5, -0.02,
             "* p_adj<0.05, ** p_adj<0.01, *** p_adj<0.001; BH correction within each comparison",
             ha="center", fontsize=7)
    plt.tight_layout()

    out = os.path.join(FIGURES_DIR, "fusion_pairwise_heatmap.png")
    out_pdf = os.path.join(FIGURES_DIR, "fusion_pairwise_heatmap.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Heatmap → {out}")
    print(f"Heatmap → {out_pdf}")


def main():
    all_rows = []
    for task in TASKS:
        rows = run_pairwise_tests(task)
        if not rows.empty:
            all_rows.append(rows)

    if not all_rows:
        print("No results found.")
        return

    results = pd.concat(all_rows, ignore_index=True)

    _, p_adj_fa, _, _ = multipletests(results["p_raw_fa"].fillna(1), method="fdr_bh")
    results["p_adj_fa"] = p_adj_fa.round(4)

    _, p_adj_fk, _, _ = multipletests(results["p_raw_fk"].fillna(1), method="fdr_bh")
    results["p_adj_fk"] = p_adj_fk.round(4)

    results["sig_fa"] = results["p_adj_fa"] < 0.05
    results["sig_fk"] = results["p_adj_fk"] < 0.05

    out_csv = os.path.join(RESULTS_DIR, "fusion_pairwise_comparison.csv")
    results.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    supp_csv = os.path.join(SUPPLEMENTARY_DIR, "Table_S8_Fusion_pairwise_same_classifier.csv")
    results.to_csv(supp_csv, index=False)
    print(f"Saved → {supp_csv}\n")

    print("=" * 100)
    print("FUSION WITHIN-CLASSIFIER PAIRWISE COMPARISON (one-sided: Fusion > single modality)")
    print("=" * 100)

    print("\n[FUSION vs ABUNDANCE]")
    print("-" * 100)
    for _, row in results.sort_values(["task", "classifier", "p_adj_fa"]).iterrows():
        sig = "*" if row["sig_fa"] else " "
        print(f"  {sig} {row['task']:12s} {row['classifier']:15s} {row['metric']:8s}  "
              f"Fusion={row['fusion_mean']:.3f} Abund={row['abund_mean']:.3f}  "
              f"Δ={row['delta_fa']:+.3f}  p_adj={row['p_adj_fa']:.4f}")

    print("\n[FUSION vs K-MER]")
    print("-" * 100)
    for _, row in results.sort_values(["task", "classifier", "p_adj_fk"]).iterrows():
        sig = "*" if row["sig_fk"] else " "
        print(f"  {sig} {row['task']:12s} {row['classifier']:15s} {row['metric']:8s}  "
              f"Fusion={row['fusion_mean']:.3f} Kmer={row['kmer_mean']:.3f}  "
              f"Δ={row['delta_fk']:+.3f}  p_adj={row['p_adj_fk']:.4f}")

    print("\n" + "=" * 100)
    print("SUMMARY: Classifier performance against best single modality")
    print("=" * 100)
    for clf in CLASSIFIERS:
        clf_df = results[results["classifier"] == clf]
        if clf_df.empty:
            continue
        total  = len(clf_df)
        sig_fa = clf_df["sig_fa"].sum()
        sig_fk = clf_df["sig_fk"].sum()
        both   = (clf_df["sig_fa"] & clf_df["sig_fk"]).sum()
        print(f"\n{clf}:")
        print(f"  Significantly beats Abundance: {sig_fa}/{total} (p_adj < 0.05)")
        print(f"  Significantly beats K-mer:     {sig_fk}/{total} (p_adj < 0.05)")
        print(f"  Beats BOTH:                    {both}/{total}")

    plot_pairwise_heatmap(results)


if __name__ == "__main__":
    main()
