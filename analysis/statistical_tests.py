"""
Wilcoxon signed-rank test: K-mer (Line2) vs Abundance (Line1)
across 10 rounds for Obesity, Geography, and IBD tasks.

For each task × classifier × metric:
  H0: no difference between k-mer and abundance performance
  Test: Wilcoxon signed-rank (paired, n=10 rounds, non-parametric)
  Correction: Benjamini-Hochberg across all 48 task-classifier-metric comparisons

Output: statistical_tests.csv
        supplementary_tables/Table_S4_Wilcoxon_tests.csv
        figures/statistical_tests_heatmap.png
        figures/statistical_tests_heatmap.pdf
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import warnings

ROOT        = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(ROOT, "figures")
SUPPLEMENTARY_DIR = os.path.join(ROOT, "..", "supplementary_tables")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(SUPPLEMENTARY_DIR, exist_ok=True)

METRICS = ["AUC", "AUPRC", "F1_macro", "MCC"]
METRIC_LABELS = {"AUC": "AUC", "AUPRC": "AUPRC", "F1_macro": "F1-macro", "MCC": "MCC"}
CLASSIFIER_ORDER = ["LASSO-LR", "Random Forest", "SVM (RBF)", "XGBoost"]
TASK_ORDER = ["Ibd", "Obesity", "Geography"]
TASK_LABELS = {"Ibd": "IBD", "Obesity": "Obesity", "Geography": "Geography"}
DELTA_CMAP = LinearSegmentedColormap.from_list(
    "abundance_to_kmer",
    ["#0072b2", "#f7f7f7", "#e69f00"],
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


def load_rounds(task, pipeline_label):
    fname = "abundance" if "Line1" in pipeline_label else "kmer"
    path  = os.path.join(ROOT, "..", task, "results", f"{fname}_all_rounds.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["pipeline"] = pipeline_label
    return df


def load_ibd_rounds(pipeline_label):
    """Load IBD repeated-split CSVs from ibd/results/."""
    if "Line1" in pipeline_label:
        path = os.path.join(ROOT, "../ibd/results/ibd_abundance_repeated.csv")
    else:
        path = os.path.join(ROOT, "../ibd/results/ibd_kmer_repeated.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["pipeline"] = pipeline_label
    return df


def run_tests(task):
    if task == "ibd":
        abund = load_ibd_rounds("Abundance (Line1)")
        kmer  = load_ibd_rounds("K-mer (Line2)")
    else:
        abund = load_rounds(task, "Abundance (Line1)")
        kmer  = load_rounds(task, "K-mer (Line2)")
    if abund.empty or kmer.empty:
        return pd.DataFrame()

    classifiers = sorted(abund["classifier"].unique())
    rows = []
    for clf in classifiers:
        a_sub = abund[abund["classifier"] == clf][["round"] + METRICS]
        k_sub = kmer[kmer["classifier"] == clf][["round"] + METRICS]
        paired = a_sub.merge(k_sub, on="round", suffixes=("_a", "_k"))
        if len(paired) < 3:
            continue
        for m in METRICS:
            a_vals = paired[f"{m}_a"].values
            k_vals = paired[f"{m}_k"].values
            diff   = k_vals - a_vals
            delta  = diff.mean()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    # Two-sided Wilcoxon test (main analysis)
                    stat_2s, p_2s = wilcoxon(k_vals, a_vals, alternative="two-sided")
                    # One-sided Wilcoxon test: k-mer > abundance (supplementary analysis)
                    stat_1s, p_1s = wilcoxon(k_vals, a_vals, alternative="greater")
                except ValueError:
                    stat_2s, p_2s = float("nan"), 1.0
                    stat_1s, p_1s = float("nan"), 1.0
            rows.append({
                "task":       task.capitalize(),
                "classifier": clf,
                "metric":     m,
                "mean_kmer":  round(k_vals.mean(), 4),
                "mean_abund": round(a_vals.mean(), 4),
                "delta":      round(delta, 4),
                "W_stat":     round(stat_2s, 3) if not np.isnan(stat_2s) else float("nan"),
                "p_raw_2s":   round(p_2s, 4),
                "p_raw_1s":   round(p_1s, 4),
            })
    return pd.DataFrame(rows)


def plot_heatmap(results):
    rows = []
    matrix = []
    pmat = []
    for task in TASK_ORDER:
        task_df = results[results["task"] == task]
        if task_df.empty:
            continue
        for clf in CLASSIFIER_ORDER:
            clf_df = task_df[task_df["classifier"] == clf].set_index("metric")
            if clf_df.empty:
                continue
            rows.append(f"{TASK_LABELS.get(task, task)} | {clf.replace('Random Forest', 'RF').replace('SVM (RBF)', 'SVM').replace('XGBoost', 'XGB')}")
            matrix.append([float(clf_df.loc[m, "delta"]) if m in clf_df.index else np.nan for m in METRICS])
            pmat.append([float(clf_df.loc[m, "p_adj_2s"]) if m in clf_df.index else np.nan for m in METRICS])

    matrix = np.asarray(matrix)
    pmat = np.asarray(pmat)
    vmax = np.nanmax(np.abs(matrix))
    norm = TwoSlopeNorm(vmin=-max(vmax, 0.02), vcenter=0, vmax=max(vmax, 0.02))

    fig, ax = plt.subplots(figsize=(6.7, 5.4))
    im = ax.imshow(matrix, cmap=DELTA_CMAP, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.tick_params(length=0)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            p = pmat[i, j]
            if np.isnan(val):
                continue
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            text_color = "white" if abs(val) > vmax * 0.58 else "black"
            ax.text(j, i, f"{val:+.2f}{star}", ha="center", va="center",
                    fontsize=7, color=text_color)

    # Light separators between tasks.
    for sep in [3.5, 7.5]:
        if sep < len(rows) - 0.5:
            ax.axhline(sep, color="white", lw=2.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Δ performance (k-mer − abundance)")
    fig.text(0.01, 0.01, "* p_adj<0.05, ** p_adj<0.01, *** p_adj<0.001; BH correction across 48 comparisons",
             fontsize=7)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    out = os.path.join(FIGURES_DIR, "statistical_tests_heatmap.png")
    out_pdf = os.path.join(FIGURES_DIR, "statistical_tests_heatmap.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Heatmap → {out}")
    print(f"Heatmap → {out_pdf}")


def main():
    all_rows = []
    for task in ["ibd", "obesity", "geography"]:
        rows = run_tests(task)
        if not rows.empty:
            all_rows.append(rows)
            print(f"{task.capitalize()}: {len(rows)} comparisons")

    if not all_rows:
        print("No round-level results found.")
        return

    results = pd.concat(all_rows, ignore_index=True)

    # BH correction across all task-classifier-metric comparisons.
    # Main analysis: two-sided
    _, p_adj_2s, _, _ = multipletests(results["p_raw_2s"].fillna(1), method="fdr_bh")
    results["p_adj_2s"] = p_adj_2s.round(4)

    # Supplementary analysis: one-sided (k-mer > abundance)
    _, p_adj_1s, _, _ = multipletests(results["p_raw_1s"].fillna(1), method="fdr_bh")
    results["p_adj_1s"] = p_adj_1s.round(4)

    results["significant_2s"] = results["p_adj_2s"] < 0.05
    results["significant_1s"] = results["p_adj_1s"] < 0.05

    out = os.path.join(ROOT, "statistical_tests.csv")
    results.to_csv(out, index=False)
    print(f"\nSaved → {out}")

    supp_out = os.path.join(SUPPLEMENTARY_DIR, "Table_S4_Wilcoxon_tests.csv")
    results.to_csv(supp_out, index=False)
    print(f"Saved → {supp_out}")

    print("\n" + "=" * 85)
    print("WILCOXON TEST: K-mer vs Abundance")
    print("=" * 85)
    print("\n[MAIN ANALYSIS] Two-sided test (BH correction across 48 comparisons):")
    print("-" * 85)
    for _, row in results.sort_values(["task","metric","p_adj_2s"]).iterrows():
        sig = "*" if row["p_adj_2s"] < 0.05 else " "
        print(f"  {sig} {row['task']:12s} {row['classifier']:15s} {row['metric']:8s}  "
              f"Δ={row['delta']:+.3f}  p_adj={row['p_adj_2s']:.3f}")

    print("\n[SUPPLEMENTARY ANALYSIS] One-sided test: k-mer > abundance (BH correction):")
    print("-" * 85)
    sig_1s = results[results["significant_1s"]].sort_values("p_adj_1s")
    if len(sig_1s) > 0:
        for _, row in sig_1s.iterrows():
            print(f"  * {row['task']:12s} {row['classifier']:15s} {row['metric']:8s}  "
                  f"Δ={row['delta']:+.3f}  p_adj={row['p_adj_1s']:.3f}")
    else:
        print("  (No significant findings at p_adj < 0.05)")

    plot_heatmap(results)


if __name__ == "__main__":
    main()
