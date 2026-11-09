"""
Step 3 – Unified comparison table and figure for obesity task.
Input:  results/results_abundance.csv  (mean ± SD across 5 rounds)
        results/results_kmer.csv       (mean ± SD across 5 rounds)
Output: results/summary_obesity.csv
        results/figures/summary_obesity_comparison.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_all_rounds():
    """Load raw per-round results for plotting with error bars."""
    dfs = []
    for fname, pipeline in [("abundance_all_rounds.csv", "Abundance (Line1)"),
                             ("kmer_all_rounds.csv",     "K-mer (Line2)")]:
        p = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["pipeline_label"] = pipeline
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def make_summary_table(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pipeline, clf), grp in all_df.groupby(["pipeline_label", "classifier"]):
        row = {"pipeline": pipeline, "classifier": clf}
        for m in ["AUC", "AUPRC", "F1_macro", "MCC"]:
            row[f"{m}_mean"] = round(grp[m].mean(), 4)
            row[f"{m}_std"]  = round(grp[m].std(), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["pipeline", "AUC_mean"], ascending=[True, False])


def plot_comparison(all_df: pd.DataFrame, summary: pd.DataFrame):
    metrics = ["AUC", "AUPRC", "F1_macro", "MCC"]
    pipelines  = all_df["pipeline_label"].unique()
    classifiers = sorted(all_df["classifier"].unique())
    colors = {"Abundance (Line1)": "steelblue", "K-mer (Line2)": "darkorange"}

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    x = np.arange(len(classifiers))
    w = 0.35

    for ax, m in zip(axes, metrics):
        for i, pipe in enumerate(pipelines):
            sub = summary[summary["pipeline"] == pipe].set_index("classifier")
            means = [sub.loc[c, f"{m}_mean"] if c in sub.index else 0 for c in classifiers]
            stds  = [sub.loc[c, f"{m}_std"]  if c in sub.index else 0 for c in classifiers]
            offset = (i - 0.5) * w
            ax.bar(x + offset, means, w, yerr=stds, capsize=4,
                   label=pipe, color=colors.get(pipe, "gray"), alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(classifiers, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1); ax.set_title(m, fontsize=11)
        ax.axhline(0.5, color="gray", ls="--", lw=0.8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.05), fontsize=9)
    fig.suptitle("Obesity Classification: Abundance vs K-mer\n"
                 "(10-round mean ± SD, n≈1600/round)", fontsize=11, y=1.02)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "summary_obesity_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison figure → {out}")


def main():
    all_df = load_all_rounds()
    if all_df.empty:
        print("No results found. Run 02a and/or 02b first.")
        return

    summary = make_summary_table(all_df)
    summary.to_csv(os.path.join(RESULTS_DIR, "summary_obesity.csv"), index=False)

    print("\n" + "=" * 70)
    print("OBESITY CLASSIFICATION SUMMARY (10-round mean ± SD, n≈1600/round)")
    print("=" * 70)
    for _, row in summary.iterrows():
        print(f"  {row['pipeline']:20s} {row['classifier']:15s}  "
              f"AUC={row['AUC_mean']:.3f}±{row['AUC_std']:.3f}  "
              f"AUPRC={row['AUPRC_mean']:.3f}±{row['AUPRC_std']:.3f}  "
              f"F1={row['F1_macro_mean']:.3f}±{row['F1_macro_std']:.3f}  "
              f"MCC={row['MCC_mean']:.3f}±{row['MCC_std']:.3f}")

    plot_comparison(all_df, summary)


if __name__ == "__main__":
    main()
