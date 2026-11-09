"""
Global summary: IBD + Obesity + Geography across all three pipelines.
Reads all existing result CSVs and produces a unified comparison table + figure.

Sources:
  IBD abundance  – ibd/results/ibd_abundance_repeated.csv   (10-round repeated split)
  IBD k-mer      – ibd/results/ibd_kmer_repeated.csv        (10-round repeated split)
  Obesity        – obesity/results/{abundance,kmer}_all_rounds.csv
  Geography      – geography/results/{abundance,kmer}_all_rounds.csv
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

ROOT       = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Load task results ─────────────────────────────────────────────────

def load_task_results(task, pipeline_label):
    fname = "abundance" if "Line1" in pipeline_label else "kmer"
    path  = os.path.join(ROOT, "..", task, "results", f"{fname}_all_rounds.csv")
    if not os.path.exists(path):
        print(f"  Missing: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["task"]     = task.capitalize()
    df["pipeline"] = pipeline_label
    return df


def summarise(df):
    rows = []
    for (task, pipe, clf), grp in df.groupby(["task", "pipeline", "classifier"]):
        row = {"task": task, "pipeline": pipe, "classifier": clf}
        for m in ["AUC", "AUPRC", "F1_macro", "MCC"]:
            row[f"{m}_mean"] = round(grp[m].mean(), 4)
            row[f"{m}_std"]  = round(grp[m].std(),  4)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Load IBD results from 10-round repeated-split CSVs ───────────────

def load_ibd_repeated(pipeline_label):
    if "Line1" in pipeline_label:
        path = os.path.join(ROOT, "../ibd/results/ibd_abundance_repeated.csv")
    else:
        path = os.path.join(ROOT, "../ibd/results/ibd_kmer_repeated.csv")
    if not os.path.exists(path):
        print(f"  Missing: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["task"]     = "IBD"
    df["pipeline"] = pipeline_label
    return summarise(df)


# ── Plot ──────────────────────────────────────────────────────────────

def plot_global(summary):
    tasks      = ["IBD", "Obesity", "Geography"]
    pipelines  = ["Abundance (Line1)", "K-mer (Line2)"]
    colors     = {"Abundance (Line1)": "#0072b2", "K-mer (Line2)": "#e69f00"}
    classifiers = ["LASSO-LR", "Random Forest", "SVM (RBF)", "XGBoost"]
    clf_short = ["LASSO", "RF", "SVM", "XGB"]

    fig, axes = plt.subplots(1, len(tasks), figsize=(10.2, 3.3), sharey=True)

    for ax, task in zip(axes, tasks):
        task_df = summary[summary["task"] == task]
        x  = np.arange(len(classifiers))
        w  = 0.34
        for pi, pipe in enumerate(pipelines):
            sub = task_df[task_df["pipeline"] == pipe].set_index("classifier")
            means = [sub.loc[c, "AUC_mean"] if c in sub.index else np.nan for c in classifiers]
            stds  = [sub.loc[c, "AUC_std"] if c in sub.index else 0 for c in classifiers]
            ax.bar(x + (pi - 0.5) * w, means, w, yerr=stds, capsize=2.5,
                   label=pipe, color=colors[pipe], alpha=0.88, edgecolor="white", lw=0.4)
        ax.set_title(task)
        ax.set_xticks(x)
        ax.set_xticklabels(clf_short)
        ax.set_ylim(0.45, 1.00)
        ax.axhline(0.5, color="gray", ls="--", lw=0.7, alpha=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("AUC (mean ± SD)")

    handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.85) for c in colors.values()]
    fig.legend(handles, ["Abundance", "K-mer"], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.04), frameon=False)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "global_summary.png")
    out_pdf = os.path.join(FIGURES_DIR, "global_summary.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out}")
    print(f"Figure → {out_pdf}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parts = []

    # IBD abundance + k-mer (10-round repeated split)
    for pipe in ["Abundance (Line1)", "K-mer (Line2)"]:
        df = load_ibd_repeated(pipe)
        if not df.empty:
            parts.append(df)
            print(f"IBD {pipe}: {len(df)} classifiers loaded (10-round repeated)")

    # Obesity + Geography (10-round repeated split)
    for task in ["obesity", "geography"]:
        for pipe in ["Abundance (Line1)", "K-mer (Line2)"]:
            df = load_task_results(task, pipe)
            if not df.empty:
                parts.append(summarise(df))

    if not parts:
        print("No results found.")
        return

    summary = pd.concat(parts, ignore_index=True)

    # Re-summarise non-IBD parts (already per-round rows); summarise() collapses them.
    # IBD parts are already summarised by load_ibd_repeated().

    out_csv = os.path.join(ROOT, "global_summary.csv")
    summary.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    print("\n" + "=" * 80)
    print("GLOBAL SUMMARY")
    print("=" * 80)
    for _, row in summary.sort_values(["task", "pipeline", "AUC_mean"],
                                       ascending=[True, True, False]).iterrows():
        print(f"  {row['task']:12s} {row['pipeline']:20s} {row['classifier']:15s}  "
              f"AUC={row['AUC_mean']:.3f}±{row['AUC_std']:.3f}  "
              f"F1={row['F1_macro_mean']:.3f}±{row['F1_macro_std']:.3f}  "
              f"MCC={row['MCC_mean']:.3f}±{row['MCC_std']:.3f}")

    plot_global(summary)


if __name__ == "__main__":
    main()
