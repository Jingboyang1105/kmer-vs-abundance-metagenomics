"""
Supplementary Figure S4: K-mer design ablation (two focused questions)
  Panel A: Does k-mer length matter?       → frequency, k=3/6/8
  Panel B: Does frequency information matter? → frequency vs presence-absence (k=8)

One-hot position encoding removed: it is a heterogeneous representation
(frequency-weighted nucleotide composition of pre-selected discriminative k-mers)
that cannot be fairly compared to raw frequency encoding and does not constitute
a meaningful ablation of the k-mer frequency encoding.

Output: figures/ablation_kmer_design.png
        figures/ablation_kmer_design.pdf
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

ROOT        = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

CLASSIFIERS = ["LASSO-LR", "Random Forest", "SVM (RBF)", "XGBoost"]
CLF_SHORT   = ["LASSO", "RF", "SVM", "XGB"]


# ─── Pre-computed: presence-absence ─────────────────────────────────────────
pa_csv = pd.read_csv(os.path.join(ROOT, "../ibd/presence_absence/results/results_presence_absence.csv"))
pa_results = {}
for k in [3, 6, 8]:
    sub = pa_csv[pa_csv["k"] == k].set_index("Classifier")["AUC"]
    pa_results[k] = {clf: float(sub.get(clf, np.nan)) for clf in CLASSIFIERS}

# Frequency results from ibd/kmer_pipeline.py, k=3/6/8, top-n 1000.
freq_results = {
    3: {
        "LASSO-LR":      0.712,
        "Random Forest": 0.635,
        "SVM (RBF)":     0.667,
        "XGBoost":       0.628,
    },
    6: {
        "LASSO-LR":      0.746,
        "Random Forest": 0.788,
        "SVM (RBF)":     0.739,
        "XGBoost":       0.791,
    },
    8: {
        "LASSO-LR":      0.753,
        "Random Forest": 0.838,
        "SVM (RBF)":     0.793,
        "XGBoost":       0.776,
    },
}


# ─── Plotting ────────────────────────────────────────────────────────────────

K_VALS   = [3, 6, 8]
# Keep the paper-wide visual language:
#   Abundance = blue, K-mer = orange, Fusion = green.
# This ablation is entirely within the k-mer pipeline, so k-values use
# a light-to-dark orange ramp; presence-absence is shown in neutral grey.
K_COLORS = ["#f6c85f", "#f0a202", "#d97706"]
FREQ_COLOR = "#e69f00"
PA_COLOR = "#9e9e9e"

fig, (ax_A, ax_B) = plt.subplots(1, 2, figsize=(9.6, 3.55))
x = np.arange(len(CLASSIFIERS))

# ── Panel A: k-value comparison (frequency encoding) ──────────────────────────
width_A = 0.22
for ki, (k, kcol) in enumerate(zip(K_VALS, K_COLORS)):
    vals = [freq_results[k][clf] for clf in CLASSIFIERS]
    bars = ax_A.bar(x + (ki - 1) * width_A, vals, width_A, label=fr"$k={k}$",
                    color=kcol, alpha=0.88, edgecolor="white", lw=0.5)
    for bar, v in zip(bars, vals):
        ax_A.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                  f"{v:.2f}", ha="center", va="bottom", fontsize=6.4)

ax_A.set_xticks(x)
ax_A.set_xticklabels(CLF_SHORT)
ax_A.set_ylabel("AUC")
ax_A.set_ylim(0.45, 0.90)
ax_A.axhline(0.5, color="gray", ls="--", lw=0.9, alpha=0.5, label="chance (0.5)")
ax_A.set_title("A", loc="left", fontweight="bold")
ax_A.legend(loc="upper center", frameon=False, ncol=4, handlelength=1.2,
            columnspacing=0.9, bbox_to_anchor=(0.5, -0.15), borderaxespad=0.0)
ax_A.tick_params(axis="y")
ax_A.spines["top"].set_visible(False)
ax_A.spines["right"].set_visible(False)

# ── Panel B: frequency vs presence-absence (k=8) ──────────────────────────────
width_B = 0.32
enc_pairs = [
    ("Frequency",        freq_results[8], FREQ_COLOR),
    ("Presence-absence", pa_results[8],   PA_COLOR),
]
for ei, (label, data, color) in enumerate(enc_pairs):
    vals = [data[clf] for clf in CLASSIFIERS]
    bars = ax_B.bar(x + (ei - 0.5) * width_B, vals, width_B,
                    label=label, color=color, alpha=0.88, edgecolor="white", lw=0.5)
    for bar, v in zip(bars, vals):
        ax_B.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                  f"{v:.2f}", ha="center", va="bottom", fontsize=6.4)

ax_B.set_xticks(x)
ax_B.set_xticklabels(CLF_SHORT)
ax_B.set_ylabel("AUC")
ax_B.set_ylim(0.45, 0.90)
ax_B.axhline(0.5, color="gray", ls="--", lw=0.9, alpha=0.5, label="chance (0.5)")
ax_B.set_title("B", loc="left", fontweight="bold")
ax_B.legend(loc="upper center", frameon=False, ncol=3, handlelength=1.2,
            bbox_to_anchor=(0.5, -0.15), borderaxespad=0.0)
ax_B.tick_params(axis="y")
ax_B.spines["top"].set_visible(False)
ax_B.spines["right"].set_visible(False)

plt.tight_layout(rect=(0, 0.08, 1, 1))
out = os.path.join(FIGURES_DIR, "ablation_kmer_design.png")
out_pdf = os.path.join(FIGURES_DIR, "ablation_kmer_design.pdf")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"\nSaved → {out}")
print(f"Saved → {out_pdf}")

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"Panel A — Frequency encoding by k")
print(f"{'k':>4}  {'LASSO':>7} {'RF':>7} {'SVM':>7} {'XGB':>7}  {'Best':>7}")
for k in K_VALS:
    vals = [freq_results[k][c] for c in CLASSIFIERS]
    print(f"k={k}   " + "  ".join(f"{v:>7.3f}" for v in vals) + f"  {max(vals):>7.3f}")

print(f"\nPanel B — Encoding type at k=8")
for label, data, _ in enc_pairs:
    vals = [data[c] for c in CLASSIFIERS]
    name = label.split("\n")[0]
    print(f"{name:<22}" + "  ".join(f"{v:>7.3f}" for v in vals) + f"  {max(vals):>7.3f}")
