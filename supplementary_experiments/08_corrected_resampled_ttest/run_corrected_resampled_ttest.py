#!/usr/bin/env python3
"""
Dependence-aware sensitivity analysis for the 48 primary k-mer vs. abundance
comparisons (Table S32).

The primary analysis (analysis/statistical_tests.py, Table S4) uses paired
two-sided Wilcoxon signed-rank tests across the ten repeated rounds. Because
the rounds are repeated 80/20 partitions of a fixed sample pool, their
training sets overlap and the round-level differences are positively
correlated; an ordinary paired test therefore underestimates the variance of
the mean difference. This script re-tests the same 48 paired differences with
the Nadeau-Bengio corrected resampled t-test, which inflates the variance
term to account for training-set overlap:

    t = mean(d) / sqrt( (1/J + n_test/n_train) * var(d) ),   df = J - 1

with J = 10 rounds and n_test/n_train = 0.2/0.8 = 0.25 for all three tasks.
p-values are Benjamini-Hochberg corrected across the same 48 comparisons.

Inputs (shipped):
  data/matched_data/{ibd,obesity,geography}_{abundance,kmer}_matched_repeated.csv
  data/supplementary_tables/Table_S4_Wilcoxon_tests.csv

Output:
  results/Table_S32_corrected_resampled_ttest.csv
"""
import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MATCHED = os.path.join(REPO, "data", "matched_data")
S4 = os.path.join(REPO, "data", "supplementary_tables", "Table_S4_Wilcoxon_tests.csv")
OUT_DIR = os.path.join(HERE, "results")

TASKS = {"ibd": "IBD", "obesity": "Obesity", "geography": "Geographic origin"}
METRICS = ["AUC", "AUPRC", "F1_macro", "MCC"]
CLASSIFIERS = ["LASSO-LR", "Random Forest", "SVM (RBF)", "XGBoost"]
TEST_TRAIN_RATIO = 0.2 / 0.8


def corrected_ttest(d, ratio=TEST_TRAIN_RATIO):
    d = np.asarray(d, dtype=float)
    j = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return np.nan, 1.0
    t = d.mean() / np.sqrt((1.0 / j + ratio) * var)
    p = 2 * stats.t.sf(abs(t), df=j - 1)
    return t, p


def main():
    rows = []
    for key, label in TASKS.items():
        ab = pd.read_csv(os.path.join(MATCHED, f"{key}_abundance_matched_repeated.csv"))
        km = pd.read_csv(os.path.join(MATCHED, f"{key}_kmer_matched_repeated.csv"))
        for clf in CLASSIFIERS:
            a = ab[ab.classifier == clf].set_index("round").sort_index()
            k = km[km.classifier == clf].set_index("round").sort_index()
            assert list(a.index) == list(k.index) and len(a) == 10
            for m in METRICS:
                d = k[m].values - a[m].values
                t, p = corrected_ttest(d)
                ci_half = stats.t.ppf(0.975, 9) * np.sqrt((1 / 10 + TEST_TRAIN_RATIO) * d.var(ddof=1))
                rows.append({
                    "task": label, "classifier": clf, "metric": m,
                    "mean_delta_kmer_minus_abund": round(d.mean(), 4),
                    "corrected_CI95_low": round(d.mean() - ci_half, 4),
                    "corrected_CI95_high": round(d.mean() + ci_half, 4),
                    "rounds_kmer_better": int((d > 0).sum()),
                    "t_corrected": round(t, 3), "p_raw_corrected_t": p,
                })
    df = pd.DataFrame(rows)
    df["p_adj_corrected_t"] = multipletests(df["p_raw_corrected_t"], method="fdr_bh")[1]
    df["significant_corrected_t"] = df["p_adj_corrected_t"] < 0.05

    # Attach the primary Wilcoxon result for side-by-side comparison.
    s4 = pd.read_csv(S4)
    s4["task"] = s4["task"].replace({"Ibd": "IBD", "Geography": "Geographic origin"})
    s4["classifier"] = s4["classifier"].replace({"SVM-RBF": "SVM (RBF)"})
    s4["metric"] = s4["metric"].replace({"F1": "F1_macro", "Macro-F1": "F1_macro"})
    df = df.merge(
        s4[["task", "classifier", "metric", "p_adj_2s", "significant_2s"]].rename(
            columns={"p_adj_2s": "p_adj_wilcoxon", "significant_2s": "significant_wilcoxon"}),
        on=["task", "classifier", "metric"], how="left")

    os.makedirs(OUT_DIR, exist_ok=True)
    df["p_raw_corrected_t"] = df["p_raw_corrected_t"].round(6)
    df["p_adj_corrected_t"] = df["p_adj_corrected_t"].round(4)
    out = os.path.join(OUT_DIR, "Table_S32_corrected_resampled_ttest.csv")
    df.to_csv(out, index=False)

    print(f"Wrote {out}")
    for label in TASKS.values():
        sub = df[df.task == label]
        sig = sub[sub.significant_corrected_t]
        print(f"{label}: corrected-t significant {len(sig)}/16 "
              f"(k-mer {int((sig.mean_delta_kmer_minus_abund > 0).sum())}, "
              f"abundance {int((sig.mean_delta_kmer_minus_abund < 0).sum())}); "
              f"Wilcoxon significant {int(sub.significant_wilcoxon.sum())}/16")
    sig = df[df.significant_corrected_t]
    print(f"Total corrected-t significant: {len(sig)}/48 "
          f"(k-mer {int((sig.mean_delta_kmer_minus_abund > 0).sum())}, "
          f"abundance {int((sig.mean_delta_kmer_minus_abund < 0).sum())}); "
          f"Wilcoxon {int(df.significant_wilcoxon.sum())}/48")
    both = (df.significant_corrected_t & df.significant_wilcoxon).sum()
    print(f"Significant under both: {both}")


if __name__ == "__main__":
    main()
