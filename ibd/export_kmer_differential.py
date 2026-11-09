"""
Export the full IBD differential k-mer table for Supplementary Table S7.

This script mirrors the full-sample IBD k=8 differential screen used for
post-hoc interpretation, but writes every canonical k-mer feature rather than
only the top attribution rows.

Outputs:
  supplementary_tables/Table_S7_IBD_full_kmer_differential.csv
  supplementary_tables/Table_S7_IBD_kmer_differential_summary.csv
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = os.path.dirname(os.path.abspath(__file__))
KMER_DIR = os.path.join(ROOT, "kmer_counts")
OUT_DIR = os.path.join(ROOT, "..", "supplementary_tables")
os.makedirs(OUT_DIR, exist_ok=True)

K = 8
DECODE = {0: "A", 1: "C", 2: "G", 3: "T"}


def decode_kmer(idx, k=K):
    bases = []
    idx = int(idx)
    for _ in range(k):
        bases.append(DECODE[idx & 3])
        idx >>= 2
    return "".join(reversed(bases))


def revcomp(seq):
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(comp[b] for b in reversed(seq))


def canonical_seq(raw_idx):
    seq = decode_kmer(raw_idx)
    rc = revcomp(seq)
    return seq if seq <= rc else rc


def main():
    x_counts = np.load(os.path.join(KMER_DIR, "kmer_matrix_k8_X.npy"))
    kmer_index = np.load(os.path.join(KMER_DIR, "kmer_matrix_k8_kmer_index.npy"))
    sample_ids = np.load(
        os.path.join(KMER_DIR, "kmer_matrix_k8_sample_ids.npy"),
        allow_pickle=True,
    )
    labels = np.load(
        os.path.join(KMER_DIR, "kmer_matrix_k8_labels.npy"),
        allow_pickle=True,
    )

    row_sums = x_counts.sum(axis=1, keepdims=True)
    x_freq = x_counts / np.where(row_sums == 0, 1, row_sums)

    _, uniq_idx = np.unique(sample_ids, return_index=True)
    x_freq = x_freq[uniq_idx]
    sample_ids = sample_ids[uniq_idx]
    labels = labels[uniq_idx]
    y = (labels == "IBD").astype(int)

    print(
        f"Loaded deduplicated k-mer matrix: {x_freq.shape}; "
        f"IBD={int(y.sum())}, non-IBD={int((y == 0).sum())}"
    )

    x_ibd = x_freq[y == 1]
    x_nonibd = x_freq[y == 0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_stat, p_value = stats.ttest_ind(x_ibd, x_nonibd, equal_var=False, axis=0)

    t_stat = np.nan_to_num(t_stat, nan=0.0)
    p_value = np.nan_to_num(p_value, nan=1.0)
    _, p_adj, _, _ = multipletests(p_value, method="fdr_bh")

    mean_ibd = x_ibd.mean(axis=0)
    mean_nonibd = x_nonibd.mean(axis=0)
    delta = mean_ibd - mean_nonibd

    direction = np.where(
        delta > 0, "IBD-enriched",
        np.where(delta < 0, "IBD-depleted", "no-change"),
    )
    significant = p_adj < 0.05
    rank_by_padj = pd.Series(p_adj).rank(method="first").astype(int).to_numpy()

    out = pd.DataFrame({
        "feature_column":          np.arange(x_freq.shape[1], dtype=int),
        "raw_kmer_index":          kmer_index.astype(int),
        "canonical_kmer":          [canonical_seq(i) for i in kmer_index],
        "mean_IBD":                mean_ibd,
        "mean_nonIBD":             mean_nonibd,
        "delta_IBD_minus_nonIBD":  delta,
        "direction":               direction,
        "t_statistic":             t_stat,
        "p_value":                 p_value,
        "p_adj_BH":                p_adj,
        "significant_FDR_0_05":    significant,
        "rank_by_p_adj":           rank_by_padj,
    }).sort_values(["p_adj_BH", "feature_column"])

    out_csv = os.path.join(OUT_DIR, "Table_S7_IBD_full_kmer_differential.csv")
    out.to_csv(out_csv, index=False)

    summary = pd.DataFrame([{
        "analysis": "Full-sample IBD k=8 differential k-mer screen",
        "n_samples": int(x_freq.shape[0]),
        "n_IBD": int(y.sum()),
        "n_nonIBD": int((y == 0).sum()),
        "total_canonical_kmers": int(x_freq.shape[1]),
        "significant_after_BH_FDR_0_05": int(significant.sum()),
        "IBD_enriched_significant": int(((delta > 0) & significant).sum()),
        "IBD_depleted_significant": int(((delta < 0) & significant).sum()),
    }])
    summary_csv = os.path.join(OUT_DIR, "Table_S7_IBD_kmer_differential_summary.csv")
    summary.to_csv(summary_csv, index=False)

    print(f"Full S7 differential table -> {out_csv}")
    print(f"S7 summary -> {summary_csv}")


if __name__ == "__main__":
    main()
