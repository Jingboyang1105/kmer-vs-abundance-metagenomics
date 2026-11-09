#!/usr/bin/env python3
"""
Complementarity analysis: dimensionality-reduction visualization comparing
the structure of the abundance and k-mer feature spaces for IBD.

Characterizes whether the two feature spaces capture non-redundant structure.
This script builds:

  1. Unsupervised PCA on each modality's full (unselected) feature matrix,
     for the same 1,338 matched IBD samples used throughout the matched-
     subset analyses, colored by diagnosis -- do the two feature spaces
     separate IBD from non-IBD along similarly- or differently-structured
     axes?
  2. A quantitative complementarity metric: Pearson correlation between the
     top 5 PCs of each modality, computed pairwise on the same samples --
     low cross-modality PC correlation is evidence the two representations
     capture non-redundant structure, not just noisier views of the same
     structure.

Reuses the exact IBD data-loading and feature-construction functions from
04_matched_dimension_check/run_matched_dimension_check.py so the sample set
and abundance feature construction are identical to the rest of the paper.

Outputs:
  results/ibd_complementarity_pca_scores.csv
  results/ibd_complementarity_pc_correlations.csv
  results/ibd_pca_variance_explained.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04_matched_dimension_check"))
import run_matched_dimension_check as base

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_PCS = 5


def main():
    print("Loading IBD metadata, abundance and k-mer matrices...")
    meta = base.load_metadata()
    ab_raw = base.load_abundance(meta)
    ab_features = pd.DataFrame(base.build_abundance_features(ab_raw), index=ab_raw.index)
    km_features = base.load_kmer_matrix(meta)

    shared = sorted(set(ab_features.index).intersection(km_features.index))
    ab_features = ab_features.loc[shared]
    km_features = km_features.loc[shared]
    labels = meta.drop_duplicates("file_id").set_index("file_id").loc[shared, "label"]
    print(f"Matched samples: {len(shared)}  IBD={labels.sum()}  nonIBD={(labels == 0).sum()}")
    print(f"Abundance feature matrix: {ab_features.shape}")
    print(f"K-mer feature matrix: {km_features.shape}")

    # ---- PCA per modality (standardized, unsupervised, full feature set) ----
    scores = {"sample_id": shared, "label": labels.values}
    variance_rows = []
    pcs = {}
    for name, X in [("abundance", ab_features.values), ("kmer", km_features.values)]:
        Xs = StandardScaler().fit_transform(X)
        pca = PCA(n_components=N_PCS, random_state=42)
        s = pca.fit_transform(Xs)
        pcs[name] = s
        for i in range(N_PCS):
            scores[f"{name}_PC{i+1}"] = s[:, i]
        for i, ev in enumerate(pca.explained_variance_ratio_, 1):
            variance_rows.append({"modality": name, "PC": i, "variance_explained": round(float(ev), 4)})
        print(f"{name}: top-{N_PCS} PCs explain {pca.explained_variance_ratio_.sum()*100:.1f}% of variance "
              f"(PC1={pca.explained_variance_ratio_[0]*100:.1f}%)")

    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(os.path.join(RESULTS_DIR, "ibd_complementarity_pca_scores.csv"), index=False)

    var_df = pd.DataFrame(variance_rows)
    var_df.to_csv(os.path.join(RESULTS_DIR, "ibd_pca_variance_explained.csv"), index=False)

    # ---- cross-modality PC correlation matrix ----
    corr_rows = []
    for i in range(N_PCS):
        for j in range(N_PCS):
            r = np.corrcoef(pcs["abundance"][:, i], pcs["kmer"][:, j])[0, 1]
            corr_rows.append({"abundance_PC": i + 1, "kmer_PC": j + 1, "pearson_r": round(float(r), 4)})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(os.path.join(RESULTS_DIR, "ibd_complementarity_pc_correlations.csv"), index=False)

    max_abs_r = corr_df["pearson_r"].abs().max()
    diag_r = [corr_df[(corr_df.abundance_PC == i + 1) & (corr_df.kmer_PC == i + 1)]["pearson_r"].iloc[0] for i in range(N_PCS)]
    print(f"\nMax |cross-modality PC correlation| among top-{N_PCS}x{N_PCS} pairs: {max_abs_r:.3f}")
    print(f"Same-index (PC1-PC1, PC2-PC2, ...) correlations: {[round(r,3) for r in diag_r]}")
    print(f"\nSaved -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
