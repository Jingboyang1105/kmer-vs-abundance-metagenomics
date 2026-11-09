#!/usr/bin/env python3
"""
Obesity matched-dimension check: same question as the IBD matched-dimension
check (run_matched_dimension_check.py), extended to obesity.

The main obesity comparison uses k-mer top-1,000 dimensions versus ~400
abundance-derived dimensions (CLR + alpha-diversity + presence-weighted CLR,
~200 species after >=10% prevalence filtering on a typical round-1 pool) --
a smaller imbalance than IBD's 1,000-vs-251, but the same underlying concern
applies: is part of the headline obesity k-mer advantage attributable to the
larger k-mer feature space rather than representation quality?

Both modalities are selected with an identical univariate procedure (Welch
t-test + BH ranking) and capped at the identical dimension count (50, 100,
250, 500), on the mixed-study pool (same sampling design as the main obesity
pipeline / regime A of run_obesity_batch_signal.py), for a like-for-like
comparison against the headline Table 2 result.

Outputs:
  results/obesity_matched_dimension_all_rounds.csv
  results/obesity_matched_dimension_summary.csv
  results/obesity_matched_dimension_gap.csv
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "03_obesity_batch_signal"))
import run_obesity_batch_signal as base

RESULTS_DIR = base.RESULTS_DIR
MATCHED_DIMS = [50, 100, 250, 500]


def select_univariate(x_train, y_train, x_test, top_n):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pvals = stats.ttest_ind(x_train[y_train == 1], x_train[y_train == 0], equal_var=False, axis=0)
    pvals = np.nan_to_num(pvals, nan=1.0, posinf=1.0, neginf=1.0)
    _, adj, _, _ = multipletests(pvals, method="fdr_bh")
    idx = np.argsort(adj)[: min(top_n, x_train.shape[1])]
    return x_train[:, idx], x_test[:, idx]


def run_round(meta, abund_full, seed, round_num):
    pool = base.sample_mixed_pool(meta, seed)
    rows = []

    # ---- abundance side ----
    ab_ids = pool["sample_id"][pool["sample_id"].isin(abund_full.index)]
    m_ab = pool[pool["sample_id"].isin(ab_ids)].drop_duplicates("sample_id").set_index("sample_id")
    ab_feat = base.build_abundance_features(abund_full.loc[m_ab.index])

    # ---- k-mer side ----
    km_feat = base.load_kmer(pool["sample_id"].tolist())
    km_feat = km_feat[~km_feat.index.duplicated(keep="first")]
    m_km = pool[pool["sample_id"].isin(km_feat.index)].drop_duplicates("sample_id").set_index("sample_id")
    km_feat = km_feat.loc[m_km.index]

    # matched sample set: present in BOTH modalities, so dimension is the only variable
    shared = sorted(set(m_ab.index).intersection(m_km.index))
    if len(shared) < 60:
        print(f"  round {round_num}: only {len(shared)} matched samples, skipping")
        return rows
    ab_feat = ab_feat.loc[shared]
    km_feat = km_feat.loc[shared]
    m_shared = m_ab.loc[shared]

    parts = m_shared["subject_id"].fillna(pd.Series(m_shared.index, index=m_shared.index))
    y_by_part = pd.Series(m_shared["label"].values, index=parts.values).groupby(level=0).first()
    up = parts.drop_duplicates().values
    tr_p, te_p = train_test_split(up, test_size=0.2, random_state=seed, stratify=y_by_part.loc[up].values)
    tr = parts.isin(tr_p).values
    te = parts.isin(te_p).values
    y = m_shared["label"].values
    y_tr, y_te = y[tr], y[te]
    print(f"  round {round_num}: matched n={len(shared)}  train={tr.sum()}  test={te.sum()}")

    for dim in MATCHED_DIMS:
        ab_tr, ab_te = select_univariate(ab_feat.values[tr], y_tr, ab_feat.values[te], dim)
        km_tr, km_te = select_univariate(km_feat.values[tr], y_tr, km_feat.values[te], dim)
        for modality, (x_tr, x_te) in [("Abundance", (ab_tr, ab_te)), ("K-mer", (km_tr, km_te))]:
            scaler = StandardScaler()
            x_tr_sc = scaler.fit_transform(x_tr)
            x_te_sc = scaler.transform(x_te)
            for name, clf in base.make_models(seed, y_tr).items():
                xtr = x_tr_sc if name in base.SCALED else x_tr
                xte = x_te_sc if name in base.SCALED else x_te
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf.fit(xtr, y_tr)
                y_prob = clf.predict_proba(xte)[:, 1]
                from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
                y_pred = clf.predict(xte)
                print(f"    dim={dim:4d} {modality:9s} {name:15s} AUC={roc_auc_score(y_te, y_prob):.3f}")
                rows.append({
                    "round": round_num, "seed": seed, "matched_dim": dim,
                    "modality": modality, "classifier": name,
                    "AUC": round(float(roc_auc_score(y_te, y_prob)), 4),
                    "AUPRC": round(float(average_precision_score(y_te, y_prob)), 4),
                    "F1_macro": round(float(f1_score(y_te, y_pred, average="macro")), 4),
                    "MCC": round(float(matthews_corrcoef(y_te, y_pred)), 4),
                })
    return rows


def main():
    print("Loading obesity metadata and abundance matrix...")
    meta = pd.read_csv(base.META_CSV)
    meta = meta.dropna(subset=["sample_id", "label"]).copy()
    meta["label"] = meta["label"].astype(int)
    abund_full = pd.read_csv(base.ABUNDANCE_CSV, index_col="sample_id")
    abund_full = abund_full[~abund_full.index.duplicated(keep="first")]

    all_rows = []
    for i, seed in enumerate(base.SEEDS, 1):
        print(f"\n=== Round {i}/{base.N_ROUNDS} (seed={seed}) ===")
        all_rows.extend(run_round(meta, abund_full, seed, i))

    df = pd.DataFrame(all_rows)
    out = os.path.join(RESULTS_DIR, "obesity_matched_dimension_all_rounds.csv")
    df.to_csv(out, index=False)

    summary = (
        df.groupby(["matched_dim", "modality", "classifier"])[["AUC", "AUPRC", "F1_macro", "MCC"]]
        .agg(["mean", "std"]).round(4)
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary_out = os.path.join(RESULTS_DIR, "obesity_matched_dimension_summary.csv")
    summary.to_csv(summary_out, index=False)

    gaps = []
    for dim in MATCHED_DIMS:
        for clf in df["classifier"].unique():
            a = summary[(summary.matched_dim == dim) & (summary.classifier == clf) & (summary.modality == "Abundance")]
            k = summary[(summary.matched_dim == dim) & (summary.classifier == clf) & (summary.modality == "K-mer")]
            if len(a) and len(k):
                gaps.append({
                    "matched_dim": dim, "classifier": clf,
                    "abundance_AUC": float(a["AUC_mean"].iloc[0]),
                    "kmer_AUC": float(k["AUC_mean"].iloc[0]),
                    "gap_kmer_minus_abundance": round(float(k["AUC_mean"].iloc[0] - a["AUC_mean"].iloc[0]), 4),
                })
    gap_df = pd.DataFrame(gaps)
    gap_out = os.path.join(RESULTS_DIR, "obesity_matched_dimension_gap.csv")
    gap_df.to_csv(gap_out, index=False)

    print(f"\n{'='*70}\nObesity matched-dimension gap (k-mer minus abundance AUC), by dim\n{'='*70}")
    print(gap_df.to_string(index=False))
    print(f"\nSaved -> {out}\nSaved -> {summary_out}\nSaved -> {gap_out}")


if __name__ == "__main__":
    main()
