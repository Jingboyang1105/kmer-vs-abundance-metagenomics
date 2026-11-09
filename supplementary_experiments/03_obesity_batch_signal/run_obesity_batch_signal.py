#!/usr/bin/env python3
"""
Obesity: direct test of whether the random-split k-mer advantage is driven by
between-study batch signal rather than biology.

Study identity is far easier to predict from
k-mer features than from abundance features in obesity (Section 3.3:
macro-AUC 0.9548 abundance vs. 0.9867 k-mer), which raises the
possibility that some of k-mer's advantage in the main (multi-study,
mixed-pool) random-split comparison comes from batch/cohort signal rather
than obesity biology. This script tests that possibility directly.

Direct test: compare the abundance-vs-k-mer AUC gap under two evaluation
regimes that use the *same* classifiers and feature-selection procedures as
the main obesity pipelines (archive/line3/obesity/scripts/02a_abundance_
pipeline.py, 02b_kmer_pipeline.py):

  (A) Mixed-study random split  -- participant-level 80/20 split drawn from
      ALL studies pooled together (this is what the main manuscript
      evaluates; a classifier can in principle exploit "this sample's
      per-study batch profile" because many studies appear on both sides
      of the split).
  (B) Within-study-only split   -- participant-level 80/20 split performed
      *inside a single study* (MetaCardis_2020_a, n=1,310, the largest and
      best class-balanced cMD3 obesity study: 501 non-obese / 809 obese).
      Train and test are drawn from the same single cohort/batch, so
      between-study batch signal cannot contribute to discrimination here
      by construction -- only within-cohort biological (or within-cohort
      technical) signal can.

If the k-mer-minus-abundance AUC gap seen in (A) shrinks substantially in
(B), that indicates part of the mixed-pool advantage is attributable to
cross-study batch signal. If the gap persists in (B), the k-mer advantage is
not simply an artifact of between-study leakage.

Outputs:
  results/obesity_batch_signal_all_rounds.csv
  results/obesity_batch_signal_summary.csv
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# REPO_ROOT: supplementary_experiments/03_obesity_batch_signal -> up two levels -> repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_ROOT = os.environ.get("MBEC_DATA_ROOT", REPO_ROOT)

# Produced by obesity/metadata_prep.py, obesity/download_abundance.R, and
# obesity/kmer_pipeline.py's per-sample cache; see README "Data" section.
META_CSV = os.path.join(DATA_ROOT, "obesity", "data", "obesity_metadata_full.csv")
ABUNDANCE_CSV = os.path.join(DATA_ROOT, "obesity", "data", "cmd3_relative_abundance.csv")
KMER_DIR = os.path.join(DATA_ROOT, "obesity", "data", "kmer_counts_k8")

TARGET_STUDY = "MetaCardis_2020_a"
N_ROUNDS = 10
SEEDS = [42 + i * 100 for i in range(N_ROUNDS)]
TOP_KMER = 1000
POOL_SIZE = 1600           # match the manuscript's per-round pool size for regime A
POOL_RATIO = 4              # 1:4 obese/non-obese, matching the manuscript


# ─────────────────────────── feature builders ──────────────────────────────

def clr_transform(x):
    x = x + 1e-6
    log_x = np.log(x)
    return log_x - log_x.mean(axis=1, keepdims=True)


def build_abundance_features(abund: pd.DataFrame) -> pd.DataFrame:
    x = abund.values.astype(np.float32)
    clr = clr_transform(x)
    p = x / (x.sum(axis=1, keepdims=True) + 1e-12)
    shannon = -np.sum(p * np.log(np.where(p > 0, p, 1)), axis=1)
    simpson = 1 - np.sum(p ** 2, axis=1)
    richness = (x > 0).sum(axis=1).astype(np.float32)
    pw = clr * (x > 0).astype(np.float32)
    feats = np.hstack([clr, shannon[:, None], simpson[:, None], richness[:, None], pw])
    return pd.DataFrame(feats, index=abund.index)


def load_kmer(sample_ids) -> pd.DataFrame:
    vecs, ids = [], []
    for sid in sample_ids:
        path = os.path.join(KMER_DIR, f"{sid}_k8.npy")
        if os.path.exists(path):
            v = np.load(path).astype(np.float32)
            total = v.sum()
            vecs.append(v / total if total > 0 else v)
            ids.append(sid)
    return pd.DataFrame(np.vstack(vecs), index=ids) if vecs else pd.DataFrame()


def select_kmer(x_train, y_train, x_test, top_n):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pvals = stats.ttest_ind(x_train[y_train == 1], x_train[y_train == 0], equal_var=False, axis=0)
    pvals = np.nan_to_num(pvals, nan=1.0)
    _, adj, _, _ = multipletests(pvals, method="fdr_bh")
    idx = np.argsort(adj)[:top_n]
    return x_train[:, idx], x_test[:, idx]


def make_models(seed, y_train):
    pos_w = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
    return {
        "LASSO-LR": LogisticRegression(l1_ratio=1, solver="saga", C=0.05,
                                        class_weight="balanced", max_iter=2000, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=seed,
                                                  n_jobs=4, class_weight="balanced"),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed,
                          class_weight="balanced", probability=True),
        "XGBoost": XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
                                  scale_pos_weight=pos_w, eval_metric="logloss",
                                  random_state=seed, n_jobs=4),
    }


SCALED = {"SVM (RBF)", "LASSO-LR"}


def fit_score(x_train, x_test, y_train, y_test, seed):
    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train)
    x_test_sc = scaler.transform(x_test)
    rows = []
    for name, clf in make_models(seed, y_train).items():
        xtr = x_train_sc if name in SCALED else x_train
        xte = x_test_sc if name in SCALED else x_test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(xtr, y_train)
        y_pred = clf.predict(xte)
        y_prob = clf.predict_proba(xte)[:, 1]
        rows.append({
            "classifier": name,
            "AUC": round(float(roc_auc_score(y_test, y_prob)), 4),
            "AUPRC": round(float(average_precision_score(y_test, y_prob)), 4),
            "F1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
            "MCC": round(float(matthews_corrcoef(y_test, y_pred)), 4),
        })
    return rows


# ─────────────────────────── regime A: mixed-study pool ────────────────────

def sample_mixed_pool(meta: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    pos = meta[meta["label"] == 1]
    neg = meta[meta["label"] == 0]
    n_pos = min(len(pos), POOL_SIZE // (POOL_RATIO + 1))
    n_neg = min(len(neg), n_pos * POOL_RATIO)
    pos_s = pos.sample(n=n_pos, random_state=seed)
    neg_s = neg.sample(n=n_neg, random_state=seed)
    return pd.concat([pos_s, neg_s]).sample(frac=1, random_state=seed)


def run_regime_a(meta: pd.DataFrame, abund_full: pd.DataFrame, seed: int, round_num: int):
    pool = sample_mixed_pool(meta, seed)
    rows = []

    # abundance
    ab_ids = pool["sample_id"][pool["sample_id"].isin(abund_full.index)]
    m_ab = pool[pool["sample_id"].isin(ab_ids)].drop_duplicates("sample_id").set_index("sample_id")
    ab_feat = build_abundance_features(abund_full.loc[m_ab.index])
    parts = m_ab["subject_id"].fillna(pd.Series(m_ab.index, index=m_ab.index))
    y_by_part = pd.Series(m_ab["label"].values, index=parts.values).groupby(level=0).first()
    up = parts.drop_duplicates().values
    tr_p, te_p = train_test_split(up, test_size=0.2, random_state=seed, stratify=y_by_part.loc[up].values)
    tr_mask = parts.isin(tr_p).values
    te_mask = parts.isin(te_p).values
    y = m_ab["label"].values
    for r in fit_score(ab_feat.values[tr_mask], ab_feat.values[te_mask], y[tr_mask], y[te_mask], seed):
        r.update({"regime": "A_mixed_study", "pipeline": "Abundance", "round": round_num,
                   "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum())})
        rows.append(r)

    # k-mer
    km_feat = load_kmer(pool["sample_id"].tolist())
    km_feat = km_feat[~km_feat.index.duplicated(keep="first")]
    m_km = pool[pool["sample_id"].isin(km_feat.index)].drop_duplicates("sample_id").set_index("sample_id")
    km_feat = km_feat.loc[m_km.index]
    parts_k = m_km["subject_id"].fillna(pd.Series(m_km.index, index=m_km.index))
    y_by_part_k = pd.Series(m_km["label"].values, index=parts_k.values).groupby(level=0).first()
    up_k = parts_k.drop_duplicates().values
    tr_pk, te_pk = train_test_split(up_k, test_size=0.2, random_state=seed,
                                     stratify=y_by_part_k.loc[up_k].values)
    tr_mk = parts_k.isin(tr_pk).values
    te_mk = parts_k.isin(te_pk).values
    yk = m_km["label"].values
    x_tr, x_te = select_kmer(km_feat.values[tr_mk], yk[tr_mk], km_feat.values[te_mk], TOP_KMER)
    for r in fit_score(x_tr, x_te, yk[tr_mk], yk[te_mk], seed):
        r.update({"regime": "A_mixed_study", "pipeline": "K-mer", "round": round_num,
                   "n_train": int(tr_mk.sum()), "n_test": int(te_mk.sum())})
        rows.append(r)
    return rows


# ─────────────────────────── regime B: within single study ─────────────────

def run_regime_b(meta_study: pd.DataFrame, abund_full: pd.DataFrame, seed: int, round_num: int):
    rows = []

    ab_ids = meta_study["sample_id"][meta_study["sample_id"].isin(abund_full.index)]
    m_ab = meta_study[meta_study["sample_id"].isin(ab_ids)].drop_duplicates("sample_id").set_index("sample_id")
    ab_feat = build_abundance_features(abund_full.loc[m_ab.index])
    parts = m_ab["subject_id"].fillna(pd.Series(m_ab.index, index=m_ab.index))
    y_by_part = pd.Series(m_ab["label"].values, index=parts.values).groupby(level=0).first()
    up = parts.drop_duplicates().values
    tr_p, te_p = train_test_split(up, test_size=0.2, random_state=seed, stratify=y_by_part.loc[up].values)
    tr_mask = parts.isin(tr_p).values
    te_mask = parts.isin(te_p).values
    y = m_ab["label"].values
    for r in fit_score(ab_feat.values[tr_mask], ab_feat.values[te_mask], y[tr_mask], y[te_mask], seed):
        r.update({"regime": "B_within_study", "pipeline": "Abundance", "round": round_num,
                   "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum())})
        rows.append(r)

    km_feat = load_kmer(meta_study["sample_id"].tolist())
    km_feat = km_feat[~km_feat.index.duplicated(keep="first")]
    m_km = meta_study[meta_study["sample_id"].isin(km_feat.index)].drop_duplicates("sample_id").set_index("sample_id")
    km_feat = km_feat.loc[m_km.index]
    parts_k = m_km["subject_id"].fillna(pd.Series(m_km.index, index=m_km.index))
    y_by_part_k = pd.Series(m_km["label"].values, index=parts_k.values).groupby(level=0).first()
    up_k = parts_k.drop_duplicates().values
    tr_pk, te_pk = train_test_split(up_k, test_size=0.2, random_state=seed,
                                     stratify=y_by_part_k.loc[up_k].values)
    tr_mk = parts_k.isin(tr_pk).values
    te_mk = parts_k.isin(te_pk).values
    yk = m_km["label"].values
    x_tr, x_te = select_kmer(km_feat.values[tr_mk], yk[tr_mk], km_feat.values[te_mk], TOP_KMER)
    for r in fit_score(x_tr, x_te, yk[tr_mk], yk[te_mk], seed):
        r.update({"regime": "B_within_study", "pipeline": "K-mer", "round": round_num,
                   "n_train": int(tr_mk.sum()), "n_test": int(te_mk.sum())})
        rows.append(r)
    return rows


def main():
    missing = [p for p in (META_CSV, ABUNDANCE_CSV, KMER_DIR) if not os.path.exists(p)]
    if missing:
        print("Missing required local data, cannot run obesity batch-signal test:")
        for p in missing:
            print(f"  - {p}")
        print(
            "Run obesity/metadata_prep.py, obesity/download_abundance.R, and "
            "obesity/kmer_pipeline.py first (see README 'Data' and 'Execution order'). "
            "Set MBEC_DATA_ROOT if these live outside the repo."
        )
        raise SystemExit(1)

    print("Loading metadata and abundance matrix...")
    meta = pd.read_csv(META_CSV)
    meta = meta.dropna(subset=["sample_id", "label"]).copy()
    meta["label"] = meta["label"].astype(int)
    abund_full = pd.read_csv(ABUNDANCE_CSV, index_col="sample_id")
    abund_full = abund_full[~abund_full.index.duplicated(keep="first")]

    meta_study = meta[meta["study_name"] == TARGET_STUDY].copy()
    print(f"Target study {TARGET_STUDY}: n={len(meta_study)}, "
          f"label counts={meta_study['label'].value_counts().to_dict()}")

    all_rows = []
    for i, seed in enumerate(SEEDS, 1):
        print(f"\n=== Round {i}/{N_ROUNDS} (seed={seed}) ===")
        print("-- Regime A: mixed-study pool --")
        all_rows.extend(run_regime_a(meta, abund_full, seed, i))
        print("-- Regime B: within-study (MetaCardis_2020_a) only --")
        all_rows.extend(run_regime_b(meta_study, abund_full, seed, i))

    df = pd.DataFrame(all_rows)
    out = os.path.join(RESULTS_DIR, "obesity_batch_signal_all_rounds.csv")
    df.to_csv(out, index=False)

    summary = (
        df.groupby(["regime", "pipeline", "classifier"])[["AUC", "AUPRC", "F1_macro", "MCC"]]
        .agg(["mean", "std"]).round(4)
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()

    # gap = kmer - abundance, per regime per classifier
    piv = summary.pivot(index=["regime", "classifier"], columns=[], values=[]) if False else None
    gaps = []
    for regime in summary["regime"].unique():
        for clf in summary["classifier"].unique():
            ab = summary[(summary.regime == regime) & (summary.classifier == clf) & (summary.pipeline == "Abundance")]
            km = summary[(summary.regime == regime) & (summary.classifier == clf) & (summary.pipeline == "K-mer")]
            if len(ab) and len(km):
                gaps.append({
                    "regime": regime, "classifier": clf,
                    "abundance_AUC_mean": float(ab["AUC_mean"].iloc[0]),
                    "kmer_AUC_mean": float(km["AUC_mean"].iloc[0]),
                    "AUC_gap_kmer_minus_abundance": round(float(km["AUC_mean"].iloc[0] - ab["AUC_mean"].iloc[0]), 4),
                })
    gap_df = pd.DataFrame(gaps).sort_values(["classifier", "regime"])

    summary_out = os.path.join(RESULTS_DIR, "obesity_batch_signal_summary.csv")
    summary.to_csv(summary_out, index=False)
    gap_out = os.path.join(RESULTS_DIR, "obesity_batch_signal_gap_comparison.csv")
    gap_df.to_csv(gap_out, index=False)

    print(f"\n{'='*70}\nRegime comparison: k-mer minus abundance AUC gap\n{'='*70}")
    print(gap_df.to_string(index=False))
    print(f"\nSaved -> {out}\nSaved -> {summary_out}\nSaved -> {gap_out}")


if __name__ == "__main__":
    main()
