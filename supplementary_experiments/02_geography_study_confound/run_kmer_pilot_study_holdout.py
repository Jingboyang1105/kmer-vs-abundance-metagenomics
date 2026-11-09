#!/usr/bin/env python3
"""
Geography k-mer study-held-out pilot classification.

Companion to extract_kmer_pilot.py: once k=8 vectors are extracted for the
1,587-sample round-1 pool (data/kmer_counts_pilot/), re-evaluate 30-country
classification under the same study-held-out design already used for the
abundance pipeline (run_geography_study_confound.py Part B), so the two are
directly comparable on an identical sample pool.

Feature selection matches the published cMD3 k-mer pipeline (ANOVA F-test +
BH correction, top 1000 dimensions; archive/line3/geography/scripts/
02b_kmer_pipeline.py) rather than the ElasticNet selection used for
abundance, since that is the k-mer-appropriate procedure used throughout the
manuscript.

Scope: n=1,587 (one round), not the full n=22,397 cohort -- reported as a
bounded pilot, not a replacement for the full-cohort analysis.

Outputs:
  results/geography_kmer_pilot_study_holdout_all_rounds.csv
  results/geography_kmer_pilot_study_holdout_summary.csv
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# REPO_ROOT: supplementary_experiments/02_geography_study_confound -> up two levels -> repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_ROOT = os.environ.get("MBEC_DATA_ROOT", REPO_ROOT)

# Produced by geography/metadata_prep.py; see README "Execution order".
META_CSV = os.path.join(DATA_ROOT, "geography", "data", "geo_metadata_round1.csv")
KMER_DIR = os.path.join(SCRIPT_DIR, "data", "kmer_counts_pilot")

N_REPEATS = 5
TEST_STUDY_FRACTION = 0.20
MIN_TEST_SAMPLES_PER_COUNTRY = 3
TOP_KMER = 1000


def load_kmer_matrix(meta: pd.DataFrame) -> pd.DataFrame:
    files = {f[: -len("_k8.npy")]: os.path.join(KMER_DIR, f)
             for f in os.listdir(KMER_DIR) if f.endswith("_k8.npy")}
    ids = sorted(set(files).intersection(meta["sample_id"]))
    vecs = [np.load(files[i]).astype(np.float32) for i in ids]
    return pd.DataFrame(np.vstack(vecs), index=ids)


def select_kmers_multiclass(x_train, y_train, top_k=TOP_KMER):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        groups = [x_train[y_train == c] for c in np.unique(y_train)]
        _, pvals = f_oneway(*groups)
    pvals = np.nan_to_num(pvals, nan=1.0)
    _, adj, _, _ = multipletests(pvals, method="fdr_bh")
    sig = np.where(adj < 0.05)[0]
    if len(sig) < 10:
        idx = np.argsort(adj)[:top_k]
    else:
        idx = sig[np.argsort(adj[sig])][:top_k]
    return idx


def multiclass_metrics(y_true, y_pred, y_prob):
    present = np.unique(y_true)
    auc_vals, auprc_vals = [], []
    for cls in present:
        if cls >= y_prob.shape[1]:
            continue
        y_bin = (y_true == cls).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        try:
            auc_vals.append(roc_auc_score(y_bin, y_prob[:, cls]))
        except Exception:
            pass
        try:
            auprc_vals.append(average_precision_score(y_bin, y_prob[:, cls]))
        except Exception:
            pass
    auc = float(np.mean(auc_vals)) if auc_vals else float("nan")
    auprc = float(np.mean(auprc_vals)) if auprc_vals else float("nan")
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    return round(auc, 4), round(auprc, 4), round(float(f1), 4), round(float(mcc), 4)


def study_held_out_split(meta: pd.DataFrame, seed: int):
    rng = np.random.RandomState(seed)
    studies = meta["study_name"].unique().tolist()
    rng.shuffle(studies)
    n_test_studies = max(1, int(len(studies) * TEST_STUDY_FRACTION))
    test_studies = set(studies[:n_test_studies])
    train_idx = meta.index[~meta["study_name"].isin(test_studies)]
    test_idx = meta.index[meta["study_name"].isin(test_studies)]
    return train_idx, test_idx, test_studies


def run_repeat(meta_kmer: pd.DataFrame, km_features: pd.DataFrame, seed: int):
    meta_kmer = meta_kmer.set_index("sample_id")
    train_idx, test_idx, test_studies = study_held_out_split(meta_kmer, seed)

    train_countries = set(meta_kmer.loc[train_idx, "country"].unique())
    test_all = meta_kmer.loc[test_idx]
    n_test_total = len(test_all)
    test_keep = test_all[test_all["country"].isin(train_countries)]
    n_dropped = n_test_total - len(test_keep)
    country_counts = test_keep["country"].value_counts()
    keep_countries = country_counts[country_counts >= MIN_TEST_SAMPLES_PER_COUNTRY].index
    test_keep = test_keep[test_keep["country"].isin(keep_countries)]

    print(f"  [seed={seed}] held-out studies={len(test_studies)}  "
          f"test total={n_test_total}  dropped(unseen country)={n_dropped} "
          f"({100*n_dropped/max(n_test_total,1):.1f}%)  scored={len(test_keep)}")

    if len(test_keep) < 20 or test_keep["country"].nunique() < 2:
        return [], n_test_total, n_dropped

    le = LabelEncoder()
    le.fit(meta_kmer.loc[train_idx, "country"].values)
    y_train = le.transform(meta_kmer.loc[train_idx, "country"].values)
    y_test = le.transform(test_keep["country"].values)

    X_train = km_features.loc[train_idx].values
    X_test = km_features.loc[test_keep.index].values

    sel_idx = select_kmers_multiclass(X_train, y_train)
    X_train_sel = X_train[:, sel_idx]
    X_test_sel = X_test[:, sel_idx]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_sel)
    X_test_sc = scaler.transform(X_test_sel)

    n_classes = len(le.classes_)
    models = {
        "LASSO-LR": LogisticRegression(l1_ratio=1, solver="saga", C=0.05,
                                        class_weight="balanced", max_iter=2000, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=seed,
                                                  n_jobs=4, class_weight="balanced"),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed,
                          class_weight="balanced", probability=True, decision_function_shape="ovr"),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
                                  num_class=n_classes, objective="multi:softprob",
                                  eval_metric="mlogloss", random_state=seed, n_jobs=4),
    }
    rows = []
    for name, clf in models.items():
        sc = name in ("SVM (RBF)", "LASSO-LR")
        xtr = X_train_sc if sc else X_train_sel
        xte = X_test_sc if sc else X_test_sel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(xtr, y_train)
        y_pred = clf.predict(xte)
        y_prob = clf.predict_proba(xte)
        auc, auprc, f1, mcc = multiclass_metrics(y_test, y_pred, y_prob)
        print(f"    {name:15s} AUC={auc:.3f}  F1={f1:.3f}  MCC={mcc:.3f}  (n_kmer_sel={len(sel_idx)})")
        rows.append({
            "seed": seed, "classifier": name, "pipeline": "K-mer (study-held-out pilot)",
            "n_train": int(len(y_train)), "n_test_scored": int(len(y_test)),
            "n_test_dropped_unseen_country": int(n_dropped),
            "n_held_out_studies": len(test_studies), "n_kmer_selected": len(sel_idx),
            "AUC": auc, "AUPRC": auprc, "F1_macro": f1, "MCC": mcc,
        })
    return rows, n_test_total, n_dropped


def main():
    if not os.path.exists(META_CSV):
        print(f"Missing {META_CSV}")
        print(
            "Run geography/metadata_prep.py first (see README 'Execution order'). "
            "Set MBEC_DATA_ROOT if it lives outside the repo."
        )
        raise SystemExit(1)
    if not os.path.isdir(KMER_DIR) or not os.listdir(KMER_DIR):
        print(f"Missing k-mer pilot vectors at {KMER_DIR}")
        print("Run extract_kmer_pilot.py in this same folder first.")
        raise SystemExit(1)

    print("Loading round-1 metadata and k-mer pilot matrix...")
    meta_full = pd.read_csv(META_CSV)
    km_features = load_kmer_matrix(meta_full)
    print(f"  k-mer vectors available: {len(km_features)} / {len(meta_full)} round-1 samples")
    meta_kmer = meta_full[meta_full["sample_id"].isin(km_features.index)].drop_duplicates("sample_id").copy()
    print(f"  Using {len(meta_kmer)} samples with k-mer vectors, "
          f"{meta_kmer['country'].nunique()} countries, {meta_kmer['study_name'].nunique()} studies")

    all_rows = []
    for i in range(N_REPEATS):
        seed = 4000 + i
        print(f"\n== Study-held-out k-mer pilot repeat {i+1}/{N_REPEATS} (seed={seed}) ==")
        rows, n_total, n_dropped = run_repeat(meta_kmer, km_features, seed)
        all_rows.extend(rows)

    if not all_rows:
        print("No usable repeats.")
        return

    df = pd.DataFrame(all_rows)
    out = os.path.join(RESULTS_DIR, "geography_kmer_pilot_study_holdout_all_rounds.csv")
    df.to_csv(out, index=False)

    summary = (
        df.groupby("classifier")[["AUC", "AUPRC", "F1_macro", "MCC"]]
        .agg(["mean", "std", "count"]).round(4)
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary_out = os.path.join(RESULTS_DIR, "geography_kmer_pilot_study_holdout_summary.csv")
    summary.to_csv(summary_out, index=False)

    print(f"\n{'='*70}\nGeography k-mer study-held-out PILOT summary ({N_REPEATS} repeats, n=1,587 pool)\n{'='*70}")
    print(summary.to_string(index=False))
    print(f"\nSaved -> {out}\nSaved -> {summary_out}")


if __name__ == "__main__":
    main()
