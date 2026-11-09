#!/usr/bin/env python3
"""
Extended IBD k-mer length ablation: k in {3, 6, 7, 8, 9, 10}, 10 participant-
level splits, all 4 main-pipeline classifiers.

Provides a multi-split, multi-classifier justification for k=8 over a
wider range of k values than the original {3, 6, 8} ablation, since the original Table 1 ablation used one
split and the ten-split sensitivity (Table S9) covered only LASSO-LR.

This evaluates k in {3, 6, 7, 8, 9, 10} across the same ten participant-level
splits used throughout the manuscript, with all four classifiers (LASSO-LR,
Random Forest, SVM-RBF, XGBoost), using cached matrices for k=3,6,8
(archive/line2/kmer_counts) and newly extracted matrices for k=7,9,10
(extract_new_k.sh -> data/kmer_counts/).

Outputs:
  results/ibd_k_ablation_extended_all_rounds.csv
  results/ibd_k_ablation_extended_summary.csv
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

# REPO_ROOT: supplementary_experiments/06_kmer_k_ablation_extended -> up two levels -> repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_ROOT = os.environ.get("MBEC_DATA_ROOT", REPO_ROOT)

META_CSV = os.path.join(DATA_ROOT, "ibd", "data", "raw", "hmp2_metadata_2018-08-20.csv")
CACHED_DIR = os.path.join(DATA_ROOT, "ibd", "kmer_matrix")              # k3, k6, k8
NEW_DIR = os.path.join(SCRIPT_DIR, "data", "kmer_counts")               # k7, k9, k10

K_VALUES = [3, 6, 7, 8, 9, 10]
TOP_KMER = 1000
SEEDS = [42 + i * 100 for i in range(10)]


def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(META_CSV, low_memory=False)
    meta = meta[meta["data_type"] == "metagenomics"].copy()
    meta = meta[meta["diagnosis"].isin(["CD", "UC", "nonIBD"])].copy()
    meta["diagnosis"] = meta["diagnosis"].replace({"CD": "IBD", "UC": "IBD"})
    meta = meta[["External ID", "Participant ID", "diagnosis"]].drop_duplicates("External ID")
    meta.rename(columns={"External ID": "sample_id", "Participant ID": "participant"}, inplace=True)
    meta["file_id"] = meta["sample_id"].str.replace(r"_P$", "", regex=True)
    meta["label"] = (meta["diagnosis"] == "IBD").astype(int)
    return meta.drop_duplicates("file_id").set_index("file_id")


def matrix_dir_for_k(k: int) -> tuple[str, str]:
    if k in (3, 6, 8):
        return os.path.join(CACHED_DIR, f"k{k}"), f"_k{k}.npy"
    return NEW_DIR, f"_k{k}.npy"


def load_matrix(k: int, meta: pd.DataFrame):
    directory, suffix = matrix_dir_for_k(k)
    files = {f[: -len(suffix)]: os.path.join(directory, f)
             for f in os.listdir(directory) if f.endswith(suffix)}
    ids = sorted(set(files).intersection(meta.index))
    vecs = []
    for fid in ids:
        v = np.load(files[fid]).astype(np.float32)
        total = v.sum()
        vecs.append(v / total if total > 0 else v)
    X = np.vstack(vecs)
    y = meta.loc[ids, "label"].values
    parts = meta.loc[ids, "participant"].values
    return X, y, parts, ids


def select_kmers(x_train, y_train, top_n):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pvals = stats.ttest_ind(x_train[y_train == 0], x_train[y_train == 1], equal_var=False, axis=0)
    pvals = np.nan_to_num(pvals, nan=1.0)
    _, adj, _, _ = multipletests(pvals, method="fdr_bh")
    return np.argsort(adj)[: min(top_n, x_train.shape[1])]


def make_models(seed, y_train):
    pos_w = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
    return {
        "LASSO-LR": LogisticRegression(l1_ratio=1.0, solver="saga", C=0.05,
                                        class_weight="balanced", max_iter=5000, random_state=seed),
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


def evaluate_k(k, X, y, parts):
    unique_parts = pd.Series(parts).drop_duplicates().values
    part_label = pd.Series(y, index=parts).groupby(level=0).first()
    rows = []
    for round_num, seed in enumerate(SEEDS, 1):
        parts_train, parts_test = train_test_split(
            unique_parts, test_size=0.2, random_state=seed,
            stratify=part_label.loc[unique_parts].values,
        )
        tr = np.isin(parts, parts_train)
        te = np.isin(parts, parts_test)
        x_tr_all, x_te_all = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        idx = select_kmers(x_tr_all, y_tr, TOP_KMER)
        x_tr, x_te = x_tr_all[:, idx], x_te_all[:, idx]
        scaler = StandardScaler()
        x_tr_sc = scaler.fit_transform(x_tr)
        x_te_sc = scaler.transform(x_te)

        for name, clf in make_models(seed, y_tr).items():
            xtr = x_tr_sc if name in SCALED else x_tr
            xte = x_te_sc if name in SCALED else x_te
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(xtr, y_tr)
            y_pred = clf.predict(xte)
            y_prob = clf.predict_proba(xte)[:, 1]
            auc = round(float(roc_auc_score(y_te, y_prob)), 4)
            print(f"  k={k:2d} round={round_num} {name:15s} AUC={auc:.3f}")
            rows.append({
                "k": k, "round": round_num, "seed": seed, "classifier": name,
                "n_train": int(tr.sum()), "n_test": int(te.sum()),
                "AUC": auc,
                "AUPRC": round(float(average_precision_score(y_te, y_prob)), 4),
                "F1_macro": round(float(f1_score(y_te, y_pred, average="macro")), 4),
                "MCC": round(float(matthews_corrcoef(y_te, y_pred)), 4),
            })
    return rows


def main():
    missing = [p for p in (META_CSV, CACHED_DIR, NEW_DIR) if not os.path.exists(p)]
    if missing:
        print("Missing required local data, cannot run extended IBD k-ablation:")
        for p in missing:
            print(f"  - {p}")
        print(
            "Run ibd/extract_kmers.py + ibd/build_kmer_matrix.py for k in {3,6,8}, place "
            "the HMP2 metadata CSV under ibd/data/raw/, and run extract_new_k.sh in this "
            "folder for k in {7,9,10} (see README 'Data' section). Set MBEC_DATA_ROOT if "
            "these live outside the repo."
        )
        raise SystemExit(1)

    meta = load_metadata()
    all_rows = []
    for k in K_VALUES:
        print(f"\n=== k={k} ===")
        X, y, parts, ids = load_matrix(k, meta)
        print(f"  n={len(ids)}  IBD={int(y.sum())}  nonIBD={int((y==0).sum())}")
        all_rows.extend(evaluate_k(k, X, y, parts))

    df = pd.DataFrame(all_rows)
    out = os.path.join(RESULTS_DIR, "ibd_k_ablation_extended_all_rounds.csv")
    df.to_csv(out, index=False)

    summary = (
        df.groupby(["k", "classifier"])[["AUC", "AUPRC", "F1_macro", "MCC"]]
        .agg(["mean", "std"]).round(4)
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index().sort_values(["classifier", "k"])
    summary_out = os.path.join(RESULTS_DIR, "ibd_k_ablation_extended_summary.csv")
    summary.to_csv(summary_out, index=False)

    print(f"\n{'='*70}\nExtended k-ablation summary (k=3..10, 10 splits, 4 classifiers)\n{'='*70}")
    print(summary.to_string(index=False))
    print(f"\nSaved -> {out}\nSaved -> {summary_out}")


if __name__ == "__main__":
    main()
