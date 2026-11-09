"""
Repeated IBD k-mer length ablation (LASSO-LR only).

Uses precomputed HMP2 k-mer matrices for k = 3, 6, and 8 and evaluates the
same ten participant-level splits used in the main IBD benchmark. Feature
selection is performed inside each training split. Only LASSO-LR is evaluated
here; the main benchmark (kmer_pipeline.py) covers all four classifiers.

Outputs:
  ../supplementary_tables/Table_S9_IBD_kmer_k_ablation_repeated.csv
  results/ibd_kmer_k_ablation_repeated.csv
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


ROOT = os.path.dirname(os.path.abspath(__file__))
KMER_DIR = os.path.join(ROOT, "kmer_counts")
RESULTS_DIR = os.path.join(ROOT, "results")
SUPP_DIR = os.path.join(ROOT, "..", "supplementary_tables")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SUPP_DIR, exist_ok=True)

K_VALUES = [3, 6, 8]
TOP_N = 1000
SEEDS = [42 + i * 100 for i in range(10)]


def select_kmers(x_train, y_train, top_n):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pvals = stats.ttest_ind(
            x_train[y_train == 0],
            x_train[y_train == 1],
            equal_var=False,
            axis=0,
        )
    pvals = np.nan_to_num(pvals, nan=1.0, posinf=1.0, neginf=1.0)
    _, pvals_bh, _, _ = multipletests(pvals, method="fdr_bh")
    return np.argsort(pvals_bh)[: min(top_n, x_train.shape[1])]


def make_lasso_model(seed, y_train):
    return {
        "LASSO-LR": LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=0.05,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        ),
    }


def evaluate_one_k(k):
    prefix = os.path.join(KMER_DIR, f"kmer_matrix_k{k}")
    x = np.load(f"{prefix}_X.npy")
    labels = np.load(f"{prefix}_labels.npy", allow_pickle=True)
    participants = np.load(f"{prefix}_participant.npy", allow_pickle=True)
    y = (labels == "IBD").astype(int)
    unique_parts = pd.Series(participants).drop_duplicates().values
    part_label = pd.Series(y, index=participants).groupby(level=0).first()

    rows = []
    for round_num, seed in enumerate(SEEDS, 1):
        print(f"k={k}, round={round_num}, seed={seed}", flush=True)
        parts_train, parts_test = train_test_split(
            unique_parts,
            test_size=0.2,
            random_state=seed,
            stratify=part_label.loc[unique_parts].values,
        )
        train_mask = np.isin(participants, parts_train)
        test_mask = np.isin(participants, parts_test)
        x_train_all = x[train_mask]
        x_test_all = x[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        selected = select_kmers(x_train_all, y_train, TOP_N)
        x_train = x_train_all[:, selected]
        x_test = x_test_all[:, selected]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        for clf_name, clf in make_lasso_model(seed, y_train).items():
            use_scaled = clf_name in {"LASSO-LR", "SVM (RBF)"}
            xtr = x_train_scaled if use_scaled else x_train
            xte = x_test_scaled if use_scaled else x_test
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(xtr, y_train)
            y_pred = clf.predict(xte)
            if hasattr(clf, "predict_proba"):
                y_score = clf.predict_proba(xte)[:, 1]
            else:
                y_score = clf.decision_function(xte)
            rows.append({
                "k": k,
                "round": round_num,
                "seed": seed,
                "classifier": clf_name,
                "n_selected": len(selected),
                "AUC": round(float(roc_auc_score(y_test, y_score)), 4),
                "AUPRC": round(float(average_precision_score(y_test, y_score)), 4),
                "F1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
                "MCC": round(float(matthews_corrcoef(y_test, y_pred)), 4),
            })
    return rows


def main():
    rows = []
    for k in K_VALUES:
        rows.extend(evaluate_one_k(k))

    all_df = pd.DataFrame(rows)
    all_out = os.path.join(RESULTS_DIR, "ibd_kmer_k_ablation_repeated.csv")
    all_df.to_csv(all_out, index=False)

    metric_cols = ["AUC", "AUPRC", "F1_macro", "MCC"]
    summary = (
        all_df
        .groupby(["k", "classifier"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    for col in summary.columns:
        if col.endswith("_mean") or col.endswith("_std"):
            summary[col] = summary[col].round(4)

    summary_out = os.path.join(SUPP_DIR, "Table_S9_IBD_kmer_k_ablation_repeated.csv")
    summary.to_csv(summary_out, index=False)

    print(f"Saved full results -> {all_out}")
    print(f"Saved summary -> {summary_out}")
    print(summary[["k", "classifier", "AUC_mean", "AUC_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
