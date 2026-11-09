"""
IBD feature-selection sensitivity analysis.

This analysis addresses whether the main abundance-vs-k-mer comparison depends
on using ElasticNet feature selection for abundance and univariate Welch tests
for k-mers. It restricts the analysis to samples shared by both modalities and
uses the same univariate Welch t-test + BH ranking procedure for both feature
sets inside each training split.

Outputs:
  results/ibd_univariate_feature_selection_sensitivity.csv
  supplementary_tables/Table_S10_IBD_univariate_feature_selection_sensitivity.csv
"""

import os
import sys
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


ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
SUPP_DIR = os.path.join(ROOT, "..", "supplementary_tables")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SUPP_DIR, exist_ok=True)

sys.path.insert(0, ROOT)
from abundance_pipeline import build_features, load_data  # noqa: E402


SEEDS = [42 + i * 100 for i in range(10)]
TOP_ABUNDANCE = 100
TOP_KMER = 1000
METRICS = ["AUC", "AUPRC", "F1_macro", "MCC"]


def load_matched_modalities():
    abundance_df, abundance_labels, abundance_participants = load_data()
    abundance_features = build_features(abundance_df)
    abundance_ids = abundance_df.index.astype(str).to_numpy()

    kmer_prefix = os.path.join(ROOT, "kmer_counts", "kmer_matrix_k8")
    kmer_features = np.load(f"{kmer_prefix}_X.npy")
    kmer_labels = np.load(f"{kmer_prefix}_labels.npy", allow_pickle=True)
    kmer_participants = np.load(f"{kmer_prefix}_participant.npy", allow_pickle=True)
    kmer_ids = np.load(f"{kmer_prefix}_sample_ids.npy", allow_pickle=True).astype(str)

    abundance_pos = {sample_id: i for i, sample_id in enumerate(abundance_ids)}
    kmer_pos = {sample_id: i for i, sample_id in enumerate(kmer_ids)}
    shared_ids = sorted(set(abundance_pos).intersection(kmer_pos))

    a_idx = np.array([abundance_pos[sample_id] for sample_id in shared_ids])
    k_idx = np.array([kmer_pos[sample_id] for sample_id in shared_ids])

    y_abundance = (abundance_labels[a_idx] == "IBD").astype(int)
    y_kmer = (kmer_labels[k_idx] == "IBD").astype(int)
    if not np.array_equal(y_abundance, y_kmer):
        raise RuntimeError("Matched abundance and k-mer labels do not agree.")

    abundance_part = abundance_participants[a_idx].astype(str)
    kmer_part = kmer_participants[k_idx].astype(str)
    if not np.array_equal(abundance_part, kmer_part):
        raise RuntimeError("Matched abundance and k-mer participant IDs do not agree.")

    return {
        "sample_ids": np.array(shared_ids),
        "participants": abundance_part,
        "labels": y_abundance,
        "abundance": abundance_features[a_idx],
        "kmer": kmer_features[k_idx],
    }


def select_univariate(x_train, y_train, x_test, top_n):
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
    idx = np.argsort(pvals_bh)[: min(top_n, x_train.shape[1])]
    return x_train[:, idx], x_test[:, idx], len(idx)


def make_models(seed, y_train):
    return {
        "LASSO-LR": LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=0.05,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        ),
    }


def evaluate_pipeline(x_train_all, x_test_all, y_train, y_test, pipeline, seed, round_num):
    top_n = TOP_ABUNDANCE if pipeline == "Abundance univariate" else TOP_KMER
    x_train, x_test, n_selected = select_univariate(
        x_train_all, y_train, x_test_all, top_n=top_n,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    rows = []
    for clf_name, clf in make_models(seed, y_train).items():
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
            "round": round_num,
            "seed": seed,
            "pipeline": pipeline,
            "classifier": clf_name,
            "n_selected": n_selected,
            "AUC": round(float(roc_auc_score(y_test, y_score)), 4),
            "AUPRC": round(float(average_precision_score(y_test, y_score)), 4),
            "F1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
            "MCC": round(float(matthews_corrcoef(y_test, y_pred)), 4),
        })
    return rows


def main():
    data = load_matched_modalities()
    y = data["labels"]
    participants = data["participants"]
    unique_parts = pd.Series(participants).drop_duplicates().values
    part_label = pd.Series(y, index=participants).groupby(level=0).first()

    all_rows = []
    for round_num, seed in enumerate(SEEDS, 1):
        print(f"Round {round_num}, seed={seed}", flush=True)
        parts_train, parts_test = train_test_split(
            unique_parts,
            test_size=0.2,
            random_state=seed,
            stratify=part_label.loc[unique_parts].values,
        )
        train_mask = np.isin(participants, parts_train)
        test_mask = np.isin(participants, parts_test)
        y_train = y[train_mask]
        y_test = y[test_mask]

        all_rows.extend(evaluate_pipeline(
            data["abundance"][train_mask], data["abundance"][test_mask],
            y_train, y_test, "Abundance univariate", seed, round_num,
        ))
        all_rows.extend(evaluate_pipeline(
            data["kmer"][train_mask], data["kmer"][test_mask],
            y_train, y_test, "K-mer univariate", seed, round_num,
        ))

    all_df = pd.DataFrame(all_rows)
    all_out = os.path.join(RESULTS_DIR, "ibd_univariate_feature_selection_sensitivity.csv")
    all_df.to_csv(all_out, index=False)

    summary = (
        all_df
        .groupby(["pipeline", "classifier"])[METRICS]
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

    comparisons = []
    for clf_name in sorted(all_df["classifier"].unique()):
        a = all_df[
            (all_df["pipeline"] == "Abundance univariate") & (all_df["classifier"] == clf_name)
        ].sort_values("round")
        k = all_df[
            (all_df["pipeline"] == "K-mer univariate") & (all_df["classifier"] == clf_name)
        ].sort_values("round")
        for metric in METRICS:
            comparisons.append({
                "classifier": clf_name,
                "metric": metric,
                "mean_abundance": round(float(a[metric].mean()), 4),
                "mean_kmer": round(float(k[metric].mean()), 4),
                "delta_kmer_minus_abundance": round(float(k[metric].mean() - a[metric].mean()), 4),
            })
    comp_df = pd.DataFrame(comparisons)
    summary_out = os.path.join(SUPP_DIR, "Table_S10_IBD_univariate_feature_selection_sensitivity.csv")
    comp_df.to_csv(summary_out, index=False)

    print(f"Saved full results -> {all_out}")
    print(f"Saved comparison summary -> {summary_out}")
    print(comp_df[comp_df["metric"] == "AUC"].to_string(index=False))


if __name__ == "__main__":
    main()
