"""
Step 2a – Abundance pipeline for obesity (BMI classification), 10-round subsampling.
Input:  data/cmd3_relative_abundance.csv      (from download_abundance.R)
        splits/obesity/obesity_metadata_round{1..10}.csv  (from metadata_prep.py)
Output: results/abundance_round{1..10}.csv
        results/abundance_all_rounds.csv  (all rounds combined)
        results/results_abundance.csv     (mean ± SD summary)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "../data")
SPLITS_DIR  = os.path.join(SCRIPT_DIR, "../splits/obesity")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

ABUNDANCE_CSV = os.path.join(DATA_DIR, "cmd3_relative_abundance.csv")
N_ROUNDS = 10


def clr_transform(df):
    X = df.values + 1e-6
    log_X = np.log(X)
    return pd.DataFrame(log_X - log_X.mean(axis=1, keepdims=True),
                        index=df.index, columns=df.columns)


def build_features(abund):
    X_clr = clr_transform(abund)
    X_vals = abund.values
    p = X_vals / (X_vals.sum(axis=1, keepdims=True) + 1e-12)
    shannon  = -np.sum(p * np.log(np.where(p > 0, p, 1)), axis=1)
    simpson  = 1 - np.sum(p ** 2, axis=1)
    richness = (X_vals > 0).sum(axis=1).astype(float)
    X_div = pd.DataFrame({"alpha_shannon": shannon, "alpha_simpson": simpson,
                           "alpha_richness": richness}, index=abund.index)
    pw = X_clr.values * (abund.values > 0).astype(float)
    X_pw = pd.DataFrame(pw, index=abund.index,
                        columns=[f"pw__{c}" for c in X_clr.columns])
    return pd.concat([X_clr, X_div, X_pw], axis=1)


def run_one_round(meta, abund_full, round_num):
    shared = meta["sample_id"][meta["sample_id"].isin(abund_full.index)]
    meta   = meta[meta["sample_id"].isin(shared)].set_index("sample_id")
    abund  = abund_full.loc[shared]

    prev  = (abund > 0).mean()
    abund = abund.loc[:, prev >= 0.10]
    X_full = build_features(abund)

    train_idx = meta.index[meta["split"] == "train"]
    test_idx  = meta.index[meta["split"] == "test"]
    y_train = meta.loc[train_idx, "label"].values
    y_test  = meta.loc[test_idx,  "label"].values
    groups_raw = meta.loc[train_idx, "subject_id"].values
    groups = np.where(pd.isna(groups_raw),
                      np.arange(len(groups_raw)).astype(str), groups_raw)

    X_tr_all = X_full.loc[train_idx].values
    X_te_all = X_full.loc[test_idx].values

    selector = SelectFromModel(
        LogisticRegression(l1_ratio=0.5, solver="saga", C=0.1,
                           max_iter=3000, class_weight="balanced", random_state=42),
        threshold="mean",
    )
    selector.fit(X_tr_all, y_train)
    X_train = selector.transform(X_tr_all)
    X_test  = selector.transform(X_te_all)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    pos_w = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
    models = {
        "LASSO-LR": LogisticRegression(
            l1_ratio=1, solver="saga", C=0.05,
            class_weight="balanced", max_iter=2000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"),
        "SVM (RBF)": SVC(
            kernel="rbf", C=1.0, gamma="scale", random_state=42,
            class_weight="balanced", probability=True),
        "XGBoost": XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
            scale_pos_weight=pos_w, eval_metric="logloss",
            random_state=42, n_jobs=-1),
    }

    rows = []
    for name, clf in models.items():
        sc = name in ("SVM (RBF)", "LASSO-LR")
        clf.fit(X_train_sc if sc else X_train, y_train)
        y_pred = clf.predict(X_test_sc if sc else X_test)
        y_prob = (clf.predict_proba(X_test_sc if sc else X_test)[:, 1]
                  if hasattr(clf, "predict_proba")
                  else clf.decision_function(X_test_sc if sc else X_test))
        rows.append({
            "round": round_num, "pipeline": "abundance (Line1-style)",
            "classifier": name,
            "AUC":    round(roc_auc_score(y_test, y_prob), 4),
            "AUPRC":  round(average_precision_score(y_test, y_prob), 4),
            "F1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
            "MCC":    round(matthews_corrcoef(y_test, y_pred), 4),
        })
        print(f"  R{round_num} {name}: AUC={rows[-1]['AUC']:.3f}  "
              f"F1={rows[-1]['F1_macro']:.3f}  MCC={rows[-1]['MCC']:.3f}")
    return pd.DataFrame(rows)


def plot_final(summary):
    classifiers = summary["classifier"].unique()
    metrics = ["AUC", "AUPRC", "F1_macro", "MCC"]
    x = np.arange(len(classifiers))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    for ax, m in zip(axes, metrics):
        means = [summary[summary["classifier"] == c][m].mean() for c in classifiers]
        stds  = [summary[summary["classifier"] == c][m].std()  for c in classifiers]
        ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(classifiers, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1); ax.set_title(m); ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    fig.suptitle(f"Obesity – Abundance Pipeline ({N_ROUNDS}-round mean ± SD)", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"abundance_{N_ROUNDS}round_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    if not os.path.exists(ABUNDANCE_CSV):
        print(f"ERROR: {ABUNDANCE_CSV} not found. Run download_abundance.R first.")
        sys.exit(1)

    print("Loading abundance matrix...")
    abund_full = pd.read_csv(ABUNDANCE_CSV, index_col="sample_id")
    print(f"  {abund_full.shape[0]} samples × {abund_full.shape[1]} species")

    all_rows = []
    for r in range(1, N_ROUNDS + 1):
        meta_path = os.path.join(SPLITS_DIR, f"obesity_metadata_round{r}.csv")
        if not os.path.exists(meta_path):
            print(f"Missing {meta_path} – run metadata_prep.py first or check splits/obesity/")
            continue
        meta = pd.read_csv(meta_path)
        print(f"\n── Round {r} ──────────────")
        rows = run_one_round(meta, abund_full, r)
        rows.to_csv(os.path.join(RESULTS_DIR, f"abundance_round{r}.csv"), index=False)
        all_rows.append(rows)

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(os.path.join(RESULTS_DIR, "abundance_all_rounds.csv"), index=False)

    # Summary: mean ± SD per classifier
    summary = (
        all_df.groupby("classifier")[["AUC", "AUPRC", "F1_macro", "MCC"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(RESULTS_DIR, "results_abundance.csv"), index=False)

    print("\n" + "=" * 60)
    print(f"Abundance pipeline – {N_ROUNDS}-round summary")
    print("=" * 60)
    for _, row in summary.iterrows():
        print(f"  {row['classifier']:15s}  "
              f"AUC={row['AUC_mean']:.3f}±{row['AUC_std']:.3f}  "
              f"AUPRC={row['AUPRC_mean']:.3f}±{row['AUPRC_std']:.3f}  "
              f"F1={row['F1_macro_mean']:.3f}±{row['F1_macro_std']:.3f}  "
              f"MCC={row['MCC_mean']:.3f}±{row['MCC_std']:.3f}")

    plot_final(all_df)
    print(f"\nFigure → {FIGURES_DIR}/abundance_{N_ROUNDS}round_summary.png")


if __name__ == "__main__":
    main()
