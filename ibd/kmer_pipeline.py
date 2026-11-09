"""
IBD K-mer pipeline: 10 repeated random splits (k=8).
Uses pre-computed kmer_matrix_k8_*.npy — no re-download needed.
Output: results/ibd_kmer_repeated.csv
"""

import os, warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from xgboost import XGBClassifier

ROOT       = os.path.dirname(os.path.abspath(__file__))
MATRIX_DIR = os.path.join(ROOT, "kmer_counts")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

K       = 8
TOP_N   = 1000
N_ROUNDS = 10
SEEDS   = [42 + i * 100 for i in range(N_ROUNDS)]


def ttest_select(X, y, top_n):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pvals = stats.ttest_ind(X[y == 0], X[y == 1], equal_var=False, axis=0)
    pvals = np.nan_to_num(pvals, nan=1.0)
    _, pvals_bh, _, _ = multipletests(pvals, method="fdr_bh")
    return np.argsort(pvals_bh)[:top_n]


def main():
    prefix = os.path.join(MATRIX_DIR, f"kmer_matrix_k{K}")
    X            = np.load(f"{prefix}_X.npy")
    labels       = np.load(f"{prefix}_labels.npy", allow_pickle=True)
    participants = np.load(f"{prefix}_participant.npy", allow_pickle=True)
    print(f"Loaded k={K} matrix: {X.shape}, classes={np.unique(labels)}")

    y = (labels == "IBD").astype(int)

    all_rows = []
    for r, seed in enumerate(SEEDS, 1):
        print(f"\n── Round {r} (seed={seed}) ──────────────")
        unique_parts = pd.Series(participants).drop_duplicates().values
        part_label   = pd.Series(y, index=participants).groupby(level=0).first()
        parts_train, parts_test = train_test_split(
            unique_parts, test_size=0.2, random_state=seed,
            stratify=part_label.loc[unique_parts].values,
        )
        train_mask = np.isin(participants, parts_train)
        test_mask  = np.isin(participants, parts_test)

        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]

        top_idx = ttest_select(X_tr, y_tr, TOP_N)
        X_tr_sel, X_te_sel = X_tr[:, top_idx], X_te[:, top_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_sel)
        X_te_sc = scaler.transform(X_te_sel)

        pos_w = int((y_tr == 0).sum()) / max(int((y_tr == 1).sum()), 1)
        models = {
            "LASSO-LR": LogisticRegression(
                l1_ratio=1.0, solver="saga", C=0.05,
                class_weight="balanced", max_iter=5000),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, random_state=seed, n_jobs=-1, class_weight="balanced"),
            "SVM (RBF)": SVC(
                kernel="rbf", C=1.0, gamma="scale", random_state=seed,
                class_weight="balanced", probability=True),
            "XGBoost": XGBClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
                scale_pos_weight=pos_w, eval_metric="logloss",
                random_state=seed, n_jobs=-1),
        }
        scaled = {"SVM (RBF)", "LASSO-LR"}
        for name, clf in models.items():
            Xtr = X_tr_sc if name in scaled else X_tr_sel
            Xte = X_te_sc if name in scaled else X_te_sel
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(Xtr, y_tr)
            y_pred = clf.predict(Xte)
            y_prob = clf.predict_proba(Xte)[:, 1]
            auc   = roc_auc_score(y_te, y_prob)
            auprc = average_precision_score(y_te, y_prob)
            f1    = f1_score(y_te, y_pred, average="macro")
            mcc   = matthews_corrcoef(y_te, y_pred)
            print(f"  R{r} {name}: AUC={auc:.3f}  F1={f1:.3f}  MCC={mcc:.3f}")
            all_rows.append({"round": r, "seed": seed, "pipeline": "K-mer (Line2)",
                             "classifier": name, "AUC": round(auc,4),
                             "AUPRC": round(auprc,4), "F1_macro": round(f1,4),
                             "MCC": round(mcc,4)})

    df = pd.DataFrame(all_rows)
    out = os.path.join(RESULTS_DIR, "ibd_kmer_repeated.csv")
    df.to_csv(out, index=False)

    print(f"\n{'='*60}")
    print(f"IBD K-mer – {N_ROUNDS}-round repeated split summary")
    print(f"{'='*60}")
    for clf, grp in df.groupby("classifier"):
        print(f"  {clf:15s} AUC={grp['AUC'].mean():.3f}±{grp['AUC'].std():.3f}  "
              f"AUPRC={grp['AUPRC'].mean():.3f}±{grp['AUPRC'].std():.3f}  "
              f"F1={grp['F1_macro'].mean():.3f}±{grp['F1_macro'].std():.3f}  "
              f"MCC={grp['MCC'].mean():.3f}±{grp['MCC'].std():.3f}")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
