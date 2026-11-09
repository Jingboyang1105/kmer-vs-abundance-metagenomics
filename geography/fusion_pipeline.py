"""
Geography Fusion pipeline: Abundance + K-mer combined (10-round subsampling).
30-country multi-class classification.
Output: results/fusion_geo_all_rounds.csv
"""

import os, warnings
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, average_precision_score
from xgboost import XGBClassifier

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "../data")
SPLITS_DIR  = os.path.join(SCRIPT_DIR, "../splits/geography")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ABUNDANCE_CSV = os.path.join(DATA_DIR, "cmd3_geography_abundance.csv")
KMER_DIR      = os.path.join(SCRIPT_DIR, "../../obesity/data/kmer_counts_k8")
N_ROUNDS      = 10
TOP_KMER      = 500


def clr_transform(df):
    X = df.values + 1e-6
    log_X = np.log(X)
    return pd.DataFrame(log_X - log_X.mean(axis=1, keepdims=True),
                        index=df.index, columns=df.columns)


def build_abundance_features(abund):
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


def ttest_select_multiclass(X_tr, y_tr, top_n):
    """For multi-class: average p-values across one-vs-rest comparisons."""
    classes = np.unique(y_tr)
    all_pvals = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c in classes:
            binary = (y_tr == c).astype(int)
            if binary.sum() < 3 or (1 - binary).sum() < 3:
                continue
            _, pv = ttest_ind(X_tr[binary == 0], X_tr[binary == 1],
                              equal_var=False, axis=0)
            all_pvals.append(np.nan_to_num(pv, nan=1.0))
    mean_pvals = np.mean(all_pvals, axis=0)
    _, pvals_bh, _, _ = multipletests(mean_pvals, method="fdr_bh")
    return np.argsort(pvals_bh)[:top_n]


def run_one_round(meta, abund_full, kmer_mat, kmer_ids, round_num):
    shared_ab = meta["sample_id"][meta["sample_id"].isin(abund_full.index)]
    meta_ab   = meta[meta["sample_id"].isin(shared_ab)].set_index("sample_id")
    abund     = abund_full.loc[shared_ab]
    prev      = (abund > 0).mean()
    abund     = abund.loc[:, prev >= 0.10]
    X_ab_full = build_abundance_features(abund)

    shared_km = meta["sample_id"][meta["sample_id"].isin(kmer_ids)]
    shared    = meta_ab.index.intersection(pd.Index(shared_km))
    meta_s    = meta_ab.loc[shared]

    X_ab = X_ab_full.loc[shared].values
    km_pos = [list(kmer_ids).index(s) for s in shared]
    X_km   = kmer_mat[km_pos]

    train_idx = meta_s.index[meta_s["split"] == "train"]
    test_idx  = meta_s.index[meta_s["split"] == "test"]
    le = LabelEncoder()
    le.fit(meta_s.loc[train_idx, "label"].values)
    y_train = le.transform(meta_s.loc[train_idx, "label"].values)
    # test: only samples whose label was seen in training
    test_labels = meta_s.loc[test_idx, "label"].values
    seen_mask = np.isin(test_labels, le.classes_)
    test_idx  = test_idx[seen_mask]
    y_test    = le.transform(test_labels[seen_mask])

    shared_list = list(shared)
    tr_pos = [shared_list.index(s) for s in train_idx]
    te_pos = [shared_list.index(s) for s in test_idx]

    X_ab_tr, X_ab_te = X_ab[tr_pos], X_ab[te_pos]
    X_km_tr, X_km_te = X_km[tr_pos], X_km[te_pos]

    # Abundance: ElasticNet selection
    selector = SelectFromModel(
        LogisticRegression(l1_ratio=0.5, solver="saga", C=0.1,
                           max_iter=3000, class_weight="balanced", random_state=42),
        threshold="mean")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector.fit(X_ab_tr, y_train)
    X_ab_tr_sel = selector.transform(X_ab_tr)
    X_ab_te_sel = selector.transform(X_ab_te)

    # K-mer: multi-class t-test top-500
    km_idx      = ttest_select_multiclass(X_km_tr, y_train, TOP_KMER)
    X_km_tr_sel = X_km_tr[:, km_idx]
    X_km_te_sel = X_km_te[:, km_idx]

    X_fused_tr = np.hstack([X_ab_tr_sel, X_km_tr_sel])
    X_fused_te = np.hstack([X_ab_te_sel, X_km_te_sel])

    n_ab = X_ab_tr_sel.shape[1]; n_km = X_km_tr_sel.shape[1]
    print(f"  Round {round_num}: {n_ab} abundance + {n_km} k-mer = {n_ab+n_km} features | "
          f"train={len(y_train)} test={len(y_test)}")

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_fused_tr)
    X_te_sc = scaler.transform(X_fused_te)

    n_classes = len(le.classes_)
    models = {
        "LASSO-LR": LogisticRegression(
            l1_ratio=1, solver="saga", C=0.05,
            class_weight="balanced", max_iter=5000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"),
        "SVM (RBF)": SVC(
            kernel="rbf", C=1.0, gamma="scale", random_state=42,
            class_weight="balanced", probability=True),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
            num_class=n_classes, objective="multi:softprob",
            eval_metric="mlogloss", random_state=42, n_jobs=-1),
    }
    scaled = {"SVM (RBF)", "LASSO-LR"}
    rows = []
    for name, clf in models.items():
        Xtr = X_tr_sc if name in scaled else X_fused_tr
        Xte = X_te_sc if name in scaled else X_fused_te
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xte)
        y_prob = clf.predict_proba(Xte)
        present = np.unique(y_test)
        auc_vals, auprc_vals = [], []
        for cls in present:
            y_bin = (y_test == cls).astype(int)
            if cls >= y_prob.shape[1]:
                continue
            try:
                auc_vals.append(roc_auc_score(y_bin, y_prob[:, cls]))
            except Exception:
                pass
            try:
                auprc_vals.append(average_precision_score(y_bin, y_prob[:, cls]))
            except Exception:
                pass
        auc   = float(np.mean(auc_vals))   if auc_vals   else float("nan")
        auprc = float(np.mean(auprc_vals)) if auprc_vals else float("nan")
        f1    = f1_score(y_test, y_pred, average="macro")
        mcc   = matthews_corrcoef(y_test, y_pred)
        print(f"  R{round_num} {name}: AUC={auc:.3f}  F1={f1:.3f}  MCC={mcc:.3f}")
        rows.append({"round": round_num, "pipeline": "Fusion (Abundance+K-mer)",
                     "classifier": name,
                     "AUC": round(auc, 4), "AUPRC": round(auprc, 4),
                     "F1_macro": round(f1, 4), "MCC": round(mcc, 4)})
    return rows


def main():
    print("Loading abundance matrix…")
    abund_full = pd.read_csv(ABUNDANCE_CSV, index_col=0)
    print(f"  {abund_full.shape[0]} samples × {abund_full.shape[1]} species")

    print("Loading k-mer matrix…")
    npy_files = sorted([f for f in os.listdir(KMER_DIR) if f.endswith("_k8.npy")])
    kmer_ids  = [f.replace("_k8.npy", "") for f in npy_files]
    kmer_mat  = np.vstack([np.load(os.path.join(KMER_DIR, f)) for f in npy_files])
    row_sums  = kmer_mat.sum(axis=1, keepdims=True)
    kmer_mat  = kmer_mat / np.where(row_sums == 0, 1, row_sums)
    print(f"  {kmer_mat.shape[0]} samples × {kmer_mat.shape[1]} k-mers")

    # Fit label encoder on all labels across all rounds
    all_meta = pd.concat([
        pd.read_csv(os.path.join(SPLITS_DIR, f"geo_metadata_round{r}.csv"))
        for r in range(1, N_ROUNDS + 1)])
    le = LabelEncoder()
    le.fit(all_meta["label"].values)
    print(f"  {len(le.classes_)} classes: {le.classes_[:5]}…")

    all_rows = []
    for r in range(1, N_ROUNDS + 1):
        print(f"\n── Round {r} ──────────────")
        meta = pd.read_csv(os.path.join(SPLITS_DIR, f"geo_metadata_round{r}.csv"))
        rows = run_one_round(meta, abund_full, kmer_mat, kmer_ids, r)
        all_rows.extend(rows)
        pd.DataFrame(rows).to_csv(
            os.path.join(RESULTS_DIR, f"fusion_geo_round{r}.csv"), index=False)

    df  = pd.DataFrame(all_rows)
    out = os.path.join(RESULTS_DIR, "fusion_geo_all_rounds.csv")
    df.to_csv(out, index=False)

    print(f"\n{'='*60}\nGeography Fusion – {N_ROUNDS}-round summary\n{'='*60}")
    for clf, grp in df.groupby("classifier"):
        print(f"  {clf:15s} AUC={grp['AUC'].mean():.3f}±{grp['AUC'].std():.3f}  "
              f"F1={grp['F1_macro'].mean():.3f}±{grp['F1_macro'].std():.3f}  "
              f"MCC={grp['MCC'].mean():.3f}±{grp['MCC'].std():.3f}")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
