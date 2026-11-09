"""
IBD Abundance pipeline: 10 repeated random splits.
Uses HMP2 metadata + MetaPhlAn species profiles.
Output: results/ibd_abundance_repeated.csv
"""

import os, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from xgboost import XGBClassifier

ROOT        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(ROOT, "data", "raw")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

METADATA_CSV = os.path.join(DATA_DIR, "hmp2_metadata_2018-08-20.csv")
TAXON_GZ     = os.path.join(DATA_DIR, "taxonomic_profiles_3.tsv.gz")

N_ROUNDS = 10
SEEDS    = [42 + i * 100 for i in range(N_ROUNDS)]


def load_data():
    meta = pd.read_csv(METADATA_CSV, low_memory=False)
    meta = meta[meta["data_type"] == "metagenomics"].copy()
    meta = meta[meta["diagnosis"].isin(["CD", "UC", "nonIBD"])].copy()
    meta["diagnosis"] = meta["diagnosis"].replace({"CD": "IBD", "UC": "IBD"})
    meta = meta[["External ID", "Participant ID", "diagnosis"]].drop_duplicates("External ID")
    meta.rename(columns={"External ID": "sample_id"}, inplace=True)

    taxon = pd.read_csv(TAXON_GZ, sep="\t", index_col=0)
    species_mask = taxon.index.map(
        lambda x: x.count("|") == 6 and "s__" in x and "t__" not in x
    )
    species = taxon[species_mask].copy()
    species.index = species.index.map(lambda x: x.split("|")[-1])
    species = species.T
    species.index = species.index.str.replace("_profile$", "", regex=True)
    species.index.name = "sample_id"

    merged = meta.set_index("sample_id").join(species, how="inner")
    labels       = merged["diagnosis"].values
    participants = merged["Participant ID"].values
    feat_df      = merged.drop(columns=["Participant ID", "diagnosis"]).astype(float)

    prev = (feat_df > 0).mean()
    feat_df = feat_df.loc[:, prev >= 0.10]
    print(f"Loaded: {feat_df.shape[0]} samples × {feat_df.shape[1]} species, "
          f"classes={np.unique(labels)}")
    return feat_df, labels, participants


def build_features(feat_df):
    X_vals = feat_df.values + 1e-6
    log_X  = np.log(X_vals)
    X_clr  = log_X - log_X.mean(axis=1, keepdims=True)

    p = feat_df.values / (feat_df.values.sum(axis=1, keepdims=True) + 1e-12)
    shannon  = -np.sum(p * np.log(np.where(p > 0, p, 1)), axis=1)
    simpson  = 1 - np.sum(p ** 2, axis=1)
    richness = (feat_df.values > 0).sum(axis=1).astype(float)

    X_pw = X_clr * (feat_df.values > 0).astype(float)

    return np.hstack([
        X_clr,
        shannon[:, None], simpson[:, None], richness[:, None],
        X_pw,
    ])


def main():
    feat_df, labels, participants = load_data()
    y = (labels == "IBD").astype(int)
    X_all = build_features(feat_df)

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

        X_tr, X_te = X_all[train_mask], X_all[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]

        selector = SelectFromModel(
            LogisticRegression(l1_ratio=0.5, solver="saga", C=0.1,
                               max_iter=3000, class_weight="balanced",
                               random_state=seed),
            threshold="mean",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selector.fit(X_tr, y_tr)
        X_tr_sel = selector.transform(X_tr)
        X_te_sel = selector.transform(X_te)

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
            all_rows.append({"round": r, "seed": seed,
                             "pipeline": "Abundance (Line1)",
                             "classifier": name,
                             "AUC":      round(auc,  4),
                             "AUPRC":    round(auprc,4),
                             "F1_macro": round(f1,   4),
                             "MCC":      round(mcc,  4)})

    df = pd.DataFrame(all_rows)
    out = os.path.join(RESULTS_DIR, "ibd_abundance_repeated.csv")
    df.to_csv(out, index=False)

    print(f"\n{'='*60}")
    print(f"IBD Abundance – {N_ROUNDS}-round repeated split summary")
    print(f"{'='*60}")
    for clf, grp in df.groupby("classifier"):
        print(f"  {clf:15s} AUC={grp['AUC'].mean():.3f}±{grp['AUC'].std():.3f}  "
              f"AUPRC={grp['AUPRC'].mean():.3f}±{grp['AUPRC'].std():.3f}  "
              f"F1={grp['F1_macro'].mean():.3f}±{grp['F1_macro'].std():.3f}  "
              f"MCC={grp['MCC'].mean():.3f}±{grp['MCC'].std():.3f}")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
