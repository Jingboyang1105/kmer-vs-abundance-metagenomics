"""
IBD Fusion pipeline: Abundance + K-mer combined (10-round repeated split).
Abundance: HMP2 MetaPhlAn species profiles  (ibd/data/raw/)
K-mer:     canonical k=8 frequency counts   (ibd/kmer_counts/)
Output:    ibd/results/fusion_ibd_all_rounds.csv
"""

import os, warnings
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from xgboost import XGBClassifier

ROOT        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_ROUNDS  = 10
SEEDS     = [42 + i * 100 for i in range(N_ROUNDS)]
TOP_KMER  = 500


def load_abundance():
    data_dir   = os.path.join(ROOT, "data", "raw")
    meta_path  = os.path.join(data_dir, "hmp2_metadata_2018-08-20.csv")
    taxon_path = os.path.join(data_dir, "taxonomic_profiles_3.tsv.gz")

    meta = pd.read_csv(meta_path, low_memory=False)
    meta = meta[meta["data_type"] == "metagenomics"]
    meta = meta[meta["diagnosis"].isin(["CD", "UC", "nonIBD"])]
    meta["diagnosis"] = meta["diagnosis"].replace({"CD": "IBD", "UC": "IBD"})
    meta = (meta[["External ID", "Participant ID", "diagnosis"]]
            .drop_duplicates("External ID")
            .rename(columns={"External ID": "sample_id"})
            .set_index("sample_id"))

    print("  Loading taxonomic profiles…")
    taxon = pd.read_csv(taxon_path, sep="\t", index_col=0)
    sp_mask = taxon.index.map(
        lambda x: x.count("|") == 6 and "s__" in x and "t__" not in x)
    species = taxon[sp_mask].copy()
    species.index = species.index.map(lambda x: x.split("|")[-1])
    species = species.T
    species.index = species.index.str.replace("_profile$", "", regex=True)
    species.index.name = "sample_id"
    species = species.astype(float)

    merged = meta.join(species, how="inner")
    feat   = merged.drop(columns=["Participant ID", "diagnosis"])
    prev   = (feat > 0).mean()
    feat   = feat.loc[:, prev >= 0.10]
    labels = (merged["diagnosis"] == "IBD").astype(int).values
    parts  = merged["Participant ID"].values
    return feat, labels, parts, merged.index.values


def build_abundance_features(feat_df):
    X = feat_df.values + 1e-6
    log_X = np.log(X)
    clr   = log_X - log_X.mean(axis=1, keepdims=True)
    p     = feat_df.values / (feat_df.values.sum(axis=1, keepdims=True) + 1e-12)
    shannon  = -np.sum(p * np.log(np.where(p > 0, p, 1)), axis=1)
    simpson  = 1 - np.sum(p ** 2, axis=1)
    richness = (feat_df.values > 0).sum(axis=1).astype(float)
    pw = clr * (feat_df.values > 0).astype(float)
    return np.hstack([clr, shannon[:, None], simpson[:, None], richness[:, None], pw])


def ttest_select(X_tr, y_tr, top_n):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pvals = ttest_ind(X_tr[y_tr == 0], X_tr[y_tr == 1], equal_var=False, axis=0)
    pvals = np.nan_to_num(pvals, nan=1.0)
    _, pvals_bh, _, _ = multipletests(pvals, method="fdr_bh")
    return np.argsort(pvals_bh)[:top_n]


def main():
    print("Loading abundance data…")
    feat_df, y_ab, participants, sample_ids_ab = load_abundance()
    X_ab_all = build_abundance_features(feat_df)

    print("Loading k-mer matrix…")
    kmer_dir  = os.path.join(ROOT, "kmer_counts")
    X_km_raw  = np.load(os.path.join(kmer_dir, "kmer_matrix_k8_X.npy"))
    ids_km    = np.load(os.path.join(kmer_dir, "kmer_matrix_k8_sample_ids.npy"), allow_pickle=True)
    labels_km = np.load(os.path.join(kmer_dir, "kmer_matrix_k8_labels.npy"), allow_pickle=True)
    parts_km  = np.load(os.path.join(kmer_dir, "kmer_matrix_k8_participant.npy"), allow_pickle=True)

    # deduplicate
    _, uniq = np.unique(ids_km, return_index=True)
    X_km_raw  = X_km_raw[uniq];  ids_km = ids_km[uniq]
    labels_km = labels_km[uniq]; parts_km = parts_km[uniq]
    row_sums  = X_km_raw.sum(axis=1, keepdims=True)
    X_km_freq = X_km_raw / np.where(row_sums == 0, 1, row_sums)
    y_km      = (labels_km == "IBD").astype(int)

    # intersect samples
    shared = np.intersect1d(sample_ids_ab, ids_km)
    ab_pos = [np.where(sample_ids_ab == s)[0][0] for s in shared]
    km_pos = [np.where(ids_km == s)[0][0] for s in shared]

    X_ab   = X_ab_all[ab_pos]
    X_km   = X_km_freq[km_pos]
    y      = y_ab[ab_pos]
    parts  = participants[ab_pos]
    print(f"  Shared samples: {len(shared)}, IBD={y.sum()}, nonIBD={(y==0).sum()}")

    all_rows = []
    for r, seed in enumerate(SEEDS, 1):
        print(f"\n── Round {r} (seed={seed}) ──")
        unique_parts = pd.Series(parts).drop_duplicates().values
        part_label   = pd.Series(y, index=parts).groupby(level=0).first()
        parts_train, parts_test = train_test_split(
            unique_parts, test_size=0.2, random_state=seed,
            stratify=part_label.loc[unique_parts].values)
        tr = np.isin(parts, parts_train)
        te = np.isin(parts, parts_test)

        X_ab_tr, X_ab_te = X_ab[tr], X_ab[te]
        X_km_tr, X_km_te = X_km[tr], X_km[te]
        y_tr, y_te       = y[tr], y[te]

        # Abundance: ElasticNet selection
        selector = SelectFromModel(
            LogisticRegression(l1_ratio=0.5, solver="saga", C=0.1,
                               max_iter=3000, class_weight="balanced", random_state=seed),
            threshold="mean")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selector.fit(X_ab_tr, y_tr)
        X_ab_tr_sel = selector.transform(X_ab_tr)
        X_ab_te_sel = selector.transform(X_ab_te)

        # K-mer: t-test top-500
        km_idx      = ttest_select(X_km_tr, y_tr, TOP_KMER)
        X_km_tr_sel = X_km_tr[:, km_idx]
        X_km_te_sel = X_km_te[:, km_idx]

        X_fused_tr = np.hstack([X_ab_tr_sel, X_km_tr_sel])
        X_fused_te = np.hstack([X_ab_te_sel, X_km_te_sel])

        n_ab = X_ab_tr_sel.shape[1]; n_km = X_km_tr_sel.shape[1]
        print(f"  {n_ab} abundance + {n_km} k-mer = {n_ab+n_km} features | "
              f"train={y_tr.sum()}/{(y_tr==0).sum()} test={y_te.sum()}/{(y_te==0).sum()}")

        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_fused_tr)
        X_te_sc  = scaler.transform(X_fused_te)

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
            Xtr = X_tr_sc if name in scaled else X_fused_tr
            Xte = X_te_sc if name in scaled else X_fused_te
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
                             "pipeline": "Fusion (Abundance+K-mer)",
                             "classifier": name,
                             "AUC": round(auc, 4), "AUPRC": round(auprc, 4),
                             "F1_macro": round(f1, 4), "MCC": round(mcc, 4)})

    df  = pd.DataFrame(all_rows)
    out = os.path.join(RESULTS_DIR, "fusion_ibd_all_rounds.csv")
    df.to_csv(out, index=False)

    print(f"\n{'='*60}\nIBD Fusion – {N_ROUNDS}-round summary\n{'='*60}")
    for clf, grp in df.groupby("classifier"):
        print(f"  {clf:15s} AUC={grp['AUC'].mean():.3f}±{grp['AUC'].std():.3f}  "
              f"F1={grp['F1_macro'].mean():.3f}±{grp['F1_macro'].std():.3f}  "
              f"MCC={grp['MCC'].mean():.3f}±{grp['MCC'].std():.3f}")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
