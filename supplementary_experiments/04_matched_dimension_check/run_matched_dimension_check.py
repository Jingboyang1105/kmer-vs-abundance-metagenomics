#!/usr/bin/env python3
"""
IBD matched-dimension check: is the k-mer advantage a dimensionality effect?

k-mer models use up to 1,000 features while abundance
models use 251 (HMP2) or fewer. A fair comparison needs a matched-dimension
check -- e.g. compare top-250 k-mers to top-250 abundance features -- to rule
out that the "k-mer advantage" is simply because the k-mer model gets to pick
from a much larger feature pool.

This extends the existing unified-feature-selection sensitivity analysis
(ibd/feature_selection_sensitivity.py, Table S10) which already applies the
same Welch t-test + BH-ranking procedure to both modalities, but did not fix
both modalities to the *same* number of selected dimensions. Here both
modalities are selected with the identical univariate procedure AND capped at
the identical dimension count, swept over several matched dimensions
(50, 100, 250, 500), so the comparison isolates representation quality from both
feature-selection method and feature-space size, on samples shared by both
modalities.

Uses all 4 classifiers from the main IBD pipelines (Table 2), not just the
2 in the original sensitivity script, for a like-for-like comparison against
the abstract's headline result.

Outputs:
  results/ibd_matched_dimension_all_rounds.csv
  results/ibd_matched_dimension_summary.csv
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

# REPO_ROOT: supplementary_experiments/04_matched_dimension_check -> up two levels -> repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_ROOT = os.environ.get("MBEC_DATA_ROOT", REPO_ROOT)

META_CSV = os.path.join(DATA_ROOT, "ibd", "data", "raw", "hmp2_metadata_2018-08-20.csv")
TAXON_GZ = os.path.join(DATA_ROOT, "ibd", "data", "raw", "taxonomic_profiles_3.tsv.gz")
KMER_DIR = os.path.join(DATA_ROOT, "ibd", "kmer_matrix", "k8")

SEEDS = [42 + i * 100 for i in range(10)]
MATCHED_DIMS = [50, 100, 250, 500]


def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(META_CSV, low_memory=False)
    meta = meta[meta["data_type"] == "metagenomics"].copy()
    meta = meta[meta["diagnosis"].isin(["CD", "UC", "nonIBD"])].copy()
    meta["diagnosis"] = meta["diagnosis"].replace({"CD": "IBD", "UC": "IBD"})
    meta = meta[["External ID", "Participant ID", "diagnosis"]].drop_duplicates("External ID")
    meta.rename(columns={"External ID": "sample_id", "Participant ID": "participant"}, inplace=True)
    meta["file_id"] = meta["sample_id"].str.replace(r"_P$", "", regex=True)
    meta["label"] = (meta["diagnosis"] == "IBD").astype(int)
    return meta


def load_abundance(meta: pd.DataFrame) -> pd.DataFrame:
    taxon = pd.read_csv(TAXON_GZ, sep="\t", index_col=0)
    species_mask = taxon.index.map(lambda x: x.count("|") == 6 and "s__" in x and "t__" not in x)
    species = taxon[species_mask].copy()
    species.index = species.index.map(lambda x: x.split("|")[-1])
    species = species.T
    species.index = species.index.str.replace("_profile$", "", regex=True)
    species.index.name = "sample_id"
    species = species.astype(float)

    merged = meta.set_index("sample_id").join(species, how="inner")
    feat = merged.drop(columns=["participant", "diagnosis", "file_id", "label"])
    prev = (feat > 0).mean()
    feat = feat.loc[:, prev >= 0.10]
    return feat


def build_abundance_features(feat_df: pd.DataFrame) -> np.ndarray:
    x = feat_df.values + 1e-6
    log_x = np.log(x)
    clr = log_x - log_x.mean(axis=1, keepdims=True)
    p = feat_df.values / (feat_df.values.sum(axis=1, keepdims=True) + 1e-12)
    shannon = -np.sum(p * np.log(np.where(p > 0, p, 1)), axis=1)
    simpson = 1 - np.sum(p ** 2, axis=1)
    richness = (feat_df.values > 0).sum(axis=1).astype(float)
    pw = clr * (feat_df.values > 0).astype(float)
    return np.hstack([clr, shannon[:, None], simpson[:, None], richness[:, None], pw])


def load_kmer_matrix(meta: pd.DataFrame) -> pd.DataFrame:
    files = {f[:-len("_k8.npy")]: os.path.join(KMER_DIR, f)
             for f in os.listdir(KMER_DIR) if f.endswith("_k8.npy")}
    matched = meta[meta["file_id"].isin(files)].drop_duplicates("file_id")
    vecs = []
    for fid in matched["file_id"]:
        v = np.load(files[fid]).astype(np.float32)
        total = v.sum()
        vecs.append(v / total if total > 0 else v)
    return pd.DataFrame(np.vstack(vecs), index=matched["file_id"].values)


def select_univariate(x_train, y_train, x_test, top_n):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pvals = stats.ttest_ind(x_train[y_train == 0], x_train[y_train == 1], equal_var=False, axis=0)
    pvals = np.nan_to_num(pvals, nan=1.0, posinf=1.0, neginf=1.0)
    _, adj, _, _ = multipletests(pvals, method="fdr_bh")
    idx = np.argsort(adj)[: min(top_n, x_train.shape[1])]
    return x_train[:, idx], x_test[:, idx]


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


def main():
    missing = [p for p in (META_CSV, TAXON_GZ, KMER_DIR) if not os.path.exists(p)]
    if missing:
        print("Missing required local data, cannot run IBD matched-dimension check:")
        for p in missing:
            print(f"  - {p}")
        print(
            "Run ibd/extract_kmers.py + ibd/build_kmer_matrix.py and place the HMP2 "
            "metadata/taxonomic-profile files under ibd/data/raw/ first (see README "
            "'Data' section). Set MBEC_DATA_ROOT if these live outside the repo."
        )
        raise SystemExit(1)

    meta = load_metadata()
    ab_raw = load_abundance(meta)
    ab_features = pd.DataFrame(build_abundance_features(ab_raw), index=ab_raw.index)
    km_features = load_kmer_matrix(meta)

    # matched samples = present in both, with the same label
    shared = sorted(set(ab_features.index).intersection(km_features.index))
    ab_features = ab_features.loc[shared]
    km_features = km_features.loc[shared]
    labels = meta.drop_duplicates("file_id").set_index("file_id").loc[shared, "label"]
    participants = meta.drop_duplicates("file_id").set_index("file_id").loc[shared, "participant"]
    print(f"Matched samples: {len(shared)}  IBD={labels.sum()}  nonIBD={(labels==0).sum()}")

    y = labels.values
    parts = participants.values
    unique_parts = pd.Series(parts).drop_duplicates().values
    part_label = pd.Series(y, index=parts).groupby(level=0).first()

    all_rows = []
    for round_num, seed in enumerate(SEEDS, 1):
        print(f"\n=== Round {round_num}/10 (seed={seed}) ===")
        parts_train, parts_test = train_test_split(
            unique_parts, test_size=0.2, random_state=seed,
            stratify=part_label.loc[unique_parts].values,
        )
        tr = np.isin(parts, parts_train)
        te = np.isin(parts, parts_test)
        y_tr, y_te = y[tr], y[te]

        for dim in MATCHED_DIMS:
            ab_tr, ab_te = select_univariate(ab_features.values[tr], y_tr, ab_features.values[te], dim)
            km_tr, km_te = select_univariate(km_features.values[tr], y_tr, km_features.values[te], dim)

            for modality, (x_tr, x_te) in [("Abundance", (ab_tr, ab_te)), ("K-mer", (km_tr, km_te))]:
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
                    print(f"  dim={dim:4d} {modality:9s} {name:15s} AUC={auc:.3f}")
                    all_rows.append({
                        "round": round_num, "seed": seed, "matched_dim": dim,
                        "modality": modality, "classifier": name,
                        "AUC": auc,
                        "AUPRC": round(float(average_precision_score(y_te, y_prob)), 4),
                        "F1_macro": round(float(f1_score(y_te, y_pred, average="macro")), 4),
                        "MCC": round(float(matthews_corrcoef(y_te, y_pred)), 4),
                    })

    df = pd.DataFrame(all_rows)
    out = os.path.join(RESULTS_DIR, "ibd_matched_dimension_all_rounds.csv")
    df.to_csv(out, index=False)

    summary = (
        df.groupby(["matched_dim", "modality", "classifier"])[["AUC", "AUPRC", "F1_macro", "MCC"]]
        .agg(["mean", "std"]).round(4)
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary_out = os.path.join(RESULTS_DIR, "ibd_matched_dimension_summary.csv")
    summary.to_csv(summary_out, index=False)

    gaps = []
    for dim in MATCHED_DIMS:
        for clf in df["classifier"].unique():
            a = summary[(summary.matched_dim == dim) & (summary.classifier == clf) & (summary.modality == "Abundance")]
            k = summary[(summary.matched_dim == dim) & (summary.classifier == clf) & (summary.modality == "K-mer")]
            if len(a) and len(k):
                gaps.append({
                    "matched_dim": dim, "classifier": clf,
                    "abundance_AUC": float(a["AUC_mean"].iloc[0]),
                    "kmer_AUC": float(k["AUC_mean"].iloc[0]),
                    "gap_kmer_minus_abundance": round(float(k["AUC_mean"].iloc[0] - a["AUC_mean"].iloc[0]), 4),
                })
    gap_df = pd.DataFrame(gaps)
    gap_out = os.path.join(RESULTS_DIR, "ibd_matched_dimension_gap.csv")
    gap_df.to_csv(gap_out, index=False)

    print(f"\n{'='*70}\nMatched-dimension gap (k-mer minus abundance AUC), by dim\n{'='*70}")
    print(gap_df.to_string(index=False))
    print(f"\nSaved -> {out}\nSaved -> {summary_out}\nSaved -> {gap_out}")


if __name__ == "__main__":
    main()
