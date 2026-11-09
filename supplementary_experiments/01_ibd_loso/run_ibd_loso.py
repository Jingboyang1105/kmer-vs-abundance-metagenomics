#!/usr/bin/env python3
"""
IBD leave-one-site-out (LOSiteO) validation.

Extends held-out-cohort robustness testing (leave-one-study-out, used for
obesity) to IBD. HMP2 has five clinical collection sites (Cedars-Sinai,
Cincinnati, Emory, MGH, MGH Pediatrics); each of the 130 participants belongs
to exactly one site, so a leave-one-site-out split is the direct IBD analogue
of the obesity leave-one-study-out design already in the manuscript
(obesity/loso_validation.py).

For each held-out site: train on the remaining four sites (participant-level,
no leakage), test on the held-out site. Repeated for Abundance, K-mer, and
Fusion pipelines using the exact feature construction / feature selection /
classifier hyperparameters as the main IBD pipelines (ibd/abundance_pipeline.py,
ibd/kmer_pipeline.py, ibd/fusion_pipeline.py), so results are directly
comparable to Table 2 in the manuscript.

Outputs:
  results/ibd_loso_per_site.csv      one row per (site, pipeline, classifier)
  results/ibd_loso_summary.csv       mean +/- SD across the 5 site-folds
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# REPO_ROOT: supplementary_experiments/01_ibd_loso -> up two levels -> repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_ROOT = os.environ.get("MBEC_DATA_ROOT", REPO_ROOT)

# Same source data as the main IBD pipelines (ibd/abundance_pipeline.py,
# ibd/extract_kmers.py + ibd/build_kmer_matrix.py); see README "Data" section.
META_CSV = os.path.join(DATA_ROOT, "ibd", "data", "raw", "hmp2_metadata_2018-08-20.csv")
TAXON_GZ = os.path.join(DATA_ROOT, "ibd", "data", "raw", "taxonomic_profiles_3.tsv.gz")
KMER_DIR = os.path.join(DATA_ROOT, "ibd", "kmer_matrix", "k8")

TOP_KMER = 1000
TOP_KMER_FUSION = 500
MIN_TEST_PER_CLASS = 3
SEED_BASE = 2000


# ─────────────────────────── data loading ────────────────────────────────

def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(META_CSV, low_memory=False)
    meta = meta[meta["data_type"] == "metagenomics"].copy()
    meta = meta[meta["diagnosis"].isin(["CD", "UC", "nonIBD"])].copy()
    meta["diagnosis"] = meta["diagnosis"].replace({"CD": "IBD", "UC": "IBD"})
    meta = meta[["External ID", "Participant ID", "site_name", "diagnosis"]].drop_duplicates("External ID")
    meta.rename(columns={"External ID": "sample_id", "Participant ID": "participant"}, inplace=True)
    meta["file_id"] = meta["sample_id"].str.replace(r"_P$", "", regex=True)
    meta["label"] = (meta["diagnosis"] == "IBD").astype(int)
    return meta


def load_abundance_features(meta: pd.DataFrame) -> pd.DataFrame:
    taxon = pd.read_csv(TAXON_GZ, sep="\t", index_col=0)
    species_mask = taxon.index.map(lambda x: x.count("|") == 6 and "s__" in x and "t__" not in x)
    species = taxon[species_mask].copy()
    species.index = species.index.map(lambda x: x.split("|")[-1])
    species = species.T
    species.index = species.index.str.replace("_profile$", "", regex=True)
    species.index.name = "sample_id"
    species = species.astype(float)

    keep = meta.set_index("sample_id")
    merged = keep.join(species, how="inner")
    feat = merged.drop(columns=["participant", "site_name", "diagnosis", "file_id", "label"])
    prev = (feat > 0).mean()
    feat = feat.loc[:, prev >= 0.10]
    print(f"[abundance] {feat.shape[0]} samples x {feat.shape[1]} species (prevalence-filtered)")
    return feat  # index = sample_id (with _P suffix where present)


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
    print(f"[k-mer] {len(matched)} / {len(meta)} samples have cached k=8 vectors")
    vecs = []
    for fid in matched["file_id"]:
        v = np.load(files[fid]).astype(np.float32)
        total = v.sum()
        vecs.append(v / total if total > 0 else v)
    X = np.vstack(vecs)
    return pd.DataFrame(X, index=matched["file_id"].values)


# ─────────────────────────── feature selection ────────────────────────────

def select_abundance(x_train, y_train, x_test, seed):
    selector = SelectFromModel(
        LogisticRegression(l1_ratio=0.5, solver="saga", C=0.1, max_iter=3000,
                            class_weight="balanced", random_state=seed),
        threshold="mean",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector.fit(x_train, y_train)
    return selector.transform(x_train), selector.transform(x_test), int(selector.get_support().sum())


def select_kmer(x_train, y_train, x_test, top_n):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pvals = ttest_ind(x_train[y_train == 0], x_train[y_train == 1], equal_var=False, axis=0)
    pvals = np.nan_to_num(pvals, nan=1.0)
    _, adj, _, _ = multipletests(pvals, method="fdr_bh")
    idx = np.argsort(adj)[:top_n]
    return x_train[:, idx], x_test[:, idx], len(idx)


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


def fit_score(x_train, x_test, y_train, y_test, seed):
    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train)
    x_test_sc = scaler.transform(x_test)
    rows = []
    for name, clf in make_models(seed, y_train).items():
        xtr = x_train_sc if name in SCALED else x_train
        xte = x_test_sc if name in SCALED else x_test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(xtr, y_train)
        y_pred = clf.predict(xte)
        y_prob = clf.predict_proba(xte)[:, 1]
        rows.append({
            "classifier": name,
            "AUC": round(float(roc_auc_score(y_test, y_prob)), 4),
            "AUPRC": round(float(average_precision_score(y_test, y_prob)), 4),
            "F1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
            "MCC": round(float(matthews_corrcoef(y_test, y_pred)), 4),
        })
    return rows


# ─────────────────────────── LOSiteO folds ────────────────────────────────

def run_fold(site, meta, ab_features, km_features, pipeline, seed):
    if pipeline == "Abundance":
        ids = ab_features.index
    elif pipeline == "K-mer":
        ids = km_features.index
    else:
        ids = ab_features.index.intersection(km_features.index)

    m = meta[meta["file_id"].isin(ids) | meta["sample_id"].isin(ids)].drop_duplicates("file_id")
    # abundance is keyed by sample_id (may carry _P suffix), k-mer by file_id
    id_key = "sample_id" if pipeline == "Abundance" else "file_id"
    m = meta[meta[id_key].isin(ids)].drop_duplicates(id_key).set_index(id_key)

    test_idx = m.index[m["site_name"] == site]
    train_idx = m.index[m["site_name"] != site]
    if m.loc[test_idx, "label"].nunique() < 2 or min(
        (m.loc[test_idx, "label"] == 0).sum(), (m.loc[test_idx, "label"] == 1).sum()
    ) < MIN_TEST_PER_CLASS:
        return None

    y_train = m.loc[train_idx, "label"].values
    y_test = m.loc[test_idx, "label"].values

    if pipeline == "Abundance":
        x_train_all = ab_features.loc[train_idx].values.astype(np.float32)
        x_test_all = ab_features.loc[test_idx].values.astype(np.float32)
        x_train, x_test, n_feat = select_abundance(x_train_all, y_train, x_test_all, seed)
    elif pipeline == "K-mer":
        x_train_all = km_features.loc[train_idx].values.astype(np.float32)
        x_test_all = km_features.loc[test_idx].values.astype(np.float32)
        x_train, x_test, n_feat = select_kmer(x_train_all, y_train, x_test_all, TOP_KMER)
    else:  # Fusion
        x_ab_train = ab_features.loc[train_idx].values.astype(np.float32)
        x_ab_test = ab_features.loc[test_idx].values.astype(np.float32)
        x_km_train = km_features.loc[train_idx].values.astype(np.float32)
        x_km_test = km_features.loc[test_idx].values.astype(np.float32)
        x_ab_train_s, x_ab_test_s, n_ab = select_abundance(x_ab_train, y_train, x_ab_test, seed)
        x_km_train_s, x_km_test_s, n_km = select_kmer(x_km_train, y_train, x_km_test, TOP_KMER_FUSION)
        x_train = np.hstack([x_ab_train_s, x_km_train_s])
        x_test = np.hstack([x_ab_test_s, x_km_test_s])
        n_feat = n_ab + n_km

    rows = fit_score(x_train, x_test, y_train, y_test, seed)
    for row in rows:
        row.update({
            "held_out_site": site,
            "pipeline": pipeline,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "test_ibd": int(y_test.sum()),
            "test_nonibd": int((y_test == 0).sum()),
            "n_features": n_feat,
        })
    return rows


def main():
    missing = [p for p in (META_CSV, TAXON_GZ, KMER_DIR) if not os.path.exists(p)]
    if missing:
        print("Missing required local data, cannot run IBD leave-one-site-out validation:")
        for p in missing:
            print(f"  - {p}")
        print(
            "Run ibd/extract_kmers.py + ibd/build_kmer_matrix.py and place the HMP2 "
            "metadata/taxonomic-profile files under ibd/data/raw/ first (see README "
            "'Data' section). Set MBEC_DATA_ROOT if these live outside the repo."
        )
        raise SystemExit(1)

    meta = load_metadata()
    ab_raw = load_abundance_features(meta)
    ab_features = pd.DataFrame(build_abundance_features(ab_raw), index=ab_raw.index)
    km_features = load_kmer_matrix(meta)

    sites = meta["site_name"].value_counts().index.tolist()
    print(f"Sites: {sites}")

    all_rows = []
    for fold, site in enumerate(sites, 1):
        seed = SEED_BASE + fold
        print(f"\n[{fold}/{len(sites)}] Holding out site = {site} (seed={seed})")
        for pipeline in ["Abundance", "K-mer", "Fusion"]:
            rows = run_fold(site, meta, ab_features, km_features, pipeline, seed)
            if rows is None:
                print(f"  {pipeline}: skipped (insufficient test-class balance)")
                continue
            for r in rows:
                print(f"  {pipeline:10s} {r['classifier']:15s} AUC={r['AUC']:.3f} "
                      f"F1={r['F1_macro']:.3f} MCC={r['MCC']:.3f} (n_test={r['n_test']})")
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out = os.path.join(RESULTS_DIR, "ibd_loso_per_site.csv")
    df.to_csv(out, index=False)

    summary = (
        df.groupby(["pipeline", "classifier"])[["AUC", "AUPRC", "F1_macro", "MCC"]]
        .agg(["mean", "std", "count"]).round(4)
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary_out = os.path.join(RESULTS_DIR, "ibd_loso_summary.csv")
    summary.to_csv(summary_out, index=False)

    print(f"\n{'='*70}\nIBD leave-one-site-out summary (5 held-out sites)\n{'='*70}")
    print(summary.to_string(index=False))
    print(f"\nSaved -> {out}\nSaved -> {summary_out}")


if __name__ == "__main__":
    main()
