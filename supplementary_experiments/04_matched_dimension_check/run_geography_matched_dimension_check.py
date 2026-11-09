#!/usr/bin/env python3
"""
Geography matched-dimension check: same question as the IBD and obesity
matched-dimension checks, extended to the 30-country geographic-origin task.

The main geography comparison uses k-mer top-1,000 dimensions versus ~400
abundance-derived dimensions (CLR + alpha-diversity + presence-weighted CLR,
~200 species after >=10% prevalence filtering) -- is part of the headline
gap attributable to the larger k-mer feature space rather than
representation quality?

Scope: geography has no cached k-mer vectors for the full n=22,397 cohort.
This check is restricted to the same round-1 pilot pool used for the
geography k-mer study-held-out test (data/kmer_counts_pilot/, n=1,453 of
1,587 with usable k-mer vectors) -- reported as a bounded pilot on this pool,
not the full-cohort comparison. A random participant-level 80/20 split is
used (not study-held-out: this check isolates dimension, not cross-study
generalization, which is covered separately by the study-held-out tests).

Outputs:
  results/geography_matched_dimension_all_rounds.csv
  results/geography_matched_dimension_summary.csv
  results/geography_matched_dimension_gap.csv
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# REPO_ROOT: supplementary_experiments/04_matched_dimension_check -> up two levels -> repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_ROOT = os.environ.get("MBEC_DATA_ROOT", REPO_ROOT)

# Produced by geography/metadata_prep.py and geography/download_abundance.R;
# see README "Data" section.
META_CSV = os.path.join(DATA_ROOT, "geography", "data", "geo_metadata_round1.csv")
ABUNDANCE_CSV = os.path.join(DATA_ROOT, "geography", "data", "cmd3_geography_abundance.csv")
# Same pilot k-mer cache produced by ../02_geography_study_confound/extract_kmer_pilot.py.
KMER_DIR = os.path.join(
    DATA_ROOT, "supplementary_experiments", "02_geography_study_confound",
    "data", "kmer_counts_pilot",
)

SEEDS = [42 + i * 100 for i in range(10)]
MATCHED_DIMS = [50, 100, 250, 500]


def clr_transform(x):
    x = x + 1e-6
    log_x = np.log(x)
    return log_x - log_x.mean(axis=1, keepdims=True)


def build_abundance_features(abund: pd.DataFrame) -> np.ndarray:
    x = abund.values.astype(np.float32)
    clr = clr_transform(x)
    p = x / (x.sum(axis=1, keepdims=True) + 1e-12)
    shannon = -np.sum(p * np.log(np.where(p > 0, p, 1)), axis=1)
    simpson = 1 - np.sum(p ** 2, axis=1)
    richness = (x > 0).sum(axis=1).astype(np.float32)
    pw = clr * (x > 0).astype(np.float32)
    return np.hstack([clr, shannon[:, None], simpson[:, None], richness[:, None], pw])


def load_kmer_matrix(sample_ids) -> pd.DataFrame:
    files = {f[: -len("_k8.npy")]: os.path.join(KMER_DIR, f)
             for f in os.listdir(KMER_DIR) if f.endswith("_k8.npy")}
    ids = sorted(set(files).intersection(sample_ids))
    vecs = [np.load(files[i]).astype(np.float32) for i in ids]
    return pd.DataFrame(np.vstack(vecs), index=ids)


def select_univariate_multiclass(x_train, y_train, x_test, top_n):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        groups = [x_train[y_train == c] for c in np.unique(y_train)]
        _, pvals = f_oneway(*groups)
    pvals = np.nan_to_num(pvals, nan=1.0)
    _, adj, _, _ = multipletests(pvals, method="fdr_bh")
    idx = np.argsort(adj)[: min(top_n, x_train.shape[1])]
    return x_train[:, idx], x_test[:, idx]


def multiclass_metrics(y_true, y_pred, y_prob):
    present = np.unique(y_true)
    auc_vals, auprc_vals = [], []
    for cls in present:
        if cls >= y_prob.shape[1]:
            continue
        y_bin = (y_true == cls).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        try:
            auc_vals.append(roc_auc_score(y_bin, y_prob[:, cls]))
        except Exception:
            pass
        try:
            auprc_vals.append(average_precision_score(y_bin, y_prob[:, cls]))
        except Exception:
            pass
    auc = float(np.mean(auc_vals)) if auc_vals else float("nan")
    auprc = float(np.mean(auprc_vals)) if auprc_vals else float("nan")
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    return round(auc, 4), round(auprc, 4), round(float(f1), 4), round(float(mcc), 4)


def make_models(seed, n_classes):
    return {
        "LASSO-LR": LogisticRegression(l1_ratio=1, solver="saga", C=0.05,
                                        class_weight="balanced", max_iter=2000, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=seed,
                                                  n_jobs=4, class_weight="balanced"),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed,
                          class_weight="balanced", probability=True, decision_function_shape="ovr"),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
                                  num_class=n_classes, objective="multi:softprob",
                                  eval_metric="mlogloss", random_state=seed, n_jobs=4),
    }


SCALED = {"SVM (RBF)", "LASSO-LR"}


def main():
    missing = [p for p in (META_CSV, ABUNDANCE_CSV) if not os.path.exists(p)]
    if not os.path.isdir(KMER_DIR) or not os.listdir(KMER_DIR):
        missing.append(KMER_DIR)
    if missing:
        print("Missing required local data, cannot run geography matched-dimension check:")
        for p in missing:
            print(f"  - {p}")
        print(
            "Run geography/metadata_prep.py, geography/download_abundance.R, and "
            "../02_geography_study_confound/extract_kmer_pilot.py first. "
            "Set MBEC_DATA_ROOT if these live outside the repo."
        )
        raise SystemExit(1)

    print("Loading geography round-1 pool + k-mer pilot matrix...")
    meta = pd.read_csv(META_CSV)
    abund_full = pd.read_csv(ABUNDANCE_CSV, index_col="sample_id")
    abund_full = abund_full[~abund_full.index.duplicated(keep="first")]
    km_full = load_kmer_matrix(meta["sample_id"])
    print(f"  k-mer vectors available: {len(km_full)} / {len(meta)} round-1 samples")

    shared = sorted(set(abund_full.index).intersection(km_full.index).intersection(meta["sample_id"]))
    meta_shared = meta[meta["sample_id"].isin(shared)].drop_duplicates("sample_id").set_index("sample_id")
    ab_raw = abund_full.loc[meta_shared.index]
    prev = (ab_raw > 0).mean()
    ab_raw = ab_raw.loc[:, prev >= 0.10]
    ab_feat = pd.DataFrame(build_abundance_features(ab_raw), index=meta_shared.index)
    km_feat = km_full.loc[meta_shared.index]
    print(f"  Matched samples (abundance + k-mer both available): {len(meta_shared)}, "
          f"{meta_shared['country'].nunique()} countries")

    parts = meta_shared["subject_id"].fillna(pd.Series(meta_shared.index, index=meta_shared.index))
    country_by_part = pd.Series(meta_shared["country"].values, index=parts.values).groupby(level=0).first()
    up = parts.drop_duplicates().values
    # keep only countries with enough participants to stratify a split
    vc = country_by_part.loc[up].value_counts()
    keep_countries = vc[vc >= 5].index
    up = [u for u in up if country_by_part.loc[u] in keep_countries]

    all_rows = []
    for round_num, seed in enumerate(SEEDS, 1):
        print(f"\n=== Round {round_num}/10 (seed={seed}) ===")
        parts_train, parts_test = train_test_split(
            up, test_size=0.2, random_state=seed,
            stratify=[country_by_part.loc[u] for u in up],
        )
        tr = parts.isin(parts_train).values
        te = parts.isin(parts_test).values

        le = LabelEncoder()
        le.fit(meta_shared["country"].values)
        y_all = le.transform(meta_shared["country"].values)
        y_tr, y_te = y_all[tr], y_all[te]
        n_classes = len(le.classes_)
        print(f"  train={tr.sum()}  test={te.sum()}  classes={n_classes}")

        for dim in MATCHED_DIMS:
            ab_tr, ab_te = select_univariate_multiclass(ab_feat.values[tr], y_tr, ab_feat.values[te], dim)
            km_tr, km_te = select_univariate_multiclass(km_feat.values[tr], y_tr, km_feat.values[te], dim)
            for modality, (x_tr, x_te) in [("Abundance", (ab_tr, ab_te)), ("K-mer", (km_tr, km_te))]:
                scaler = StandardScaler()
                x_tr_sc = scaler.fit_transform(x_tr)
                x_te_sc = scaler.transform(x_te)
                for name, clf in make_models(seed, n_classes).items():
                    xtr = x_tr_sc if name in SCALED else x_tr
                    xte = x_te_sc if name in SCALED else x_te
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        clf.fit(xtr, y_tr)
                    y_pred = clf.predict(xte)
                    y_prob = clf.predict_proba(xte)
                    auc, auprc, f1, mcc = multiclass_metrics(y_te, y_pred, y_prob)
                    print(f"    dim={dim:4d} {modality:9s} {name:15s} AUC={auc:.3f}")
                    all_rows.append({
                        "round": round_num, "seed": seed, "matched_dim": dim,
                        "modality": modality, "classifier": name,
                        "AUC": auc, "AUPRC": auprc, "F1_macro": f1, "MCC": mcc,
                    })

    df = pd.DataFrame(all_rows)
    out = os.path.join(RESULTS_DIR, "geography_matched_dimension_all_rounds.csv")
    df.to_csv(out, index=False)

    summary = (
        df.groupby(["matched_dim", "modality", "classifier"])[["AUC", "AUPRC", "F1_macro", "MCC"]]
        .agg(["mean", "std"]).round(4)
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary_out = os.path.join(RESULTS_DIR, "geography_matched_dimension_summary.csv")
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
    gap_out = os.path.join(RESULTS_DIR, "geography_matched_dimension_gap.csv")
    gap_df.to_csv(gap_out, index=False)

    print(f"\n{'='*70}\nGeography matched-dimension gap (k-mer minus abundance AUC), by dim\n{'='*70}")
    print(gap_df.to_string(index=False))
    print(f"\nSaved -> {out}\nSaved -> {summary_out}\nSaved -> {gap_out}")


if __name__ == "__main__":
    main()
