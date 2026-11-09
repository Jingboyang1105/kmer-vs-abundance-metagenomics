#!/usr/bin/env python3
"""
HMP2 (IBD) k=8 read-depth sensitivity.

cMD3 k-mer
extraction truncates to the first 200,000 reads of one FASTQ mate, while HMP2
extraction uses full paired-end reads with no truncation -- a difference in
method, not just dataset. Any HMP2-vs-cMD3 k-mer performance difference could
be a read-depth artifact rather than a biological/representation difference.

This re-extracts HMP2 k=8 k-mer vectors at three truncated depths (50,000 /
100,000 / 200,000 reads per mate -- the last matching cMD3 exactly) via
extract_depths.sh, then re-runs the same 10-round participant-level IBD
k-mer classification (ibd/kmer_pipeline.py design: Welch t-test + BH top-1000
selection, 4 classifiers) at each depth, plus the existing untruncated full-
depth cached vectors (archive/line2/kmer_counts/k8) as the reference point.

If AUC at matched 200k depth is close to the full-depth AUC, extraction depth
is not the primary driver of the HMP2 k-mer advantage. If AUC drops
substantially toward chance as depth is restricted, depth is a material
confound that should be discussed explicitly when comparing HMP2 (IBD) and
cMD3 (obesity, geography) k-mer results.

Outputs:
  results/ibd_depth_sensitivity_all_rounds.csv
  results/ibd_depth_sensitivity_summary.csv
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

# REPO_ROOT: supplementary_experiments/05_hmp2_read_depth_sensitivity -> up two levels -> repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_ROOT = os.environ.get("MBEC_DATA_ROOT", REPO_ROOT)

META_CSV = os.path.join(DATA_ROOT, "ibd", "data", "raw", "hmp2_metadata_2018-08-20.csv")
FULL_DEPTH_DIR = os.path.join(DATA_ROOT, "ibd", "kmer_matrix", "k8")
TRUNCATED_DIR = os.path.join(SCRIPT_DIR, "data", "kmer_counts_depth")
DEPTHS = [50000, 100000, 200000]  # + "full" (untruncated, cached)

TOP_KMER = 1000
SEEDS = [42 + i * 100 for i in range(10)]


def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(META_CSV, low_memory=False)
    meta = meta[meta["data_type"] == "metagenomics"].copy()
    meta = meta[meta["diagnosis"].isin(["CD", "UC", "nonIBD"])].copy()
    meta["diagnosis"] = meta["diagnosis"].replace({"CD": "IBD", "UC": "IBD"})
    meta = meta[["External ID", "Participant ID", "diagnosis"]].drop_duplicates("External ID")
    meta.rename(columns={"External ID": "sample_id", "Participant ID": "participant"}, inplace=True)
    meta["file_id"] = meta["sample_id"].str.replace(r"_P$", "", regex=True)
    meta["label"] = (meta["diagnosis"] == "IBD").astype(int)
    return meta.drop_duplicates("file_id").set_index("file_id")


def load_matrix(directory: str, suffix: str, meta: pd.DataFrame):
    files = {f[: -len(suffix)]: os.path.join(directory, f)
             for f in os.listdir(directory) if f.endswith(suffix)}
    ids = sorted(set(files).intersection(meta.index))
    vecs = []
    for fid in ids:
        v = np.load(files[fid]).astype(np.float32)
        total = v.sum()
        vecs.append(v / total if total > 0 else v)
    X = np.vstack(vecs)
    y = meta.loc[ids, "label"].values
    parts = meta.loc[ids, "participant"].values
    return X, y, parts, ids


def select_kmers(x_train, y_train, top_n):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pvals = stats.ttest_ind(x_train[y_train == 0], x_train[y_train == 1], equal_var=False, axis=0)
    pvals = np.nan_to_num(pvals, nan=1.0)
    _, adj, _, _ = multipletests(pvals, method="fdr_bh")
    return np.argsort(adj)[: min(top_n, x_train.shape[1])]


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


def evaluate_depth(depth_label, X, y, parts):
    unique_parts = pd.Series(parts).drop_duplicates().values
    part_label = pd.Series(y, index=parts).groupby(level=0).first()
    rows = []
    for round_num, seed in enumerate(SEEDS, 1):
        parts_train, parts_test = train_test_split(
            unique_parts, test_size=0.2, random_state=seed,
            stratify=part_label.loc[unique_parts].values,
        )
        tr = np.isin(parts, parts_train)
        te = np.isin(parts, parts_test)
        x_tr_all, x_te_all = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        idx = select_kmers(x_tr_all, y_tr, TOP_KMER)
        x_tr, x_te = x_tr_all[:, idx], x_te_all[:, idx]
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
            print(f"  depth={depth_label:>8} round={round_num} {name:15s} AUC={auc:.3f}")
            rows.append({
                "depth": depth_label, "round": round_num, "seed": seed, "classifier": name,
                "n_train": int(tr.sum()), "n_test": int(te.sum()),
                "AUC": auc,
                "AUPRC": round(float(average_precision_score(y_te, y_prob)), 4),
                "F1_macro": round(float(f1_score(y_te, y_pred, average="macro")), 4),
                "MCC": round(float(matthews_corrcoef(y_te, y_pred)), 4),
            })
    return rows


def main():
    missing = [p for p in (META_CSV, FULL_DEPTH_DIR, TRUNCATED_DIR) if not os.path.exists(p)]
    if missing:
        print("Missing required local data, cannot run HMP2 read-depth sensitivity check:")
        for p in missing:
            print(f"  - {p}")
        print(
            "Run ibd/extract_kmers.py + ibd/build_kmer_matrix.py for the full-depth cache, "
            "place the HMP2 metadata CSV under ibd/data/raw/, and run extract_depths.sh in "
            "this folder for the truncated-depth vectors (see README 'Data' section). "
            "Set MBEC_DATA_ROOT if these live outside the repo."
        )
        raise SystemExit(1)

    meta = load_metadata()
    all_rows = []

    print("=== depth=full (untruncated, cached) ===")
    X, y, parts, ids = load_matrix(FULL_DEPTH_DIR, "_k8.npy", meta)
    print(f"  n={len(ids)}")
    all_rows.extend(evaluate_depth("full", X, y, parts))

    for depth in DEPTHS:
        print(f"\n=== depth={depth} ===")
        suffix = f"_k8_r{depth}.npy"
        X, y, parts, ids = load_matrix(TRUNCATED_DIR, suffix, meta)
        print(f"  n={len(ids)}")
        all_rows.extend(evaluate_depth(str(depth), X, y, parts))

    df = pd.DataFrame(all_rows)
    out = os.path.join(RESULTS_DIR, "ibd_depth_sensitivity_all_rounds.csv")
    df.to_csv(out, index=False)

    summary = (
        df.groupby(["depth", "classifier"])[["AUC", "AUPRC", "F1_macro", "MCC"]]
        .agg(["mean", "std"]).round(4)
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    depth_order = {"50000": 0, "100000": 1, "200000": 2, "full": 3}
    summary["_order"] = summary["depth"].map(depth_order)
    summary = summary.sort_values(["classifier", "_order"]).drop(columns="_order")
    summary_out = os.path.join(RESULTS_DIR, "ibd_depth_sensitivity_summary.csv")
    summary.to_csv(summary_out, index=False)

    print(f"\n{'='*70}\nDepth sensitivity summary\n{'='*70}")
    print(summary.to_string(index=False))
    print(f"\nSaved -> {out}\nSaved -> {summary_out}")


if __name__ == "__main__":
    main()
