"""
Obesity leave-one-study-out validation.

This robustness experiment reuses the cached abundance matrix and k-mer vectors.
For each eligible held-out study, models are trained on all other studies and
tested on the held-out study. Feature selection is fit inside each training fold.

Outputs:
  obesity/results/robustness/obesity_loso_xgboost.csv
  obesity/results/robustness/obesity_loso_xgboost_summary.csv
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../results/robustness")
os.makedirs(RESULTS_DIR, exist_ok=True)

META_CSV = os.path.join(DATA_DIR, "obesity_metadata_full.csv")
ABUNDANCE_CSV = os.path.join(DATA_DIR, "cmd3_relative_abundance.csv")
KMER_DIR = os.path.join(DATA_DIR, "kmer_counts_k8")

TOP_KMER = 1000
TOP_KMER_FUSION = 500
MIN_TEST_SAMPLES = 30
MIN_TEST_PER_CLASS = 3


def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(META_CSV)
    meta = meta.dropna(subset=["sample_id", "study_name", "label"]).copy()
    meta["label"] = meta["label"].astype(int)
    if "subject_id" in meta.columns:
        meta["subject_id"] = meta["subject_id"].fillna(meta["sample_id"])
    return meta


def clr_transform(df: pd.DataFrame) -> pd.DataFrame:
    x = df.values.astype(np.float32) + 1e-6
    log_x = np.log(x)
    return pd.DataFrame(log_x - log_x.mean(axis=1, keepdims=True),
                        index=df.index, columns=df.columns)


def build_abundance_features(abund: pd.DataFrame) -> pd.DataFrame:
    x_clr = clr_transform(abund)
    x_vals = abund.values.astype(np.float32)
    p = x_vals / (x_vals.sum(axis=1, keepdims=True) + 1e-12)
    shannon = -np.sum(p * np.log(np.where(p > 0, p, 1)), axis=1)
    simpson = 1 - np.sum(p ** 2, axis=1)
    richness = (x_vals > 0).sum(axis=1).astype(np.float32)
    x_div = pd.DataFrame({
        "alpha_shannon": shannon,
        "alpha_simpson": simpson,
        "alpha_richness": richness,
    }, index=abund.index)
    x_pw = pd.DataFrame(
        x_clr.values * (x_vals > 0).astype(np.float32),
        index=abund.index,
        columns=[f"pw__{c}" for c in x_clr.columns],
    )
    return pd.concat([x_clr, x_div, x_pw], axis=1)


def load_abundance_full() -> pd.DataFrame:
    abund = pd.read_csv(ABUNDANCE_CSV, index_col="sample_id")
    abund = abund[~abund.index.duplicated(keep="first")]
    return abund.astype(np.float32)


def load_kmer_full(sample_ids: list[str]) -> pd.DataFrame:
    vectors, ids = [], []
    seen = set()
    for sid in sample_ids:
        if sid in seen:
            continue
        seen.add(sid)
        path = os.path.join(KMER_DIR, f"{sid}_k8.npy")
        if os.path.exists(path):
            vec = np.load(path).astype(np.float32)
            total = float(vec.sum())
            if total > 0:
                vec = vec / total
            vectors.append(vec)
            ids.append(sid)
    if not vectors:
        return pd.DataFrame()
    return pd.DataFrame(
        np.vstack(vectors),
        index=ids,
        columns=[f"k{i}" for i in range(vectors[0].shape[0])],
    )


def eligible_studies(meta: pd.DataFrame, max_folds: int | None = None) -> list[str]:
    rows = []
    for study, grp in meta.groupby("study_name"):
        counts = grp["label"].value_counts()
        n0 = int(counts.get(0, 0))
        n1 = int(counts.get(1, 0))
        if len(grp) >= MIN_TEST_SAMPLES and min(n0, n1) >= MIN_TEST_PER_CLASS:
            rows.append((study, len(grp), min(n0, n1)))
    rows = sorted(rows, key=lambda x: (-x[1], x[0]))
    studies = [r[0] for r in rows]
    return studies[:max_folds] if max_folds else studies


def select_abundance_features(x_train: np.ndarray, y_train: np.ndarray,
                              x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    # L1 logistic selection is much faster than elastic-net/SAGA for repeated
    # LOSO folds, while still fitting the sparse selector inside the train fold.
    selector = SelectFromModel(
        LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=0.05,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        threshold=1e-8,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector.fit(x_train, y_train)
    return selector.transform(x_train), selector.transform(x_test), int(selector.get_support().sum())


def select_kmer_features(x_train: np.ndarray, y_train: np.ndarray,
                         x_test: np.ndarray, top_n: int) -> tuple[np.ndarray, np.ndarray, int]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pvals = ttest_ind(x_train[y_train == 1], x_train[y_train == 0],
                             equal_var=False, axis=0)
    pvals = np.nan_to_num(pvals, nan=1.0)
    _, adj, _, _ = multipletests(pvals, method="fdr_bh")
    idx = np.argsort(adj)[:top_n]
    return x_train[:, idx], x_test[:, idx], len(idx)


def fit_predict_xgboost(x_train: np.ndarray, y_train: np.ndarray,
                        x_test: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pos_w = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.5,
        min_child_weight=3,
        scale_pos_weight=pos_w,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    return clf.predict(x_test), clf.predict_proba(x_test)[:, 1]


def score_row(study: str, pipeline: str, y_test: np.ndarray, y_pred: np.ndarray,
              y_prob: np.ndarray, n_train: int, n_features: int) -> dict:
    return {
        "held_out_study": study,
        "pipeline": pipeline,
        "classifier": "XGBoost",
        "n_train": int(n_train),
        "n_test": int(len(y_test)),
        "test_obese": int((y_test == 1).sum()),
        "test_nonobese": int((y_test == 0).sum()),
        "n_features": int(n_features),
        "AUC": round(float(roc_auc_score(y_test, y_prob)), 4),
        "AUPRC": round(float(average_precision_score(y_test, y_prob)), 4),
        "F1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "MCC": round(float(matthews_corrcoef(y_test, y_pred)), 4),
    }


def run_pipeline_for_fold(study: str, meta: pd.DataFrame, features: pd.DataFrame,
                          pipeline: str, seed: int) -> dict | None:
    shared = meta["sample_id"][meta["sample_id"].isin(features.index)]
    m = (
        meta[meta["sample_id"].isin(shared)]
        .drop_duplicates("sample_id")
        .set_index("sample_id")
    )
    shared = pd.Index(shared).intersection(m.index)
    x_full = features.loc[shared]

    test_idx = m.index[m["study_name"] == study]
    train_idx = m.index[m["study_name"] != study]
    if len(test_idx) < MIN_TEST_SAMPLES or m.loc[test_idx, "label"].nunique() < 2:
        return None

    y_train = m.loc[train_idx, "label"].values
    y_test = m.loc[test_idx, "label"].values
    x_train_all = x_full.loc[train_idx].values.astype(np.float32)
    x_test_all = x_full.loc[test_idx].values.astype(np.float32)

    if pipeline == "Abundance":
        x_train, x_test, n_features = select_abundance_features(x_train_all, y_train, x_test_all)
    elif pipeline == "K-mer":
        x_train, x_test, n_features = select_kmer_features(x_train_all, y_train, x_test_all, TOP_KMER)
    else:
        raise ValueError(pipeline)

    if pipeline == "Abundance":
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    y_pred, y_prob = fit_predict_xgboost(x_train, y_train, x_test, seed)
    return score_row(study, pipeline, y_test, y_pred, y_prob, len(y_train), n_features)


def run_fusion_for_fold(study: str, meta: pd.DataFrame, abund_features: pd.DataFrame,
                        kmer_features: pd.DataFrame, seed: int) -> dict | None:
    shared = meta["sample_id"][
        meta["sample_id"].isin(abund_features.index)
        & meta["sample_id"].isin(kmer_features.index)
    ]
    m = (
        meta[meta["sample_id"].isin(shared)]
        .drop_duplicates("sample_id")
        .set_index("sample_id")
    )
    shared = pd.Index(shared).intersection(m.index)
    test_idx = m.index[m["study_name"] == study]
    train_idx = m.index[m["study_name"] != study]
    if len(test_idx) < MIN_TEST_SAMPLES or m.loc[test_idx, "label"].nunique() < 2:
        return None

    y_train = m.loc[train_idx, "label"].values
    y_test = m.loc[test_idx, "label"].values

    x_ab_train_all = abund_features.loc[train_idx].values.astype(np.float32)
    x_ab_test_all = abund_features.loc[test_idx].values.astype(np.float32)
    x_km_train_all = kmer_features.loc[train_idx].values.astype(np.float32)
    x_km_test_all = kmer_features.loc[test_idx].values.astype(np.float32)

    x_ab_train, x_ab_test, n_ab = select_abundance_features(
        x_ab_train_all, y_train, x_ab_test_all)
    x_km_train, x_km_test, n_km = select_kmer_features(
        x_km_train_all, y_train, x_km_test_all, TOP_KMER_FUSION)

    x_train = np.hstack([x_ab_train, x_km_train])
    x_test = np.hstack([x_ab_test, x_km_test])
    y_pred, y_prob = fit_predict_xgboost(x_train, y_train, x_test, seed)
    return score_row(study, "Fusion", y_test, y_pred, y_prob,
                     len(y_train), n_ab + n_km)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("pipeline")[["AUC", "AUPRC", "F1_macro", "MCC"]]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    return summary.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipelines", nargs="+",
                        default=["abundance", "kmer", "fusion"],
                        choices=["abundance", "kmer", "fusion"])
    parser.add_argument("--max-folds", type=int, default=None,
                        help="Limit held-out studies for smoke tests.")
    args = parser.parse_args()

    meta = load_metadata()
    rows = []

    print("Loading abundance features...")
    abund = load_abundance_full()
    shared_abund = meta["sample_id"][meta["sample_id"].isin(abund.index)]
    abund = abund.loc[shared_abund]
    prev = (abund > 0).mean()
    abund = abund.loc[:, prev >= 0.10]
    abund_features = build_abundance_features(abund)
    meta_abund = meta[meta["sample_id"].isin(abund_features.index)].copy()

    kmer_features = None
    if "kmer" in args.pipelines or "fusion" in args.pipelines:
        print("Loading cached k-mer features...")
        kmer_features = load_kmer_full(meta["sample_id"].tolist())
        print(f"  k-mer matrix: {kmer_features.shape}")

    base_meta = meta_abund if "kmer" not in args.pipelines else meta[
        meta["sample_id"].isin(kmer_features.index)
    ].copy()
    if "fusion" in args.pipelines:
        base_meta = meta[
            meta["sample_id"].isin(abund_features.index)
            & meta["sample_id"].isin(kmer_features.index)
        ].copy()
    base_meta = base_meta.drop_duplicates("sample_id")

    studies = eligible_studies(base_meta, args.max_folds)
    print(f"Eligible held-out studies: {len(studies)}")

    for fold, study in enumerate(studies, 1):
        print(f"\n[{fold}/{len(studies)}] Holding out {study}")
        seed = 1000 + fold
        if "abundance" in args.pipelines:
            row = run_pipeline_for_fold(study, base_meta, abund_features, "Abundance", seed)
            if row:
                print(f"  Abundance AUC={row['AUC']:.3f} F1={row['F1_macro']:.3f}")
                rows.append(row)
        if "kmer" in args.pipelines:
            row = run_pipeline_for_fold(study, base_meta, kmer_features, "K-mer", seed)
            if row:
                print(f"  K-mer     AUC={row['AUC']:.3f} F1={row['F1_macro']:.3f}")
                rows.append(row)
        if "fusion" in args.pipelines:
            row = run_fusion_for_fold(study, base_meta, abund_features, kmer_features, seed)
            if row:
                print(f"  Fusion    AUC={row['AUC']:.3f} F1={row['F1_macro']:.3f}")
                rows.append(row)

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "obesity_loso_xgboost.csv")
    df.to_csv(out, index=False)
    summary = summarize(df)
    summary_out = os.path.join(RESULTS_DIR, "obesity_loso_xgboost_summary.csv")
    summary.to_csv(summary_out, index=False)
    print(f"\nSaved {out}")
    print(f"Saved {summary_out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
