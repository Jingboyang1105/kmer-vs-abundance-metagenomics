"""
Obesity study-prediction sanity check.

Auxiliary classifiers predict study_name from abundance, k-mer, and fused
features. High performance indicates that microbiome representations encode
cohort-specific structure, which may include technical batch and population
signals.

Outputs:
  obesity/results/robustness/obesity_study_prediction.csv
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../results/robustness")
os.makedirs(RESULTS_DIR, exist_ok=True)

META_CSV = os.path.join(DATA_DIR, "obesity_metadata_full.csv")
ABUNDANCE_CSV = os.path.join(DATA_DIR, "cmd3_relative_abundance.csv")
KMER_DIR = os.path.join(DATA_DIR, "kmer_counts_k8")
TOP_KMER = 1000
MIN_STUDY_SAMPLES = 50


def load_meta() -> pd.DataFrame:
    meta = pd.read_csv(META_CSV)
    counts = meta["study_name"].value_counts()
    keep = counts[counts >= MIN_STUDY_SAMPLES].index
    return meta[meta["study_name"].isin(keep)].copy()


def clr_transform(df: pd.DataFrame) -> pd.DataFrame:
    x = df.values.astype(np.float32) + 1e-6
    log_x = np.log(x)
    return pd.DataFrame(log_x - log_x.mean(axis=1, keepdims=True),
                        index=df.index, columns=df.columns)


def build_abundance_features(abund: pd.DataFrame) -> pd.DataFrame:
    x_clr = clr_transform(abund)
    x_vals = abund.values.astype(np.float32)
    p = x_vals / (x_vals.sum(axis=1, keepdims=True) + 1e-12)
    div = pd.DataFrame({
        "alpha_shannon": -np.sum(p * np.log(np.where(p > 0, p, 1)), axis=1),
        "alpha_simpson": 1 - np.sum(p ** 2, axis=1),
        "alpha_richness": (x_vals > 0).sum(axis=1).astype(np.float32),
    }, index=abund.index)
    pw = pd.DataFrame(
        x_clr.values * (x_vals > 0).astype(np.float32),
        index=abund.index,
        columns=[f"pw__{c}" for c in x_clr.columns],
    )
    return pd.concat([x_clr, div, pw], axis=1)


def load_abundance_features(meta: pd.DataFrame) -> pd.DataFrame:
    abund = pd.read_csv(ABUNDANCE_CSV, index_col="sample_id")
    abund = abund[~abund.index.duplicated(keep="first")].astype(np.float32)
    shared = meta["sample_id"][meta["sample_id"].isin(abund.index)]
    abund = abund.loc[shared]
    prev = (abund > 0).mean()
    return build_abundance_features(abund.loc[:, prev >= 0.10])


def load_kmer_features(sample_ids: list[str]) -> pd.DataFrame:
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
    return pd.DataFrame(np.vstack(vectors), index=ids,
                        columns=[f"k{i}" for i in range(vectors[0].shape[0])])


def macro_auc(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    vals = []
    for cls in np.unique(y_true):
        y_bin = (y_true == cls).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        vals.append(roc_auc_score(y_bin, y_prob[:, cls]))
    return float(np.mean(vals)) if vals else float("nan")


def run_one(pipeline: str, features: pd.DataFrame, meta: pd.DataFrame,
            seed: int) -> dict:
    shared = meta["sample_id"][meta["sample_id"].isin(features.index)]
    m = (
        meta[meta["sample_id"].isin(shared)]
        .drop_duplicates("sample_id")
        .set_index("sample_id")
    )
    shared = pd.Index(shared).intersection(m.index)
    x = features.loc[shared].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(m["study_name"].values)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(x, y))
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if pipeline == "Abundance":
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    elif pipeline == "K-mer":
        selector = SelectKBest(f_classif, k=min(TOP_KMER, x_train.shape[1]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_train = selector.fit_transform(x_train, y_train)
        x_test = selector.transform(x_test)
    elif pipeline == "Fusion":
        pass

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test)
    return {
        "pipeline": pipeline,
        "classifier": "Random Forest",
        "n_samples": int(len(y)),
        "n_studies": int(len(le.classes_)),
        "n_features": int(x_train.shape[1]),
        "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "F1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "AUC_macro_ovr": round(macro_auc(y_test, y_prob, len(le.classes_)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipelines", nargs="+",
                        default=["abundance", "kmer", "fusion"],
                        choices=["abundance", "kmer", "fusion"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta = load_meta()
    rows = []
    abundance_features = None
    kmer_features = None

    if "abundance" in args.pipelines or "fusion" in args.pipelines:
        print("Loading abundance features...")
        abundance_features = load_abundance_features(meta)
        if "abundance" in args.pipelines:
            rows.append(run_one("Abundance", abundance_features, meta, args.seed))

    if "kmer" in args.pipelines or "fusion" in args.pipelines:
        print("Loading k-mer features...")
        kmer_features = load_kmer_features(meta["sample_id"].tolist())
        if "kmer" in args.pipelines:
            rows.append(run_one("K-mer", kmer_features, meta, args.seed))

    if "fusion" in args.pipelines:
        print("Building fused feature matrix...")
        shared = abundance_features.index.intersection(kmer_features.index)
        m = meta[meta["sample_id"].isin(shared)].drop_duplicates("sample_id").copy()
        shared = pd.Index(shared).intersection(pd.Index(m["sample_id"]))
        ab = abundance_features.loc[shared].values.astype(np.float32)
        km = kmer_features.loc[shared].values.astype(np.float32)
        le = LabelEncoder()
        y = le.fit_transform(m.set_index("sample_id").loc[shared, "study_name"].values)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
        train_idx, test_idx = next(splitter.split(ab, y))
        km_sel = SelectKBest(f_classif, k=min(500, km.shape[1]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km_train = km_sel.fit_transform(km[train_idx], y[train_idx])
        km_test = km_sel.transform(km[test_idx])
        scaler = StandardScaler()
        ab_train = scaler.fit_transform(ab[train_idx])
        ab_test = scaler.transform(ab[test_idx])
        x_train = np.hstack([ab_train, km_train])
        x_test = np.hstack([ab_test, km_test])
        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=args.seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        clf.fit(x_train, y[train_idx])
        y_pred = clf.predict(x_test)
        y_prob = clf.predict_proba(x_test)
        rows.append({
            "pipeline": "Fusion",
            "classifier": "Random Forest",
            "n_samples": int(len(y)),
            "n_studies": int(len(le.classes_)),
            "n_features": int(x_train.shape[1]),
            "Accuracy": round(float(accuracy_score(y[test_idx], y_pred)), 4),
            "F1_macro": round(float(f1_score(y[test_idx], y_pred, average="macro", zero_division=0)), 4),
            "AUC_macro_ovr": round(macro_auc(y[test_idx], y_prob, len(le.classes_)), 4),
        })

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "obesity_study_prediction.csv")
    df.to_csv(out, index=False)
    print(f"Saved {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
