"""
Obesity abundance-only sample-size sensitivity.

Runs the abundance pipeline at several subsample sizes to show that the main
1600-sample choice is not a cherry-picked sample size. This deliberately uses
abundance features only because it is a lightweight robustness check.

Output:
  obesity/results/robustness/obesity_abundance_subsampling_sensitivity.csv
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../results/robustness")
os.makedirs(RESULTS_DIR, exist_ok=True)

META_CSV = os.path.join(DATA_DIR, "obesity_metadata_full.csv")
ABUNDANCE_CSV = os.path.join(DATA_DIR, "cmd3_relative_abundance.csv")
BASE_SEED = 2026


def clr_transform(df: pd.DataFrame) -> pd.DataFrame:
    x = df.values.astype(np.float32) + 1e-6
    log_x = np.log(x)
    return pd.DataFrame(log_x - log_x.mean(axis=1, keepdims=True),
                        index=df.index, columns=df.columns)


def build_features(abund: pd.DataFrame) -> pd.DataFrame:
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


def stratified_subsample(meta: pd.DataFrame, n_total: int, seed: int) -> pd.DataFrame:
    n_obese = round(n_total * 0.20)
    n_nonobese = n_total - n_obese
    rng = np.random.default_rng(seed)
    parts = []
    for label, n_target in [(1, n_obese), (0, n_nonobese)]:
        subset = meta[meta["label"] == label].copy()
        study_counts = subset["study_name"].value_counts()
        sampled, remaining = [], n_target
        for study, count in study_counts.items():
            n_from = max(1, round(n_target * count / len(subset)))
            n_from = min(n_from, count, remaining)
            sampled.append(
                subset[subset["study_name"] == study].sample(
                    n=n_from, random_state=int(rng.integers(1_000_000))
                )
            )
            remaining -= n_from
            if remaining <= 0:
                break
        parts.append(pd.concat(sampled))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def add_split(meta: pd.DataFrame, seed: int) -> pd.DataFrame:
    meta = meta.copy()
    meta["subject_id"] = meta["subject_id"].fillna(meta["sample_id"])
    subj_label = (
        meta.groupby("subject_id")["label"]
        .agg(lambda x: int(x.mode()[0]))
        .reset_index()
    )
    train_subj, _ = train_test_split(
        subj_label["subject_id"].values,
        test_size=0.2,
        random_state=seed,
        stratify=subj_label["label"].values,
    )
    train_set = set(train_subj)
    meta["split"] = meta["subject_id"].map(lambda s: "train" if s in train_set else "test")
    return meta


def run_one(meta: pd.DataFrame, features: pd.DataFrame, n_total: int, round_num: int,
            seed: int) -> dict:
    sub = stratified_subsample(meta, n_total, seed)
    sub = add_split(sub, seed)
    shared = sub["sample_id"][sub["sample_id"].isin(features.index)]
    sub = sub[sub["sample_id"].isin(shared)].set_index("sample_id")
    x_full = features.loc[shared]

    train_idx = sub.index[sub["split"] == "train"]
    test_idx = sub.index[sub["split"] == "test"]
    y_train = sub.loc[train_idx, "label"].values.astype(int)
    y_test = sub.loc[test_idx, "label"].values.astype(int)
    x_train_all = x_full.loc[train_idx].values.astype(np.float32)
    x_test_all = x_full.loc[test_idx].values.astype(np.float32)

    selector = SelectFromModel(
        LogisticRegression(
            penalty="elasticnet",
            l1_ratio=0.5,
            solver="saga",
            C=0.1,
            max_iter=3000,
            class_weight="balanced",
            random_state=seed,
        ),
        threshold="mean",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector.fit(x_train_all, y_train)
    x_train = selector.transform(x_train_all)
    x_test = selector.transform(x_test_all)

    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train)
    x_test_sc = scaler.transform(x_test)

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
    clf.fit(x_train_sc, y_train)
    y_pred = clf.predict(x_test_sc)
    y_prob = clf.predict_proba(x_test_sc)[:, 1]
    return {
        "sample_size_target": int(n_total),
        "round": int(round_num),
        "seed": int(seed),
        "n_samples_matched": int(len(sub)),
        "train_n": int(len(y_train)),
        "test_n": int(len(y_test)),
        "test_obese": int((y_test == 1).sum()),
        "test_nonobese": int((y_test == 0).sum()),
        "n_features": int(selector.get_support().sum()),
        "classifier": "XGBoost",
        "AUC": round(float(roc_auc_score(y_test, y_prob)), 4),
        "AUPRC": round(float(average_precision_score(y_test, y_prob)), 4),
        "F1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "MCC": round(float(matthews_corrcoef(y_test, y_pred)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[800, 1200, 1600])
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    meta = pd.read_csv(META_CSV)
    meta["label"] = meta["label"].astype(int)
    abund = pd.read_csv(ABUNDANCE_CSV, index_col="sample_id")
    abund = abund[~abund.index.duplicated(keep="first")].astype(np.float32)
    meta = meta[meta["sample_id"].isin(abund.index)].copy()
    abund = abund.loc[meta["sample_id"]]
    prev = (abund > 0).mean()
    features = build_features(abund.loc[:, prev >= 0.10])

    rows = []
    for n_total in args.sizes:
        for r in range(1, args.rounds + 1):
            seed = BASE_SEED + n_total + r * 100
            print(f"Running n={n_total}, round={r}, seed={seed}")
            row = run_one(meta, features, n_total, r, seed)
            print(f"  AUC={row['AUC']:.3f} F1={row['F1_macro']:.3f} MCC={row['MCC']:.3f}")
            rows.append(row)

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "obesity_abundance_subsampling_sensitivity.csv")
    df.to_csv(out, index=False)
    summary = (
        df.groupby("sample_size_target")[["AUC", "AUPRC", "F1_macro", "MCC"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary_out = os.path.join(
        RESULTS_DIR, "obesity_abundance_subsampling_sensitivity_summary.csv")
    summary.to_csv(summary_out, index=False)
    print(f"Saved {out}")
    print(f"Saved {summary_out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
