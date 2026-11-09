#!/usr/bin/env python3
"""
Geography: quantify the country <-> study confound, and re-evaluate
30-country classification under a study-aware (study-held-out) split.

The very high AUC values in the geographic-origin task may reflect
study-source / technical structure rather than biological geographic signal.
This script (a) quantifies the relationship between country and study
identity and (b) repeats the analysis using study-aware splitting.

Part A: contingency analysis between `country` and `study_name` in the full
cMD3 geography metadata (22,397 samples, 30 countries, 89 studies) -- how many
countries are represented by only one study (i.e. country label is a perfect
proxy for study/batch for that class), and Cramer's V association strength.

Part B: re-run the abundance pipeline (same CLR + alpha-diversity + ElasticNet
feature-selection + 4-classifier design as
archive/line3/geography/scripts/02a_abundance_pipeline.py, matching the main
manuscript's geography abundance pipeline) but replace the original per-sample
train/test split with a STUDY-held-out split: entire studies are randomly
assigned to train or test, so no study (and therefore little of its batch
signal) is shared between train and test. Countries whose only study falls in
the held-out set cannot be predicted at all under this split -- the fraction
of test samples this affects is itself part of the confound quantification.

K-mer features are not re-evaluated here: no cached k-mer vectors exist for
the geography task on this server (raw-read extraction for ~22k cMD3 samples
would require large-scale SRA streaming, out of scope for this analysis) --
this is noted as a scope limitation in results/README.md.

Outputs:
  results/country_study_confound.csv        per-country study counts
  results/country_study_confound_summary.txt  Cramer's V + headline stats
  results/geography_study_holdout_all_rounds.csv
  results/geography_study_holdout_summary.csv
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# REPO_ROOT: supplementary_experiments/02_geography_study_confound -> up two levels -> repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_ROOT = os.environ.get("MBEC_DATA_ROOT", REPO_ROOT)

# Produced by geography/metadata_prep.py and geography/download_abundance.R;
# see README "Data" and "Execution order".
META_FULL = os.path.join(DATA_ROOT, "geography", "data", "geo_metadata_full.csv")
ABUNDANCE_CSV = os.path.join(DATA_ROOT, "geography", "data", "cmd3_geography_abundance.csv")
ROUND1_CSV = os.path.join(DATA_ROOT, "geography", "data", "geo_metadata_round1.csv")

N_REPEATS = 5
TEST_STUDY_FRACTION = 0.20
MIN_TEST_SAMPLES_PER_COUNTRY = 3


# ─────────────────────────── Part A: confound quantification ──────────────

def cramers_v(confusion: np.ndarray) -> float:
    chi2 = 0.0
    n = confusion.sum()
    row_sums = confusion.sum(axis=1, keepdims=True)
    col_sums = confusion.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(expected > 0, (confusion - expected) ** 2 / expected, 0))
    k = min(confusion.shape) - 1
    return float(np.sqrt((chi2 / n) / max(k, 1)))


def part_a(meta: pd.DataFrame):
    cs = meta.groupby("country")["study_name"].nunique().sort_values()
    cs.to_frame("n_studies").to_csv(os.path.join(RESULTS_DIR, "country_study_confound.csv"))

    ct = pd.crosstab(meta["country"], meta["study_name"])
    v = cramers_v(ct.values)

    studies_multi_country = meta.groupby("study_name")["country"].nunique()
    n_multi_country_studies = int((studies_multi_country > 1).sum())

    lines = [
        f"Total samples: {len(meta)}",
        f"Countries: {meta['country'].nunique()}   Studies: {meta['study_name'].nunique()}",
        f"Countries represented by exactly 1 study: {int((cs == 1).sum())} / {len(cs)} "
        f"({100*(cs==1).mean():.1f}%)",
        f"Studies spanning >1 country: {n_multi_country_studies} / {meta['study_name'].nunique()}",
        f"Cramer's V (country vs study_name): {v:.4f}  "
        f"(0=independent, 1=perfectly confounded)",
        "",
        "Interpretation: for the {}/{} countries with a single contributing study, "
        "country identity and study/batch identity are mathematically identical -- "
        "any classifier recovering country for those samples is, by construction, "
        "also recovering study/batch identity.".format(int((cs == 1).sum()), len(cs)),
    ]
    text = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "country_study_confound_summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    return cs


# ─────────────────────────── Part B: study-held-out classification ────────

def clr_transform(df):
    x = df.values + 1e-6
    log_x = np.log(x)
    return log_x - log_x.mean(axis=1, keepdims=True)


def build_features(abund: pd.DataFrame) -> np.ndarray:
    clr = clr_transform(abund)
    x = abund.values
    p = x / (x.sum(axis=1, keepdims=True) + 1e-12)
    shannon = -np.sum(p * np.log(np.where(p > 0, p, 1)), axis=1)
    simpson = 1 - np.sum(p ** 2, axis=1)
    richness = (x > 0).sum(axis=1).astype(float)
    pw = clr * (x > 0).astype(float)
    return np.hstack([clr, shannon[:, None], simpson[:, None], richness[:, None], pw])


def multiclass_metrics(y_true, y_pred, y_prob, n_classes):
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


def study_held_out_split(meta: pd.DataFrame, seed: int):
    rng = np.random.RandomState(seed)
    studies = meta["study_name"].unique()
    rng.shuffle(studies)
    n_test_studies = max(1, int(len(studies) * TEST_STUDY_FRACTION))
    test_studies = set(studies[:n_test_studies])
    train_idx = meta.index[~meta["study_name"].isin(test_studies)]
    test_idx = meta.index[meta["study_name"].isin(test_studies)]
    return train_idx, test_idx, test_studies


def run_repeat(meta_sub: pd.DataFrame, abund_full: pd.DataFrame, seed: int) -> pd.DataFrame:
    shared = meta_sub["sample_id"][meta_sub["sample_id"].isin(abund_full.index)]
    meta_sub = meta_sub[meta_sub["sample_id"].isin(shared)].drop_duplicates("sample_id").set_index("sample_id")
    abund = abund_full.loc[meta_sub.index]
    prev = (abund > 0).mean()
    abund = abund.loc[:, prev >= 0.10]
    X_full = build_features(abund)
    X_full = pd.DataFrame(X_full, index=meta_sub.index)

    train_idx, test_idx, test_studies = study_held_out_split(meta_sub, seed)

    train_countries = set(meta_sub.loc[train_idx, "country"].unique())
    test_all = meta_sub.loc[test_idx]
    n_test_total = len(test_all)
    test_keep = test_all[test_all["country"].isin(train_countries)]
    n_dropped = n_test_total - len(test_keep)
    country_counts = test_keep["country"].value_counts()
    keep_countries = country_counts[country_counts >= MIN_TEST_SAMPLES_PER_COUNTRY].index
    test_keep = test_keep[test_keep["country"].isin(keep_countries)]

    print(f"  [seed={seed}] held-out studies={len(test_studies)}  "
          f"test samples total={n_test_total}  "
          f"dropped (country unseen in train)={n_dropped} "
          f"({100*n_dropped/max(n_test_total,1):.1f}%)  "
          f"scored test samples={len(test_keep)}")

    if len(test_keep) < 20 or test_keep["country"].nunique() < 2:
        return pd.DataFrame(), n_test_total, n_dropped

    le = LabelEncoder()
    le.fit(meta_sub.loc[train_idx, "country"].values)
    y_train = le.transform(meta_sub.loc[train_idx, "country"].values)
    y_test = le.transform(test_keep["country"].values)

    X_train = X_full.loc[train_idx].values
    X_test = X_full.loc[test_keep.index].values

    selector = SelectFromModel(
        LogisticRegression(l1_ratio=0.5, solver="saga", C=0.1, max_iter=3000,
                            class_weight="balanced", random_state=seed),
        threshold="mean",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector.fit(X_train, y_train)
    X_train_sel = selector.transform(X_train)
    X_test_sel = selector.transform(X_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_sel)
    X_test_sc = scaler.transform(X_test_sel)

    n_classes = len(le.classes_)
    models = {
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
    rows = []
    for name, clf in models.items():
        sc = name in ("SVM (RBF)", "LASSO-LR")
        xtr = X_train_sc if sc else X_train_sel
        xte = X_test_sc if sc else X_test_sel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(xtr, y_train)
        y_pred = clf.predict(xte)
        y_prob = clf.predict_proba(xte)
        auc, auprc, f1, mcc = multiclass_metrics(y_test, y_pred, y_prob, n_classes)
        print(f"    {name:15s} AUC={auc:.3f}  F1={f1:.3f}  MCC={mcc:.3f}")
        rows.append({
            "seed": seed, "classifier": name, "pipeline": "Abundance (study-held-out)",
            "n_train": int(len(y_train)), "n_test_scored": int(len(y_test)),
            "n_test_dropped_unseen_country": int(n_dropped),
            "n_held_out_studies": len(test_studies),
            "AUC": auc, "AUPRC": auprc, "F1_macro": f1, "MCC": mcc,
        })
    return pd.DataFrame(rows), n_test_total, n_dropped


def part_b(meta_full: pd.DataFrame, abund_full: pd.DataFrame):
    all_rows = []
    for i in range(N_REPEATS):
        seed = 3000 + i
        print(f"\n── Study-held-out repeat {i+1}/{N_REPEATS} (seed={seed}) ──")
        df, n_total, n_dropped = run_repeat(meta_full, abund_full, seed)
        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        print("No usable repeats.")
        return

    full_df = pd.concat(all_rows, ignore_index=True)
    out = os.path.join(RESULTS_DIR, "geography_study_holdout_all_rounds.csv")
    full_df.to_csv(out, index=False)

    summary = (
        full_df.groupby("classifier")[["AUC", "AUPRC", "F1_macro", "MCC"]]
        .agg(["mean", "std", "count"]).round(4)
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary_out = os.path.join(RESULTS_DIR, "geography_study_holdout_summary.csv")
    summary.to_csv(summary_out, index=False)
    print(f"\n{'='*70}\nGeography study-held-out summary ({N_REPEATS} repeats)\n{'='*70}")
    print(summary.to_string(index=False))
    print(f"\nSaved -> {out}\nSaved -> {summary_out}")


def main():
    missing = [p for p in (META_FULL, ABUNDANCE_CSV, ROUND1_CSV) if not os.path.exists(p)]
    if missing:
        print("Missing required local data, cannot run geography study-confound analysis:")
        for p in missing:
            print(f"  - {p}")
        print(
            "Run geography/metadata_prep.py and geography/download_abundance.R first "
            "(see README 'Execution order'). Set MBEC_DATA_ROOT if these live outside the repo."
        )
        raise SystemExit(1)

    print("Loading full geography metadata...")
    meta_full = pd.read_csv(META_FULL)
    print(f"  {meta_full.shape}")

    print("\n" + "=" * 70)
    print("PART A: country <-> study confound quantification")
    print("=" * 70)
    part_a(meta_full)

    print("\n" + "=" * 70)
    print("PART B: study-held-out re-classification (abundance pipeline)")
    print("=" * 70)
    print("Loading abundance matrix (this can take a minute)...")
    abund_full = pd.read_csv(ABUNDANCE_CSV, index_col="sample_id")
    abund_full = abund_full[~abund_full.index.duplicated(keep="first")]
    print(f"  {abund_full.shape}")

    # Subsample to keep runtime bounded: use the same 1,587-sample balanced
    # pool as manuscript round 1, but with study-held-out (not per-sample)
    # splitting instead of the original random participant-level split.
    round1 = pd.read_csv(ROUND1_CSV)
    part_b(round1, abund_full)


if __name__ == "__main__":
    main()
