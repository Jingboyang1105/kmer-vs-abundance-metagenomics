"""
Matched-intersection benchmark for abundance vs k-mer feature modalities.

This script addresses the controlled-comparison concern that abundance and k-mer
pipelines may otherwise be evaluated on different sample subsets. For each task
and round, it restricts both single-modality pipelines to the exact same samples
with both abundance and k-mer profiles available, then applies the same split.

Outputs:
  analysis/matched_intersection_results/{task}_matched_all_rounds.csv
  analysis/matched_intersection_results/{task}_matched_summary.csv
  analysis/matched_intersection_results/matched_statistical_tests.csv
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_ind, wilcoxon
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectFromModel
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
from xgboost import XGBClassifier


ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "matched_intersection_results")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42 + i * 100 for i in range(10)]
RUN_SEEDS = SEEDS
METRICS = ["AUC", "AUPRC", "F1_macro", "MCC"]
CLASSIFIERS = ["LASSO-LR", "Random Forest", "SVM (RBF)", "XGBoost"]
SVM_PROBABILITY = True
FAST_MULTICLASS_SELECTION = False
SKIP_XGBOOST = False
SKIP_LASSO = False
ONLY_CLASSIFIERS = None


def clr_features_from_train(train_df, test_df):
    """Build abundance features using species retained by training prevalence."""
    prev = (train_df > 0).mean(axis=0)
    cols = prev[prev >= 0.10].index
    train = train_df.loc[:, cols].astype(float)
    test = test_df.loc[:, cols].astype(float)
    return build_abundance_features(train), build_abundance_features(test), len(cols)


def build_abundance_features(df):
    values = df.values.astype(np.float32)
    x = values + 1e-6
    log_x = np.log(x)
    clr = log_x - log_x.mean(axis=1, keepdims=True)

    p = values / (values.sum(axis=1, keepdims=True) + 1e-12)
    shannon = -np.sum(p * np.log(np.where(p > 0, p, 1)), axis=1)
    simpson = 1 - np.sum(p ** 2, axis=1)
    richness = (values > 0).sum(axis=1).astype(np.float32)
    presence_weighted = clr * (values > 0).astype(np.float32)

    return np.hstack([
        clr,
        shannon[:, None],
        simpson[:, None],
        richness[:, None],
        presence_weighted,
    ]).astype(np.float32)


def select_abundance_features(x_train, y_train, x_test, seed):
    selector = SelectFromModel(
        LogisticRegression(
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
        selector.fit(x_train, y_train)
    return selector.transform(x_train), selector.transform(x_test), int(selector.get_support().sum())


def select_abundance_features_fast_multiclass(x_train, y_train, x_test, top_n=200):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pvals = f_classif(x_train, y_train)
    pvals = np.nan_to_num(pvals, nan=1.0)
    _, pvals_bh, _, _ = multipletests(pvals, method="fdr_bh")
    ranked = np.argsort(pvals_bh)
    significant = ranked[pvals_bh[ranked] < 0.05]
    if len(significant) >= 10:
        idx = significant[: min(top_n, len(significant))]
    else:
        idx = ranked[: min(top_n, x_train.shape[1])]
    return x_train[:, idx], x_test[:, idx], len(idx)


def select_kmer_binary(x_train, y_train, x_test, top_n=1000):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pvals = ttest_ind(
            x_train[y_train == 0],
            x_train[y_train == 1],
            equal_var=False,
            axis=0,
        )
    pvals = np.nan_to_num(pvals, nan=1.0)
    _, pvals_bh, _, _ = multipletests(pvals, method="fdr_bh")
    idx = np.argsort(pvals_bh)[: min(top_n, x_train.shape[1])]
    return x_train[:, idx], x_test[:, idx], len(idx)


def select_kmer_multiclass(x_train, y_train, x_test, top_n=1000):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pvals = f_classif(x_train, y_train)
    pvals = np.nan_to_num(pvals, nan=1.0)
    _, pvals_bh, _, _ = multipletests(pvals, method="fdr_bh")
    idx = np.argsort(pvals_bh)[: min(top_n, x_train.shape[1])]
    return x_train[:, idx], x_test[:, idx], len(idx)


def welch_pvalues_vectorized(group_a, group_b):
    """Two-sided Welch t-test p-values for each feature column."""
    a = group_a.astype(np.float64, copy=False)
    b = group_b.astype(np.float64, copy=False)
    n_a = a.shape[0]
    n_b = b.shape[0]
    mean_a = a.mean(axis=0)
    mean_b = b.mean(axis=0)
    var_a = a.var(axis=0, ddof=1)
    var_b = b.var(axis=0, ddof=1)
    se2 = var_a / n_a + var_b / n_b
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = (mean_a - mean_b) / np.sqrt(se2)
        df = se2 ** 2 / (
            (var_a / n_a) ** 2 / max(n_a - 1, 1)
            + (var_b / n_b) ** 2 / max(n_b - 1, 1)
        )
        pvals = 2 * student_t.sf(np.abs(t_stat), df)
    return np.nan_to_num(pvals, nan=1.0, posinf=1.0, neginf=1.0)


def make_models(seed, y_train, multiclass=False):
    if multiclass:
        xgb_metric = "mlogloss"
        xgb_extra = {
            "n_estimators": 300,
            "num_class": len(np.unique(y_train)),
            "objective": "multi:softprob",
        }
        lr_extra = {"max_iter": 2000, "n_jobs": 16}
    else:
        xgb_metric = "logloss"
        pos_w = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
        xgb_extra = {"n_estimators": 500, "scale_pos_weight": pos_w}
        lr_extra = {"max_iter": 5000}

    models = {
        "LASSO-LR": LogisticRegression(
            l1_ratio=1.0,
            solver="saga",
            C=0.05,
            class_weight="balanced",
            random_state=seed,
            **lr_extra,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=SVM_PROBABILITY,
            class_weight="balanced",
            random_state=seed,
        ),
        "XGBoost": XGBClassifier(
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.5,
            min_child_weight=3,
            eval_metric=xgb_metric,
            tree_method="hist",
            random_state=seed,
            n_jobs=16,
            **xgb_extra,
        ),
    }
    if SKIP_XGBOOST:
        models.pop("XGBoost")
    if SKIP_LASSO:
        models.pop("LASSO-LR")
    if ONLY_CLASSIFIERS:
        keep = set(ONLY_CLASSIFIERS)
        models = {name: model for name, model in models.items() if name in keep}
    return models


def evaluate_models(x_train, x_test, y_train, y_test, task, pipeline, round_num, seed, multiclass=False):
    rows = []
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    for name, clf in make_models(seed, y_train, multiclass=multiclass).items():
        started = time.time()
        print(f"    Training {pipeline} / {name}...", flush=True)
        use_scaled = name in {"LASSO-LR", "SVM (RBF)"}
        xtr = x_train_scaled if use_scaled else x_train
        xte = x_test_scaled if use_scaled else x_test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(xtr, y_train)
        y_pred = clf.predict(xte)
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(xte)
        else:
            y_score = clf.decision_function(xte)

        if multiclass:
            auc, auprc = multiclass_auc_auprc(y_test, y_score)
        else:
            if np.ndim(y_score) == 2:
                y_score_1d = y_score[:, 1]
            else:
                y_score_1d = y_score
            auc = roc_auc_score(y_test, y_score_1d)
            auprc = average_precision_score(y_test, y_score_1d)
        rows.append({
            "task": task,
            "round": round_num,
            "seed": seed,
            "pipeline": pipeline,
            "classifier": name,
            "AUC": round(float(auc), 4),
            "AUPRC": round(float(auprc), 4),
            "F1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
            "MCC": round(float(matthews_corrcoef(y_test, y_pred)), 4),
        })
        print(
            f"    Finished {pipeline} / {name} in {time.time() - started:.1f}s",
            flush=True,
        )
    return rows


def multiclass_auc_auprc(y_test, y_prob):
    auc_values = []
    auprc_values = []
    for cls in np.unique(y_test):
        if cls >= y_prob.shape[1]:
            continue
        y_bin = (y_test == cls).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        auc_values.append(roc_auc_score(y_bin, y_prob[:, cls]))
        auprc_values.append(average_precision_score(y_bin, y_prob[:, cls]))
    return float(np.mean(auc_values)), float(np.mean(auprc_values))


def load_hmp2_abundance_raw():
    data_dir = os.path.join(ROOT, "../ibd/data/raw")
    meta = pd.read_csv(os.path.join(data_dir, "hmp2_metadata_2018-08-20.csv"), low_memory=False)
    meta = meta[meta["data_type"] == "metagenomics"]
    meta = meta[meta["diagnosis"].isin(["CD", "UC", "nonIBD"])]
    meta["label"] = meta["diagnosis"].replace({"CD": "IBD", "UC": "IBD"})
    meta = (
        meta[["External ID", "Participant ID", "label"]]
        .drop_duplicates("External ID")
        .rename(columns={"External ID": "sample_id", "Participant ID": "participant_id"})
        .set_index("sample_id")
    )

    taxon = pd.read_csv(os.path.join(data_dir, "taxonomic_profiles_3.tsv.gz"), sep="\t", index_col=0)
    sp_mask = taxon.index.map(lambda x: x.count("|") == 6 and "s__" in x and "t__" not in x)
    species = taxon[sp_mask].copy()
    species.index = species.index.map(lambda x: x.split("|")[-1])
    species = species.T
    species.index = species.index.str.replace("_profile$", "", regex=True)
    species.index.name = "sample_id"
    species = species.astype(float)

    shared = meta.index.intersection(species.index)
    return species.loc[shared], meta.loc[shared]


def load_hmp2_kmer():
    kmer_dir = os.path.join(ROOT, "../ibd/kmer_counts")
    x = np.load(os.path.join(kmer_dir, "kmer_matrix_k8_X.npy")).astype(np.float32)
    ids = np.load(os.path.join(kmer_dir, "kmer_matrix_k8_sample_ids.npy"), allow_pickle=True)
    labels = np.load(os.path.join(kmer_dir, "kmer_matrix_k8_labels.npy"), allow_pickle=True)
    parts = np.load(os.path.join(kmer_dir, "kmer_matrix_k8_participant.npy"), allow_pickle=True)

    _, uniq = np.unique(ids, return_index=True)
    x = x[uniq]
    ids = ids[uniq]
    labels = labels[uniq]
    parts = parts[uniq]
    row_sums = x.sum(axis=1, keepdims=True)
    x = x / np.where(row_sums == 0, 1, row_sums)
    meta = pd.DataFrame({
        "sample_id": ids,
        "label": labels,
        "participant_id": parts,
    }).set_index("sample_id")
    return x, meta


def run_ibd():
    abund, meta_ab = load_hmp2_abundance_raw()
    x_kmer, meta_km = load_hmp2_kmer()
    shared = meta_ab.index.intersection(meta_km.index)
    kmer_pos = pd.Series(np.arange(len(meta_km)), index=meta_km.index).loc[shared].values

    abund = abund.loc[shared]
    x_kmer = x_kmer[kmer_pos]
    meta = meta_ab.loc[shared].copy()
    y_all = (meta["label"].values == "IBD").astype(int)
    parts = meta["participant_id"].values

    print(f"IBD matched samples: n={len(shared)}, IBD={y_all.sum()}, nonIBD={(y_all == 0).sum()}", flush=True)
    rows = []
    for round_num, seed in enumerate(RUN_SEEDS, 1):
        unique_parts = pd.Series(parts).drop_duplicates().values
        part_labels = pd.Series(y_all, index=parts).groupby(level=0).first()
        parts_train, parts_test = train_test_split(
            unique_parts,
            test_size=0.2,
            random_state=seed,
            stratify=part_labels.loc[unique_parts].values,
        )
        train_mask = np.isin(parts, parts_train)
        test_mask = np.isin(parts, parts_test)
        rows.extend(run_matched_round(
            task="IBD",
            round_num=round_num,
            seed=seed,
            abund_train=abund.iloc[train_mask],
            abund_test=abund.iloc[test_mask],
            kmer_train=x_kmer[train_mask],
            kmer_test=x_kmer[test_mask],
            y_train=y_all[train_mask],
            y_test=y_all[test_mask],
            multiclass=False,
        ))
    return pd.DataFrame(rows)


def c_md_paths(task):
    if task == "obesity":
        return {
            "task_label": "Obesity",
            "splits_dir": os.path.join(ROOT, "../splits/obesity"),
            "abundance": os.path.join(ROOT, "../obesity/data/cmd3_relative_abundance.csv"),
            "metadata_pattern": "obesity_metadata_round{}.csv",
            "kmer_dir": os.path.join(ROOT, "../obesity/data/kmer_counts_k8"),
        }
    if task == "geography":
        return {
            "task_label": "Geography",
            "splits_dir": os.path.join(ROOT, "../splits/geography"),
            "abundance": os.path.join(ROOT, "../geography/data/cmd3_geography_abundance.csv"),
            "metadata_pattern": "geo_metadata_round{}.csv",
            "kmer_dir": os.path.join(ROOT, "../geography/data/kmer_counts_k8"),
        }
    raise ValueError(task)


def run_cmd_task(task):
    cfg = c_md_paths(task)
    abund_full = pd.read_csv(cfg["abundance"], index_col="sample_id")
    kmer_paths = {
        f.replace("_k8.npy", ""): os.path.join(cfg["kmer_dir"], f)
        for f in os.listdir(cfg["kmer_dir"])
        if f.endswith("_k8.npy")
    }
    kmer_cache = {}
    rows = []
    for round_num, seed in enumerate(SEEDS, 1):
        meta = pd.read_csv(os.path.join(cfg["splits_dir"], cfg["metadata_pattern"].format(round_num)))
        shared = [
            sid for sid in meta["sample_id"].values
            if sid in abund_full.index and sid in kmer_paths
        ]
        meta_s = meta.set_index("sample_id").loc[shared]
        train_ids = meta_s.index[meta_s["split"] == "train"]
        test_ids = meta_s.index[meta_s["split"] == "test"]

        if task == "geography":
            encoder = LabelEncoder()
            encoder.fit(meta_s.loc[train_ids, "label"].values)
            test_seen = np.isin(meta_s.loc[test_ids, "label"].values, encoder.classes_)
            test_ids = test_ids[test_seen]
            y_train = encoder.transform(meta_s.loc[train_ids, "label"].values)
            y_test = encoder.transform(meta_s.loc[test_ids, "label"].values)
            multiclass = True
        else:
            y_train = meta_s.loc[train_ids, "label"].values.astype(int)
            y_test = meta_s.loc[test_ids, "label"].values.astype(int)
            multiclass = False

        print(
            f"{cfg['task_label']} round {round_num}: matched n={len(meta_s)}, "
            f"train={len(train_ids)}, test={len(test_ids)}"
        , flush=True)

        print("  Loading matched k-mer vectors...", flush=True)
        load_started = time.time()
        x_kmer_train = load_kmer_stack(train_ids, kmer_paths, kmer_cache)
        x_kmer_test = load_kmer_stack(test_ids, kmer_paths, kmer_cache)
        print(
            f"  Loaded k-mer matrices in {time.time() - load_started:.1f}s "
            f"(cache size={len(kmer_cache)})",
            flush=True,
        )

        rows.extend(run_matched_round(
            task=cfg["task_label"],
            round_num=round_num,
            seed=seed,
            abund_train=abund_full.loc[train_ids],
            abund_test=abund_full.loc[test_ids],
            kmer_train=x_kmer_train,
            kmer_test=x_kmer_test,
            y_train=y_train,
            y_test=y_test,
            multiclass=multiclass,
        ))
    return pd.DataFrame(rows)


def load_kmer_stack(sample_ids, kmer_paths, cache=None):
    vectors = []
    for sid in sample_ids:
        if cache is not None and sid in cache:
            vec = cache[sid]
        else:
            vec = np.load(kmer_paths[sid]).astype(np.float32)
            total = vec.sum()
            if total > 0:
                vec = vec / total
            if cache is not None:
                cache[sid] = vec
        vectors.append(vec)
    return np.vstack(vectors).astype(np.float32)


def run_matched_round(task, round_num, seed, abund_train, abund_test,
                      kmer_train, kmer_test, y_train, y_test, multiclass):
    rows = []
    feature_started = time.time()
    ab_started = time.time()
    x_ab_train, x_ab_test, n_species = clr_features_from_train(abund_train, abund_test)
    if multiclass and FAST_MULTICLASS_SELECTION:
        x_ab_train, x_ab_test, n_ab_features = select_abundance_features_fast_multiclass(
            x_ab_train, y_train, x_ab_test, top_n=200
        )
    else:
        x_ab_train, x_ab_test, n_ab_features = select_abundance_features(
            x_ab_train, y_train, x_ab_test, seed
        )
    print(
        f"  Abundance feature selection finished in {time.time() - ab_started:.1f}s",
        flush=True,
    )
    km_started = time.time()
    if multiclass:
        x_km_train, x_km_test, n_kmer_features = select_kmer_multiclass(
            kmer_train, y_train, kmer_test, top_n=1000
        )
    else:
        x_km_train, x_km_test, n_kmer_features = select_kmer_binary(
            kmer_train, y_train, kmer_test, top_n=1000
        )
    print(
        f"  K-mer feature selection finished in {time.time() - km_started:.1f}s",
        flush=True,
    )

    print(
        f"  {task} R{round_num}: {n_species} retained species, "
        f"{n_ab_features} selected abundance features, "
        f"{n_kmer_features} selected k-mers "
        f"(feature selection {time.time() - feature_started:.1f}s)"
    , flush=True)

    rows.extend(evaluate_models(
        x_ab_train, x_ab_test, y_train, y_test,
        task, "Abundance matched", round_num, seed, multiclass=multiclass,
    ))
    rows.extend(evaluate_models(
        x_km_train, x_km_test, y_train, y_test,
        task, "K-mer matched", round_num, seed, multiclass=multiclass,
    ))
    return rows


def summarize(df, task):
    rows = []
    for (pipeline, classifier), group in df.groupby(["pipeline", "classifier"]):
        row = {"task": task, "pipeline": pipeline, "classifier": classifier}
        for metric in METRICS:
            row[f"{metric}_mean"] = round(group[metric].mean(), 4)
            row[f"{metric}_std"] = round(group[metric].std(), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def matched_wilcoxon(df):
    rows = []
    for task in sorted(df["task"].unique()):
        task_df = df[df["task"] == task]
        for classifier in CLASSIFIERS:
            for metric in METRICS:
                a_sub = task_df[
                    (task_df["pipeline"] == "Abundance matched")
                    & (task_df["classifier"] == classifier)
                ][["round", metric]].rename(columns={metric: "a"})
                k_sub = task_df[
                    (task_df["pipeline"] == "K-mer matched")
                    & (task_df["classifier"] == classifier)
                ][["round", metric]].rename(columns={metric: "k"})
                paired = a_sub.merge(k_sub, on="round")
                if len(paired) == 0:
                    continue
                abund = paired["a"].values
                kmer  = paired["k"].values
                try:
                    stat, p_raw = wilcoxon(kmer, abund, alternative="greater")
                except TypeError:
                    stat, p_two = wilcoxon(kmer, abund)
                    p_raw = p_two / 2 if np.nanmean(kmer - abund) > 0 else 1 - p_two / 2
                rows.append({
                    "task": task,
                    "classifier": classifier,
                    "metric": metric,
                    "mean_kmer": round(float(np.nanmean(kmer)), 4),
                    "mean_abundance": round(float(np.nanmean(abund)), 4),
                    "delta": round(float(np.nanmean(kmer - abund)), 4),
                    "W_stat": round(float(stat), 4),
                    "p_raw": float(p_raw),
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        _, p_adj, _, _ = multipletests(out["p_raw"].values, method="fdr_bh")
        out["p_adj"] = p_adj
        out["significant"] = out["p_adj"] < 0.05
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["ibd", "obesity", "geography", "all"],
        default="all",
        help="Task to run. Use 'all' to run all matched-intersection benchmarks.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of rounds to run from the standard seed list. Default: 10.",
    )
    parser.add_argument(
        "--svm-decision-scores",
        action="store_true",
        help=(
            "Use SVM decision_function scores instead of probability calibration. "
            "Useful for the 30-class geography matched-control run; AUC/AUPRC "
            "are rank-based and accept decision scores."
        ),
    )
    parser.add_argument(
        "--fast-multiclass-selection",
        action="store_true",
        help=(
            "For multiclass matched-control runs, use training-set ANOVA F-test "
            "feature selection for abundance instead of ElasticNet. This is used "
            "only as a computationally tractable geography matched-control."
        ),
    )
    parser.add_argument(
        "--skip-xgboost",
        action="store_true",
        help="Skip XGBoost for computationally heavy matched-control runs.",
    )
    parser.add_argument(
        "--skip-lasso",
        action="store_true",
        help="Skip LASSO-LR for computationally heavy matched-control runs.",
    )
    parser.add_argument(
        "--only-classifiers",
        nargs="+",
        choices=CLASSIFIERS,
        help="Run only the specified classifier names.",
    )
    args = parser.parse_args()
    global RUN_SEEDS
    global SVM_PROBABILITY
    global FAST_MULTICLASS_SELECTION
    global SKIP_XGBOOST
    global SKIP_LASSO
    global ONLY_CLASSIFIERS
    RUN_SEEDS = SEEDS[: args.rounds]
    SVM_PROBABILITY = not args.svm_decision_scores
    FAST_MULTICLASS_SELECTION = args.fast_multiclass_selection
    SKIP_XGBOOST = args.skip_xgboost
    SKIP_LASSO = args.skip_lasso
    ONLY_CLASSIFIERS = args.only_classifiers

    runners = {
        "ibd": run_ibd,
        "obesity": lambda: run_cmd_task("obesity"),
        "geography": lambda: run_cmd_task("geography"),
    }
    selected = list(runners) if args.task == "all" else [args.task]

    all_frames = []
    for task in selected:
        print(f"\n{'=' * 72}\nRunning matched-intersection benchmark: {task}\n{'=' * 72}", flush=True)
        df = runners[task]()
        all_frames.append(df)
        df.to_csv(os.path.join(OUT_DIR, f"{task}_matched_all_rounds.csv"), index=False)
        summary = summarize(df, task)
        summary.to_csv(os.path.join(OUT_DIR, f"{task}_matched_summary.csv"), index=False)
        print(summary.to_string(index=False))

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(os.path.join(OUT_DIR, "matched_all_tasks_all_rounds.csv"), index=False)
    stats = matched_wilcoxon(combined)
    stats.to_csv(os.path.join(OUT_DIR, "matched_statistical_tests.csv"), index=False)
    print(f"\nSaved matched benchmark outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
