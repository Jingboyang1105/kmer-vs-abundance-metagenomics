"""
Step 2c – Fusion pipeline: Abundance + K-mer combined features (Obesity task).
Tests whether k-mer features capture complementary information beyond abundance.

For each round, uses IDENTICAL sample splits as 02a/02b.
Feature construction:
  Abundance: CLR-transformed species + alpha diversity + presence-weighted CLR
             → ElasticNet feature selection (same as 02a)
  K-mer:     canonical k=8 frequency counts
             → Welch t-test BH top-500 selection (half of 02b's 1000)
  Combined:  concatenate selected abundance + selected k-mer features

Output: results/fusion_all_rounds.csv
        results/figures/fusion_comparison.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef,
)
from xgboost import XGBClassifier

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "../data")
SPLITS_DIR  = os.path.join(SCRIPT_DIR, "../splits/obesity")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

ABUNDANCE_CSV = os.path.join(DATA_DIR, "cmd3_relative_abundance.csv")
KMER_DIR      = os.path.join(SCRIPT_DIR, "../../obesity/data/kmer_counts_k8")
N_ROUNDS      = 10
TOP_KMER      = 500   # half of k-mer-only pipeline, to keep total features comparable


# ── Abundance feature construction (identical to 02a) ──────────────────────

def clr_transform(df):
    X = df.values + 1e-6
    log_X = np.log(X)
    return pd.DataFrame(log_X - log_X.mean(axis=1, keepdims=True),
                        index=df.index, columns=df.columns)


def build_abundance_features(abund):
    X_clr = clr_transform(abund)
    X_vals = abund.values
    p = X_vals / (X_vals.sum(axis=1, keepdims=True) + 1e-12)
    shannon  = -np.sum(p * np.log(np.where(p > 0, p, 1)), axis=1)
    simpson  = 1 - np.sum(p ** 2, axis=1)
    richness = (X_vals > 0).sum(axis=1).astype(float)
    X_div = pd.DataFrame({"alpha_shannon": shannon, "alpha_simpson": simpson,
                           "alpha_richness": richness}, index=abund.index)
    pw = X_clr.values * (abund.values > 0).astype(float)
    X_pw = pd.DataFrame(pw, index=abund.index,
                        columns=[f"pw__{c}" for c in X_clr.columns])
    return pd.concat([X_clr, X_div, X_pw], axis=1)


# ── K-mer feature selection (Welch t-test BH, top N) ──────────────────────

def ttest_select(X_train, y_train, top_n):
    cls0 = X_train[y_train == 0]
    cls1 = X_train[y_train == 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pvals = ttest_ind(cls0, cls1, equal_var=False)
    pvals = np.nan_to_num(pvals, nan=1.0)
    _, pvals_bh, _, _ = multipletests(pvals, method="fdr_bh")
    order = np.argsort(pvals_bh)
    return order[:top_n]


# ── Main round function ─────────────────────────────────────────────────────

def run_one_round(meta, abund_full, kmer_mat, kmer_ids, round_num):
    # ── Abundance side ──
    shared_ab = meta["sample_id"][meta["sample_id"].isin(abund_full.index)]
    meta_ab   = meta[meta["sample_id"].isin(shared_ab)].set_index("sample_id")
    abund     = abund_full.loc[shared_ab]
    prev      = (abund > 0).mean()
    abund     = abund.loc[:, prev >= 0.10]
    X_ab_full = build_abundance_features(abund)

    # ── K-mer side ──
    shared_km = meta["sample_id"][meta["sample_id"].isin(kmer_ids)]
    meta_km   = meta[meta["sample_id"].isin(shared_km)].set_index("sample_id")

    # Use intersection of both
    shared = meta_ab.index.intersection(meta_km.index)
    meta_s = meta_ab.loc[shared]

    X_ab = X_ab_full.loc[shared].values
    km_idx = [list(kmer_ids).index(s) for s in shared]
    X_km = kmer_mat[km_idx]

    train_idx = meta_s.index[meta_s["split"] == "train"]
    test_idx  = meta_s.index[meta_s["split"] == "test"]
    y_train = meta_s.loc[train_idx, "label"].values
    y_test  = meta_s.loc[test_idx,  "label"].values

    tr_pos = [list(shared).index(s) for s in train_idx]
    te_pos = [list(shared).index(s) for s in test_idx]

    X_ab_tr, X_ab_te = X_ab[tr_pos], X_ab[te_pos]
    X_km_tr, X_km_te = X_km[tr_pos], X_km[te_pos]

    # Abundance feature selection (ElasticNet, same as 02a)
    selector = SelectFromModel(
        LogisticRegression(l1_ratio=0.5, solver="saga", C=0.1,
                           max_iter=3000, class_weight="balanced", random_state=42),
        threshold="mean",
    )
    selector.fit(X_ab_tr, y_train)
    X_ab_tr_sel = selector.transform(X_ab_tr)
    X_ab_te_sel = selector.transform(X_ab_te)

    # K-mer feature selection (t-test BH top-500)
    km_idx_sel = ttest_select(X_km_tr, y_train, TOP_KMER)
    X_km_tr_sel = X_km_tr[:, km_idx_sel]
    X_km_te_sel = X_km_te[:, km_idx_sel]

    # Concatenate
    X_fused_tr = np.hstack([X_ab_tr_sel, X_km_tr_sel])
    X_fused_te = np.hstack([X_ab_te_sel, X_km_te_sel])

    n_ab = X_ab_tr_sel.shape[1]
    n_km = X_km_tr_sel.shape[1]
    print(f"  Round {round_num}: {n_ab} abundance + {n_km} k-mer = {n_ab+n_km} features | "
          f"train={len(y_train)} test={len(y_test)}")

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_fused_tr)
    X_te_sc = scaler.transform(X_fused_te)

    pos_w = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
    models = {
        "LASSO-LR": LogisticRegression(
            l1_ratio=1, solver="saga", C=0.05,
            class_weight="balanced", max_iter=2000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"),
        "SVM (RBF)": SVC(
            kernel="rbf", C=1.0, gamma="scale", random_state=42,
            class_weight="balanced", probability=True),
        "XGBoost": XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5, min_child_weight=3,
            scale_pos_weight=pos_w, eval_metric="logloss",
            random_state=42, n_jobs=-1),
    }

    rows = []
    for name, clf in models.items():
        sc = name in ("SVM (RBF)", "LASSO-LR")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr_sc if sc else X_fused_tr, y_train)
        y_pred = clf.predict(X_te_sc if sc else X_fused_te)
        y_prob = clf.predict_proba(X_te_sc if sc else X_fused_te)[:, 1]
        auc   = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        f1    = f1_score(y_test, y_pred, average="macro")
        mcc   = matthews_corrcoef(y_test, y_pred)
        print(f"  R{round_num} {name}: AUC={auc:.3f}  F1={f1:.3f}  MCC={mcc:.3f}")
        rows.append({
            "round": round_num, "pipeline": "Fusion (Abundance+K-mer)",
            "classifier": name,
            "AUC":      round(auc,   4),
            "AUPRC":    round(auprc, 4),
            "F1_macro": round(f1,    4),
            "MCC":      round(mcc,   4),
        })
    return rows


def plot_comparison(summary):
    metrics     = ["AUC", "AUPRC", "F1_macro", "MCC"]
    pipelines   = ["Abundance (Line1)", "K-mer (Line2)", "Fusion (Abundance+K-mer)"]
    classifiers = sorted(summary["classifier"].unique())
    colors = {
        "Abundance (Line1)":        "steelblue",
        "K-mer (Line2)":            "darkorange",
        "Fusion (Abundance+K-mer)": "mediumseagreen",
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    x = np.arange(len(classifiers))
    w = 0.25
    for ax, m in zip(axes, metrics):
        for i, pipe in enumerate(pipelines):
            sub = summary[summary["pipeline"] == pipe].set_index("classifier")
            means = [sub.loc[c, f"{m}_mean"] if c in sub.index else 0 for c in classifiers]
            stds  = [sub.loc[c, f"{m}_std"]  if c in sub.index else 0 for c in classifiers]
            ax.bar(x + (i - 1) * w, means, w, yerr=stds, capsize=4,
                   label=pipe, color=colors[pipe], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(classifiers, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1); ax.set_title(m, fontsize=11)
        ax.axhline(0.5, color="gray", ls="--", lw=0.8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.08), fontsize=9)
    fig.suptitle(f"Obesity: Abundance vs K-mer vs Fusion\n"
                 f"({N_ROUNDS}-round mean ± SD)", fontsize=11, y=1.02)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fusion_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure → {out}")


def main():
    print("Loading abundance matrix...")
    abund_full = pd.read_csv(ABUNDANCE_CSV, index_col=0)
    print(f"  {abund_full.shape[0]} samples × {abund_full.shape[1]} species")

    print("Loading k-mer matrix...")
    npy_files = sorted([f for f in os.listdir(KMER_DIR) if f.endswith("_k8.npy")])
    kmer_ids  = [f.replace("_k8.npy", "") for f in npy_files]
    kmer_mat  = np.vstack([np.load(os.path.join(KMER_DIR, f)) for f in npy_files])
    # Normalize to frequencies
    row_sums = kmer_mat.sum(axis=1, keepdims=True)
    kmer_mat = kmer_mat / np.where(row_sums == 0, 1, row_sums)
    print(f"  {kmer_mat.shape[0]} samples × {kmer_mat.shape[1]} k-mers")

    all_rows = []
    for r in range(1, N_ROUNDS + 1):
        print(f"\n── Round {r} ──────────────")
        meta_path = os.path.join(SPLITS_DIR, f"obesity_metadata_round{r}.csv")
        meta = pd.read_csv(meta_path)
        rows = run_one_round(meta, abund_full, kmer_mat, kmer_ids, r)
        all_rows.extend(rows)
        pd.DataFrame(rows).to_csv(
            os.path.join(RESULTS_DIR, f"fusion_round{r}.csv"), index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "fusion_all_rounds.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"Obesity – Fusion Pipeline – {N_ROUNDS}-round summary")
    print(f"{'='*60}")
    rows_sum = []
    for clf, grp in df.groupby("classifier"):
        row = {"pipeline": "Fusion (Abundance+K-mer)", "classifier": clf}
        for m in ["AUC", "AUPRC", "F1_macro", "MCC"]:
            row[f"{m}_mean"] = round(grp[m].mean(), 4)
            row[f"{m}_std"]  = round(grp[m].std(),  4)
        rows_sum.append(row)
        print(f"  {clf:15s} AUC={row['AUC_mean']:.3f}±{row['AUC_std']:.3f}  "
              f"AUPRC={row['AUPRC_mean']:.3f}±{row['AUPRC_std']:.3f}  "
              f"F1={row['F1_macro_mean']:.3f}±{row['F1_macro_std']:.3f}  "
              f"MCC={row['MCC_mean']:.3f}±{row['MCC_std']:.3f}")

    # Load existing results for comparison plot
    ab_df = pd.read_csv(os.path.join(RESULTS_DIR, "abundance_all_rounds.csv"))
    ab_df = ab_df.rename(columns={"pipeline": "pipeline"})
    ab_df["pipeline"] = "Abundance (Line1)"
    km_df = pd.read_csv(os.path.join(RESULTS_DIR, "kmer_all_rounds.csv"))
    km_df["pipeline"] = "K-mer (Line2)"

    all_df = pd.concat([ab_df, km_df, df], ignore_index=True)
    summary_rows = []
    for (pipe, clf), grp in all_df.groupby(["pipeline", "classifier"]):
        row = {"pipeline": pipe, "classifier": clf}
        for m in ["AUC", "AUPRC", "F1_macro", "MCC"]:
            row[f"{m}_mean"] = round(grp[m].mean(), 4)
            row[f"{m}_std"]  = round(grp[m].std(),  4)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(RESULTS_DIR, "summary_fusion.csv"), index=False)
    plot_comparison(summary)


if __name__ == "__main__":
    main()
