"""
Step 2b – K-mer pipeline for obesity (BMI classification), 10-round subsampling.
Input:  splits/obesity/obesity_metadata_round{1..10}.csv  (from metadata_prep.py)
        FASTQ files downloaded via SRA (streaming: download→count→delete)
Output: data/kmer_counts_k8/  (per-sample .npy cache, shared across rounds)
        results/kmer_round{1..10}.csv
        results/kmer_all_rounds.csv   (all rounds combined)
        results/results_kmer.csv      (mean ± SD summary)
"""

import os
import sys
import gzip
import subprocess
import shutil
import tempfile
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "../data")
SPLITS_DIR  = os.path.join(SCRIPT_DIR, "../splits/obesity")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
KMER_DIR    = os.path.join(DATA_DIR, "kmer_counts_k8")
FASTQ_TMP   = os.path.join(tempfile.gettempdir(), "obesity_fastq_tmp")  # OS temp dir, not a hardcoded /tmp
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(KMER_DIR, exist_ok=True)
os.makedirs(FASTQ_TMP, exist_ok=True)

FASTQ_DUMP   = "fastq-dump"   # update to full path if not on PATH
K            = 8
TOP_K        = 1000
N_ROUNDS     = 10
MAX_READS    = 200_000   # reads per sample (~70MB FASTQ, enough for k8)
DL_WORKERS   = 4         # parallel download + count threads


# ── k-mer counting ────────────────────────────────────────────────────

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Lookup: ASCII → base index (0-3), 255 = invalid
_ENCODE = np.full(256, 255, dtype=np.uint8)
for _c, _i in zip(b"ACGTacgt", [0, 1, 2, 3, 0, 1, 2, 3]):
    _ENCODE[_c] = _i


if HAS_NUMBA:
    from numba import njit

    @njit
    def _count_kmers_numba(seq_array: np.ndarray, k: int,
                            encode: np.ndarray, counts: np.ndarray):
        L = len(seq_array)
        for pos in range(L - k + 1):
            fwd = np.int64(0)
            rev = np.int64(0)
            ok  = True
            for j in range(k):
                b = encode[seq_array[pos + j]]
                if b == 255:
                    ok = False
                    break
                fwd = fwd * 4 + np.int64(b)
                b_rev = encode[seq_array[pos + k - 1 - j]]
                if b_rev == 255:
                    ok = False
                    break
                rev = rev * 4 + np.int64(3 - b_rev)
            if ok:
                canon = fwd if fwd < rev else rev
                counts[canon] += 1.0


def count_kmers(fastq_path: str, k: int = 8, max_reads: int = 200_000) -> np.ndarray:
    n = 4 ** k
    counts = np.zeros(n, dtype=np.float64)
    opener = gzip.open if fastq_path.endswith(".gz") else open
    n_reads = 0
    with opener(fastq_path, "rt") as fh:
        for i, line in enumerate(fh):
            if i % 4 != 1:
                continue
            seq = np.frombuffer(line.strip().encode(), dtype=np.uint8)
            if HAS_NUMBA:
                _count_kmers_numba(seq, k, _ENCODE, counts)
            else:
                enc = _ENCODE[seq]
                for pos in range(len(enc) - k + 1):
                    w = enc[pos:pos + k]
                    if 255 in w:
                        continue
                    fwd = rev = 0
                    for j in range(k):
                        fwd = fwd * 4 + int(w[j])
                        rev = rev * 4 + (3 - int(w[k - 1 - j]))
                    counts[min(fwd, rev)] += 1
            n_reads += 1
            if n_reads >= max_reads:
                break
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts.astype(np.float32)


def download_and_count(accession: str, sample_id: str) -> np.ndarray | None:
    """Download first MAX_READS reads via fastq-dump, count k-mers, delete FASTQ."""
    cache = os.path.join(KMER_DIR, f"{sample_id}_k{K}.npy")
    if os.path.exists(cache):
        return np.load(cache)

    out_dir = os.path.join(FASTQ_TMP, accession)
    os.makedirs(out_dir, exist_ok=True)
    try:
        result = subprocess.run(
            [FASTQ_DUMP, accession,
             "--outdir", out_dir,
             "--split-files",
             "--maxSpotId", str(MAX_READS),
             "--skip-technical",
             "-Q", "0"],      # omit quality scores to halve file size
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            print(f"  fastq-dump failed for {accession}: {result.stderr[:200]}")
            return None

        fastqs = sorted(f for f in os.listdir(out_dir) if f.endswith(".fastq"))
        if not fastqs:
            return None

        # Use read 1 (or the only file if single-end)
        vec = count_kmers(os.path.join(out_dir, fastqs[0]), k=K, max_reads=MAX_READS)
        np.save(cache, vec)
        return vec
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {accession}")
        return None
    except Exception as e:
        print(f"  ERROR {accession}: {e}")
        return None
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def build_kmer_matrix(meta: pd.DataFrame) -> pd.DataFrame:
    """Build k-mer matrix: load from cache or download+count in parallel."""
    needed = meta[["sample_id", "NCBI_accession"]].dropna(subset=["NCBI_accession"]).copy()
    # Some NCBI_accession fields contain semicolon-separated run IDs; use only the first
    needed["NCBI_accession"] = needed["NCBI_accession"].str.split(";").str[0].str.strip()
    to_download = [(row["sample_id"], row["NCBI_accession"])
                   for _, row in needed.iterrows()
                   if not os.path.exists(os.path.join(KMER_DIR, f"{row['sample_id']}_k{K}.npy"))]
    cached = [row["sample_id"] for _, row in needed.iterrows()
              if os.path.exists(os.path.join(KMER_DIR, f"{row['sample_id']}_k{K}.npy"))]

    print(f"  Samples cached: {len(cached)}  |  Need download: {len(to_download)}")

    if to_download:
        done = 0
        total_dl = len(to_download)
        with ThreadPoolExecutor(max_workers=DL_WORKERS) as exe:
            futures = {exe.submit(download_and_count, acc, sid): sid
                       for sid, acc in to_download}
            for fut in as_completed(futures):
                sid = futures[fut]
                done += 1
                result = fut.result()
                status = "OK" if result is not None else "FAIL"
                print(f"  [{done}/{total_dl}] {sid}: {status}", flush=True)

    # Load all from cache
    vectors, sample_ids = [], []
    for _, row in needed.iterrows():
        sid   = row["sample_id"]
        cache = os.path.join(KMER_DIR, f"{sid}_k{K}.npy")
        if os.path.exists(cache):
            vectors.append(np.load(cache))
            sample_ids.append(sid)

    if not vectors:
        return pd.DataFrame()
    df = pd.DataFrame(np.vstack(vectors), index=sample_ids,
                      columns=[f"k{i}" for i in range(4 ** K)])
    return df[~df.index.duplicated(keep="first")]


# ── Feature selection + classification ───────────────────────────────

def select_kmers(X_train, y_train, feat_names):
    _, pvals = stats.ttest_ind(X_train[y_train == 1], X_train[y_train == 0],
                               equal_var=False)
    _, adj, _, _ = multipletests(pvals, method="fdr_bh")
    sel = pd.Series(adj, index=feat_names)
    top = sel[sel < 0.05].sort_values().head(TOP_K)
    if len(top) < 10:
        top = sel.sort_values().head(TOP_K)
    return top.index.tolist()


def run_one_round(meta, kmer_full, round_num):
    shared = meta["sample_id"][meta["sample_id"].isin(kmer_full.index)].tolist()
    meta   = meta[meta["sample_id"].isin(shared)].set_index("sample_id")
    kmer   = kmer_full.loc[shared]

    train_idx = meta.index[meta["split"] == "train"]
    test_idx  = meta.index[meta["split"] == "test"]
    y_train = meta.loc[train_idx, "label"].values
    y_test  = meta.loc[test_idx,  "label"].values
    groups_raw = meta.loc[train_idx, "subject_id"].values
    groups = np.where(pd.isna(groups_raw),
                      np.arange(len(groups_raw)).astype(str), groups_raw)

    feat_names = list(kmer.columns)
    X_tr_all   = kmer.loc[train_idx].values
    X_te_all   = kmer.loc[test_idx].values

    selected = select_kmers(X_tr_all, y_train, feat_names)
    sel_idx  = [feat_names.index(f) for f in selected]
    X_train  = X_tr_all[:, sel_idx]
    X_test   = X_te_all[:, sel_idx]
    print(f"  Round {round_num}: {len(selected)} k-mers selected, "
          f"train={len(y_train)} test={len(y_test)}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

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
        clf.fit(X_train_sc if sc else X_train, y_train)
        y_pred = clf.predict(X_test_sc if sc else X_test)
        y_prob = (clf.predict_proba(X_test_sc if sc else X_test)[:, 1]
                  if hasattr(clf, "predict_proba")
                  else clf.decision_function(X_test_sc if sc else X_test))
        rows.append({
            "round": round_num, "pipeline": "k-mer (Line2-style)",
            "classifier": name,
            "AUC":    round(roc_auc_score(y_test, y_prob), 4),
            "AUPRC":  round(average_precision_score(y_test, y_prob), 4),
            "F1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
            "MCC":    round(matthews_corrcoef(y_test, y_pred), 4),
        })
        print(f"  R{round_num} {name}: AUC={rows[-1]['AUC']:.3f}  "
              f"F1={rows[-1]['F1_macro']:.3f}  MCC={rows[-1]['MCC']:.3f}")
    return pd.DataFrame(rows)


def plot_final(summary):
    classifiers = summary["classifier"].unique()
    metrics = ["AUC", "AUPRC", "F1_macro", "MCC"]
    x = np.arange(len(classifiers))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    for ax, m in zip(axes, metrics):
        means = [summary[summary["classifier"] == c][m].mean() for c in classifiers]
        stds  = [summary[summary["classifier"] == c][m].std()  for c in classifiers]
        ax.bar(x, means, yerr=stds, capsize=4, color="darkorange", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(classifiers, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1); ax.set_title(m); ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    fig.suptitle(f"Obesity – K-mer Pipeline ({N_ROUNDS}-round mean ± SD)", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"kmer_{N_ROUNDS}round_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    # Collect all unique samples across rounds for k-mer building
    all_meta_rows = []
    for r in range(1, N_ROUNDS + 1):
        p = os.path.join(SPLITS_DIR, f"obesity_metadata_round{r}.csv")
        if not os.path.exists(p):
            print(f"Missing {p} – run 01_metadata_prep.py first"); sys.exit(1)
        all_meta_rows.append(pd.read_csv(p))

    all_meta = pd.concat(all_meta_rows).drop_duplicates("sample_id")
    n_cached = sum(
        1 for sid in all_meta["sample_id"]
        if os.path.exists(os.path.join(KMER_DIR, f"{sid}_k{K}.npy"))
    )
    n_needed = len(all_meta) - n_cached
    print(f"Unique samples across {N_ROUNDS} rounds: {len(all_meta)}")
    print(f"  Cached: {n_cached}  |  Need download: {n_needed}")

    # Build k-mer matrix for all unique samples (streaming download)
    kmer_full = build_kmer_matrix(all_meta)
    if kmer_full.empty:
        print("ERROR: no k-mer data available."); sys.exit(1)
    print(f"\nK-mer matrix: {kmer_full.shape[0]} samples × {kmer_full.shape[1]} features")

    # Run N_ROUNDS rounds
    all_rows = []
    for r in range(1, N_ROUNDS + 1):
        meta = pd.read_csv(os.path.join(SPLITS_DIR, f"obesity_metadata_round{r}.csv"))
        print(f"\n── Round {r} ──────────────")
        rows = run_one_round(meta, kmer_full, r)
        rows.to_csv(os.path.join(RESULTS_DIR, f"kmer_round{r}.csv"), index=False)
        all_rows.append(rows)

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(os.path.join(RESULTS_DIR, "kmer_all_rounds.csv"), index=False)

    summary = (
        all_df.groupby("classifier")[["AUC", "AUPRC", "F1_macro", "MCC"]]
        .agg(["mean", "std"]).round(4)
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(RESULTS_DIR, "results_kmer.csv"), index=False)

    print("\n" + "=" * 60)
    print(f"K-mer pipeline – {N_ROUNDS}-round summary")
    print("=" * 60)
    for _, row in summary.iterrows():
        print(f"  {row['classifier']:15s}  "
              f"AUC={row['AUC_mean']:.3f}±{row['AUC_std']:.3f}  "
              f"AUPRC={row['AUPRC_mean']:.3f}±{row['AUPRC_std']:.3f}  "
              f"F1={row['F1_macro_mean']:.3f}±{row['F1_macro_std']:.3f}  "
              f"MCC={row['MCC_mean']:.3f}±{row['MCC_std']:.3f}")

    plot_final(all_df)
    print(f"\nFigure → {FIGURES_DIR}/kmer_{N_ROUNDS}round_summary.png")


if __name__ == "__main__":
    main()
