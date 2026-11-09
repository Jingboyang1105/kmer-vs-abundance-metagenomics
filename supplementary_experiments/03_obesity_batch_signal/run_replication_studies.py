#!/usr/bin/env python3
"""
Replicate the within-study batch-signal test (run_obesity_batch_signal.py,
Regime B) on two additional large cMD3 obesity studies, to check that the
AUC-gap reversal is not an idiosyncratic property of MetaCardis_2020_a.

Second-look review question: "you compared mixed-study vs. within-study
using only one study (MetaCardis_2020_a) -- does the reversal still hold if
you repeat the within-study regime on a different large cohort?"

Target studies (chosen for size and class-count adequacy for repeated 80/20
splits; MetaCardis remains the primary, best class-balanced example):
  - ZeeviD_2015      (n=870,  192 obese / 678 non-obese)
  - NielsenHB_2014   (n=396,  120 obese / 276 non-obese)

Reuses the exact same feature construction, feature selection, classifiers
and hyperparameters as run_obesity_batch_signal.py (Regime B only -- Regime A
is study-independent and does not need to be rerun).

Outputs:
  results/obesity_batch_signal_replication_all_rounds.csv
  results/obesity_batch_signal_replication_gap.csv
"""

import os

import pandas as pd

import run_obesity_batch_signal as base

RESULTS_DIR = base.RESULTS_DIR
REPLICATION_STUDIES = ["ZeeviD_2015", "NielsenHB_2014"]


def main():
    print("Loading metadata and abundance matrix...")
    meta = pd.read_csv(base.META_CSV)
    meta = meta.dropna(subset=["sample_id", "label"]).copy()
    meta["label"] = meta["label"].astype(int)
    abund_full = pd.read_csv(base.ABUNDANCE_CSV, index_col="sample_id")
    abund_full = abund_full[~abund_full.index.duplicated(keep="first")]

    all_rows = []
    for study in REPLICATION_STUDIES:
        meta_study = meta[meta["study_name"] == study].copy()
        print(f"\n{'='*70}\nStudy: {study}  n={len(meta_study)}  "
              f"label counts={meta_study['label'].value_counts().to_dict()}\n{'='*70}")
        for i, seed in enumerate(base.SEEDS, 1):
            print(f"-- Round {i}/{base.N_ROUNDS} (seed={seed}) --")
            rows = base.run_regime_b(meta_study, abund_full, seed, i)
            for r in rows:
                r["study"] = study
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out = os.path.join(RESULTS_DIR, "obesity_batch_signal_replication_all_rounds.csv")
    df.to_csv(out, index=False)

    summary = (
        df.groupby(["study", "pipeline", "classifier"])[["AUC", "AUPRC", "F1_macro", "MCC"]]
        .agg(["mean", "std"]).round(4)
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()

    gaps = []
    for study in REPLICATION_STUDIES:
        for clf in df["classifier"].unique():
            ab = summary[(summary.study == study) & (summary.classifier == clf) & (summary.pipeline == "Abundance")]
            km = summary[(summary.study == study) & (summary.classifier == clf) & (summary.pipeline == "K-mer")]
            if len(ab) and len(km):
                gaps.append({
                    "study": study, "classifier": clf,
                    "abundance_AUC_mean": float(ab["AUC_mean"].iloc[0]),
                    "kmer_AUC_mean": float(km["AUC_mean"].iloc[0]),
                    "AUC_gap_kmer_minus_abundance": round(float(km["AUC_mean"].iloc[0] - ab["AUC_mean"].iloc[0]), 4),
                })
    gap_df = pd.DataFrame(gaps)
    gap_out = os.path.join(RESULTS_DIR, "obesity_batch_signal_replication_gap.csv")
    gap_df.to_csv(gap_out, index=False)

    print(f"\n{'='*70}\nReplication: k-mer minus abundance AUC gap, within-study-only\n{'='*70}")
    print(gap_df.to_string(index=False))
    print(f"\nSaved -> {out}\nSaved -> {gap_out}")


if __name__ == "__main__":
    main()
