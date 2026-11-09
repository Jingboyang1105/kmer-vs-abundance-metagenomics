#!/usr/bin/env python3
"""
Second extension of the within-study batch-signal replication (see
run_replication_studies.py for the first two). Adds three more cMD3 obesity
studies that satisfy the same feasibility bar (k-mer features already
extracted, and a usable count of both classes for repeated 80/20 splits):
  - AsnicarF_2021     (n=136, 20 obese / 116 non-obese)
  - HMP_2012          (n=89,  11 obese / 78 non-obese)
  - LeChatelierE_2013 (n=37,  22 obese / 15 non-obese -- best class balance)

Reuses run_obesity_batch_signal.py's Regime B exactly (same feature
construction, feature selection, classifiers, hyperparameters).
"""
import os
import pandas as pd
import run_obesity_batch_signal as base

RESULTS_DIR = base.RESULTS_DIR
REPLICATION_STUDIES = ["AsnicarF_2021", "HMP_2012", "LeChatelierE_2013"]


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
    out = os.path.join(RESULTS_DIR, "obesity_batch_signal_replication_extra_all_rounds.csv")
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
    gap_out = os.path.join(RESULTS_DIR, "obesity_batch_signal_replication_extra_gap.csv")
    gap_df.to_csv(gap_out, index=False)

    print(f"\n{'='*70}\nReplication (extra 3 studies): k-mer minus abundance AUC gap, within-study-only\n{'='*70}")
    print(gap_df.to_string(index=False))
    print(f"\nSaved -> {out}\nSaved -> {gap_out}")


if __name__ == "__main__":
    main()
