#!/usr/bin/env python3
"""
Build clean, submission-ready supplementary tables (Table S15-S20) from the
supplementary-experiment results, matching the column/naming
conventions of the existing Table S1-S14 set.

Outputs -> ../new_supplementary_tables/Table_S15_*.csv ... Table_S20_*.csv
"""
import os
import pandas as pd

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "new_supplementary_tables")
os.makedirs(OUT, exist_ok=True)


def save(df, name):
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False)
    print(f"[saved] {name}  ({len(df)} rows)")


# ── Table S15: IBD leave-one-site-out ───────────────────────────────────────
df = pd.read_csv(f"{EXP}/01_ibd_loso/results/ibd_loso_summary.csv")
df = df.rename(columns={"pipeline": "Pipeline", "classifier": "Classifier"})
df = df[["Pipeline", "Classifier", "AUC_mean", "AUC_std", "AUPRC_mean", "AUPRC_std",
         "F1_macro_mean", "F1_macro_std", "MCC_mean", "MCC_std", "AUC_count"]]
df = df.rename(columns={"AUC_count": "n_sites"})
save(df, "Table_S15_IBD_leave_one_site_out.csv")

per_site = pd.read_csv(f"{EXP}/01_ibd_loso/results/ibd_loso_per_site.csv")
save(per_site, "Table_S15b_IBD_leave_one_site_out_per_site.csv")

# ── Table S16: Geography country-study confound + study-held-out ──────────
conf = pd.read_csv(f"{EXP}/02_geography_study_confound/results/country_study_confound.csv")
conf = conf.rename(columns={"country": "Country", "n_studies": "N_contributing_studies"})
save(conf, "Table_S16a_Geography_country_study_confound.csv")

holdout = pd.read_csv(f"{EXP}/02_geography_study_confound/results/geography_study_holdout_summary.csv")
holdout = holdout.rename(columns={"classifier": "Classifier"})
save(holdout, "Table_S16b_Geography_study_held_out_abundance.csv")

# ── Table S17: Obesity within-study vs mixed-study batch-signal test ──────
gap = pd.read_csv(f"{EXP}/03_obesity_batch_signal/results/obesity_batch_signal_gap_comparison.csv")
gap["regime"] = gap["regime"].map({
    "A_mixed_study": "A: mixed-study pool (manuscript design)",
    "B_within_study": "B: within-study only (MetaCardis_2020_a)",
})
gap = gap.rename(columns={
    "regime": "Evaluation regime", "classifier": "Classifier",
    "abundance_AUC_mean": "Abundance_AUC", "kmer_AUC_mean": "Kmer_AUC",
    "AUC_gap_kmer_minus_abundance": "Gap_kmer_minus_abundance",
})
save(gap, "Table_S17_Obesity_batch_signal_regime_comparison.csv")

# ── Table S18: IBD matched-dimension check ─────────────────────────────────
dim = pd.read_csv(f"{EXP}/04_matched_dimension_check/results/ibd_matched_dimension_gap.csv")
dim = dim.rename(columns={
    "matched_dim": "Matched_dimension", "classifier": "Classifier",
    "abundance_AUC": "Abundance_AUC", "kmer_AUC": "Kmer_AUC",
    "gap_kmer_minus_abundance": "Gap_kmer_minus_abundance",
})
save(dim, "Table_S18_IBD_matched_dimension_check.csv")

# ── Table S19: HMP2 read-depth sensitivity ─────────────────────────────────
depth = pd.read_csv(f"{EXP}/05_hmp2_read_depth_sensitivity/results/ibd_depth_sensitivity_summary.csv")
depth = depth.rename(columns={"depth": "Read_depth", "classifier": "Classifier"})
save(depth, "Table_S19_HMP2_read_depth_sensitivity.csv")

# ── Table S20: Extended k-mer length ablation ──────────────────────────────
kabl = pd.read_csv(f"{EXP}/06_kmer_k_ablation_extended/results/ibd_k_ablation_extended_summary.csv")
kabl = kabl.rename(columns={"k": "k", "classifier": "Classifier"})
save(kabl, "Table_S20_IBD_extended_k_ablation.csv")

print("\nAll new supplementary tables written to", OUT)
