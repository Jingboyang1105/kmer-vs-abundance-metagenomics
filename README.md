# Alignment-Free k-mer versus Species-Abundance Profiling for Metagenomic Classification

Code repository for the manuscript "Alignment-Free k-mer versus
Species-Abundance Profiling for Metagenomic Classification: A Comparative
Cross-Cohort Evaluation" (J. Yang and M. Patterson), *Medical & Biological
Engineering & Computing* (MBEC).

---

## Overview

Three classification tasks are evaluated:

| Task | Cohort | Classes |
|------|--------|---------|
| IBD | iHMP/IBDMDB (HMP2) | IBD vs. non-IBD |
| Obesity | curatedMetagenomicData v3 (cMD3) | obese vs. non-obese (BMI ≥ 30) |
| Geographic origin | cMD3 | 30 countries |

Three feature representations are compared: species-abundance profiles
(CLR-transformed + alpha diversity), canonical 8-mer relative-frequency
vectors, and their concatenation (fusion). Four classifiers (LASSO-LR,
Random Forest, SVM-RBF, XGBoost) are evaluated over ten repeated
participant-level splits/rounds. The primary statistical comparison for all
three tasks restricts each representation to the sample intersection available
to both (matched-sample design), with each task's full-availability
(unmatched) comparison reported as a sensitivity analysis. Statistical
comparisons use
two-sided paired Wilcoxon signed-rank tests with Benjamini–Hochberg
correction across all 48 task–classifier–metric combinations.

This repository is **code plus small derived results** — no raw sequencing
data, no downloaded abundance tables, and no per-sample k-mer count matrices
are included. A `data/` directory does ship the paper's own small
derived/summary CSVs (see "Shipped derived-result data" below); everything
larger or third-party remains external. All underlying raw data are public
(see "Data" below); this keeps the repository small and avoids
redistributing third-party data under its own terms.

---

## Repository layout

```
ibd/                             IBD task (iHMP/IBDMDB)
├── extract_kmers.py             Extract canonical k-mer frequency vectors from FASTQ
├── build_kmer_matrix.py         Aggregate per-sample .npy files into sample×k-mer matrix
├── abundance_pipeline.py        Species CLR + alpha diversity, 10-round repeated split
├── kmer_pipeline.py             Canonical k=8, Welch t-test + BH selection, 10-round
├── fusion_pipeline.py           Abundance + k-mer concatenation, 10-round
├── k_ablation.py                Original k ∈ {3,6,8} ablation (Table S9)
├── feature_selection_sensitivity.py  Matched univariate feature selection (Table S10)
└── export_kmer_differential.py  Full IBD differential k-mer screen (Tables S11–S13)

obesity/                         Obesity task (cMD3)
├── metadata_prep.py             Generate 10 subsampled rounds (1,600 samples/round)
├── abundance_pipeline.py        10-round abundance pipeline
├── kmer_pipeline.py             10-round k-mer pipeline (streams FASTQ from SRA)
├── fusion_pipeline.py           10-round fusion pipeline
├── comparison.py                Cross-pipeline summary table and plots
├── loso_validation.py           Leave-one-study-out robustness validation
├── study_prediction.py          Study-identity sanity check (cohort-effect test)
├── subsampling_sensitivity.py   Sample-size sensitivity analysis
└── download_abundance.R         Download cMD3 abundance profiles via Bioconductor

geography/                       30-country geographic-origin task (cMD3)
├── metadata_prep.py             Generate 10 subsampled rounds (~1,587 samples/round)
├── abundance_pipeline.py        10-round abundance pipeline
├── kmer_pipeline.py             10-round k-mer pipeline
├── fusion_pipeline.py           10-round fusion pipeline
├── comparison.py                Cross-pipeline summary table and plots
└── download_abundance.R         Download cMD3 abundance profiles

analysis/                        Statistical tests and biological attribution
├── matched_subset_benchmark.py  Primary matched-sample comparison (Table 2, Tables S2–S3)
├── statistical_tests.py         Wilcoxon tests + BH correction across 48 comparisons (Fig. 1, Table S4)
├── fusion_pairwise_comparison.py  Within-classifier fusion vs. single-representation tests (Table S14)
├── kmer_attribution.py          k-mer → species Spearman correlation heatmap (Fig. S10)
├── kmer_diversity_analysis.py   k-mer vs. abundance Shannon entropy correlation (Fig. S9)
└── global_summary.py            Unified summary across all tasks and pipelines

figures/                         Manuscript figure generation
├── make_clean_manuscript_figures.py  Single source of truth for the 5 main-text figures
│                                 (Fig1_statistical_tests_heatmap, Fig2_stability_boxplot,
│                                 Fig3_cohort_stress_tests, Fig4_fairness_diagnostics,
│                                 Fig5_fusion_comparison_three_tasks) plus three demoted
│                                 supplementary figures (FigS8_global_summary,
│                                 FigS9_kmer_diversity_correlation,
│                                 FigS10_kmer_attribution_heatmap)
├── ablation_kmer_design.py      Standalone k-ablation figure script
├── fusion_comparison.py         Standalone fusion-comparison figure script
├── figures_extra.py             Additional exploratory figures
└── make_graphical_abstract.py   Graphical abstract generation

supplementary_experiments/       Robustness and sensitivity analyses
├── 01_ibd_loso/                 IBD leave-one-site-out validation (Tables S15–S16, Fig. S1)
├── 02_geography_study_confound/ Country/study confound quantification + k-mer study-held-out pilot (Tables S17–S18, Fig. S2)
├── 03_obesity_batch_signal/     Within-study vs. mixed-study batch-signal test, 6 cMD3 studies (Table S19, Fig. S3)
├── 04_matched_dimension_check/  Matched feature-selection-dimension sensitivity, all 3 tasks (Tables S20/S23/S24, Fig. S4)
├── 05_hmp2_read_depth_sensitivity/  HMP2 read-depth re-extraction sensitivity (Table S21, Fig. S5)
├── 06_kmer_k_ablation_extended/ Extended k ∈ {3,...,10} ablation, 10-split, 4-classifier (Table S22; now also Table 1 in the main text)
├── 07_complementarity_analysis/ Unsupervised PCA complementarity between feature spaces (Tables S26–S27, Fig. S7)
├── 08_corrected_resampled_ttest/ Nadeau–Bengio corrected resampled t-test for the 48 primary comparisons (Table S32)
└── _common/                     Shared helpers (generic k-mer extraction, supplementary figure/table generation)

data/                             Shipped derived-result CSVs (see "Shipped derived-result data" below)
├── supplementary_tables/        Published supplementary tables S1-S32 + Online Resource 3
├── matched_data/                Matched-sample per-round AUC values (IBD, obesity, geography)
└── fusion_results/               Per-round fusion-pipeline metrics (IBD, obesity, geography)

data_manifest.csv                 Map of shipped vs. externally-obtained data paths
requirements.txt
```

---

## Data

All datasets are public; no private or restricted-access data are used.

### IBD (iHMP/IBDMDB)

Download from https://www.ibdmdb.org:

- `hmp2_metadata_2018-08-20.csv` — sample metadata
- `taxonomic_profiles_3.tsv.gz` — MetaPhlAn species-level relative abundances
- Paired-end FASTQ files for all metagenomics samples (HMP2 portal or SRA)

Place the metadata CSV and profile TSV under a directory referenced as
`ibd/data/raw/` by `ibd/abundance_pipeline.py`. Place FASTQ files where
`ibd/extract_kmers.py` can reach them.

### Obesity and geography (cMD3)

Run the R download scripts to retrieve species-level relative abundances via
the `curatedMetagenomicData` Bioconductor package:

```bash
Rscript obesity/download_abundance.R      # -> obesity/data/cmd3_relative_abundance.csv
Rscript geography/download_abundance.R    # -> geography/data/cmd3_geography_abundance.csv
```

`sampleMetadata.csv` from cMD3 is also required by the metadata preparation
scripts. K-mer vectors for cMD3 samples are obtained by streaming FASTQ
files from SRA via `fastq-dump`; see `supplementary_experiments/_common/
extract_kmers_generic.py` for the shared extraction logic used across tasks.

The complete sample-to-accession mapping for all 36,860 samples used across
the three tasks is provided as Online Resource 3 with the manuscript
submission, and is also shipped in this repository at
`data/supplementary_tables/Online_Resource_3_Sample_Accession_Mapping.csv`
(see "Shipped derived-result data" below).

---

## Shipped derived-result data

Unlike the rest of this "code-only" repository, `data/` **is** committed. It
contains the paper's own small derived/summary results — not raw sequencing
data — so a reader can regenerate every manuscript table and figure without
re-downloading and re-processing all 36,860 raw samples:

- `data/supplementary_tables/` — the published supplementary tables (Tables
  S1–S32, plus Online Resource 3), the same files consumed by
  `figures/make_clean_manuscript_figures.py` and referenced throughout
  `supplementary_experiments/`.
- `data/matched_data/` — per-round, per-classifier AUC values for the
  matched-sample abundance/k-mer comparison (IBD, obesity, geography), used
  by Fig. 1 (statistical tests heatmap) and the fusion comparison (Fig. 5).
- `data/fusion_results/` — per-round fusion-pipeline metrics for all three
  tasks, used by Fig. 5.

Total size is under 10 MB. See `data_manifest.csv` at the repo root for a
full path-by-path map of what is shipped here versus what remains external.

**Not shipped** (unchanged from the policy above): raw FASTQ files,
downloaded cMD3 abundance tables, and per-sample k-mer count matrices. Those
remain external and must be obtained or built locally per the "Data"
instructions above; only the two supplementary figures that require the full
raw-FASTQ-derived IBD k-mer matrix (FigS9_kmer_diversity_correlation,
FigS10_kmer_attribution_heatmap) will be skipped by
`figures/make_clean_manuscript_figures.py` until that matrix is built
(`ibd/extract_kmers.py` + `ibd/build_kmer_matrix.py`; see "Execution order").

---

## Environment

Python 3.12 recommended. This repository was tested against the exact
pinned versions in `requirements.txt` — Python 3.12.3, with the four
libraries the manuscript's Methods 2.7 itself reports: scikit-learn 1.8,
XGBoost 3.2, SciPy 1.11, and statsmodels 0.14.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`ibd/extract_kmers.py` requires [pigz](https://zlib.net/pigz/) on PATH.
The cMD3 k-mer pipelines call `fastq-dump` from the
[SRA Toolkit](https://github.com/ncbi/sra-tools); update the `FASTQ_DUMP`
variable near the top of each script to your local `fastq-dump` path.

---

## Execution order

### IBD task

```bash
python ibd/extract_kmers.py <sample_id> 8 <fastq_dir> <kmer_dir>
python ibd/build_kmer_matrix.py 8 <kmer_dir> <metadata_csv> <kmer_dir>
python ibd/abundance_pipeline.py
python ibd/kmer_pipeline.py
python ibd/fusion_pipeline.py
```

### Obesity task

```bash
python obesity/metadata_prep.py
python obesity/abundance_pipeline.py
python obesity/kmer_pipeline.py
python obesity/fusion_pipeline.py
python obesity/comparison.py
python obesity/loso_validation.py
```

### Geography task

```bash
python geography/metadata_prep.py
python geography/abundance_pipeline.py
python geography/kmer_pipeline.py
python geography/fusion_pipeline.py
python geography/comparison.py
```

### Primary analysis, figures, and robustness checks

Run after the three task pipelines complete:

```bash
python analysis/matched_subset_benchmark.py --task all   # Table 2, Tables S2-S3
python analysis/statistical_tests.py                      # Table S4, Fig. 1
python analysis/fusion_pairwise_comparison.py              # Table S14

export MBEC_DATA_ROOT=/path/to/this/repo  # optional; defaults to the repo root, which is where data/ (see "Shipped derived-result data" below) already lives
python figures/make_clean_manuscript_figures.py            # 5 main figures (PDF, TIFF, and editable EPS/SVG) + FigS8-FigS10
python supplementary_experiments/08_corrected_resampled_ttest/run_corrected_resampled_ttest.py   # Table S32 (runs from shipped data/ alone)
```

Each subfolder under `supplementary_experiments/` is self-contained and
documents, in its own script, which manuscript section/table it
reproduces; see `supplementary_experiments/README.md` for
an index.

---

## Reproducibility

Random seeds are fixed and listed explicitly in each pipeline script.
Classifier hyperparameters are identical across representations within a
task. Feature selection is performed inside each training fold. The primary
matched-sample analysis (`analysis/matched_subset_benchmark.py`) restricts
all representations to identical samples and splits for a controlled head-to-head
comparison; each task's full-availability comparison is reported separately
as a sensitivity analysis.

---

## Correspondence with the manuscript

Figure and table numbers in the folder comments above refer to the
published manuscript. The full list of supplementary tables (S1–S32) and
figures (S1–S10), and which script and manuscript section each corresponds
to, is in the manuscript's "Supplementary information" section (provided
separately with the submission, not part of this code-only repository).
