#!/usr/bin/env Rscript
# Download MetaPhlAn3 relative abundance for geography-eligible samples.
# Reuses obesity abundance matrix (already downloaded) for overlapping samples.
# Only fetches the additional samples not present in the obesity matrix.
# Output: <script_dir>/../data/cmd3_geography_abundance.csv

suppressMessages({
  library(curatedMetagenomicData)
  library(SummarizedExperiment)
  library(dplyr)
})

# Resolve paths relative to this script's location
args       <- commandArgs(trailingOnly = FALSE)
file_arg   <- args[grep("^--file=", args)]
SCRIPT_DIR <- if (length(file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1])))
} else {
  getwd()
}

META_CSV    <- file.path(SCRIPT_DIR, "..", "..", "sampleMetadata.csv")
OBESITY_CSV <- file.path(SCRIPT_DIR, "..", "data", "cmd3_relative_abundance.csv")
OUT_DIR     <- file.path(SCRIPT_DIR, "..", "data")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
OUT_FILE    <- file.path(OUT_DIR, "cmd3_geography_abundance.csv")

cat("Reading metadata...\n")
smd <- read.csv(META_CSV, stringsAsFactors = FALSE)

country_counts <- table(smd$country)
eligible_countries <- names(country_counts[country_counts >= 50])
cat(sprintf("Countries with n>=50: %d\n", length(eligible_countries)))

geo_meta <- smd[smd$country %in% eligible_countries & !is.na(smd$country), ]
cat(sprintf("Geography samples: %d\n", nrow(geo_meta)))

# ── Step 1: Load obesity abundance (already exists) ──────────────────
cat("\nLoading obesity abundance matrix (cached)...\n")
obesity_df <- read.csv(OBESITY_CSV, row.names = 1, check.names = FALSE)
cat(sprintf("  Obesity matrix: %d samples x %d species\n",
            nrow(obesity_df), ncol(obesity_df)))

geo_ids     <- geo_meta$sample_id
overlap_ids <- intersect(geo_ids, rownames(obesity_df))
cat(sprintf("  Overlap with geography: %d / %d samples (%.1f%%)\n",
            length(overlap_ids), length(geo_ids),
            100 * length(overlap_ids) / length(geo_ids)))

abund_from_obesity <- obesity_df[overlap_ids, , drop = FALSE]

# ── Step 2: Download remaining samples ───────────────────────────────
new_ids <- setdiff(geo_ids, overlap_ids)
cat(sprintf("\nFetching %d additional samples via returnSamples...\n", length(new_ids)))

keep_cols <- intersect(
  c("study_name", "sample_id", "country", "subject_id", "NCBI_accession"),
  colnames(geo_meta)
)
new_meta <- geo_meta[geo_meta$sample_id %in% new_ids, keep_cols]

if (nrow(new_meta) > 0) {
  tse_new <- returnSamples(new_meta, dataType = "relative_abundance", rownames = "short")
  cat(sprintf("  New matrix: %d species x %d samples\n", nrow(tse_new), ncol(tse_new)))

  mat_new <- assay(tse_new, "relative_abundance")
  df_new  <- as.data.frame(t(mat_new))

  all_species <- union(colnames(abund_from_obesity), colnames(df_new))
  cat(sprintf("  Merging: %d shared + union = %d total species\n",
              length(intersect(colnames(abund_from_obesity), colnames(df_new))),
              length(all_species)))

  for (sp in setdiff(all_species, colnames(abund_from_obesity))) {
    abund_from_obesity[[sp]] <- 0
  }
  for (sp in setdiff(all_species, colnames(df_new))) {
    df_new[[sp]] <- 0
  }

  abund_from_obesity <- abund_from_obesity[, all_species, drop = FALSE]
  df_new             <- df_new[, all_species, drop = FALSE]
  combined           <- rbind(abund_from_obesity, df_new)
} else {
  combined <- abund_from_obesity
}

cat(sprintf("\nFinal geography abundance: %d samples x %d species\n",
            nrow(combined), ncol(combined)))

df_out <- cbind(sample_id = rownames(combined), combined)
cat(sprintf("Writing %s ...\n", OUT_FILE))
write.csv(df_out, OUT_FILE, row.names = FALSE, quote = FALSE)
cat("Done.\n")
