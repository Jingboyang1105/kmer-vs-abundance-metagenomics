#!/usr/bin/env Rscript
# Download cMD3 relative abundance for BMI-annotated samples.
# Output: <script_dir>/../data/cmd3_relative_abundance.csv
# Columns: sample_id, then one column per species (short name from MetaPhlAn3)

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

META_CSV <- file.path(SCRIPT_DIR, "..", "..", "sampleMetadata.csv")
OUT_DIR  <- file.path(SCRIPT_DIR, "..", "data")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
OUT_FILE <- file.path(OUT_DIR, "cmd3_relative_abundance.csv")

cat("Reading metadata:", META_CSV, "\n")
smd <- read.csv(META_CSV, stringsAsFactors = FALSE)
bmi_meta <- smd[!is.na(smd$BMI),
                c("study_name", "sample_id", "BMI", "subject_id", "NCBI_accession")]
cat("BMI-annotated samples:", nrow(bmi_meta), "\n")
cat("Studies:", length(unique(bmi_meta$study_name)), "\n\n")

cat("Downloading relative abundance from ExperimentHub...\n")
cat("(50 studies will be downloaded; expect 30-90 minutes)\n\n")
flush.console()

tse <- returnSamples(bmi_meta, dataType = "relative_abundance", rownames = "short")

cat("\nDownload complete.\n")
cat("Dimensions (species x samples):", nrow(tse), "x", ncol(tse), "\n")
flush.console()

mat <- assay(tse, "relative_abundance")   # species x samples matrix
df_out <- as.data.frame(t(mat))           # samples x species
df_out <- cbind(sample_id = rownames(df_out), df_out)

cat("Writing CSV to:", OUT_FILE, "\n")
write.csv(df_out, OUT_FILE, row.names = FALSE, quote = FALSE)
cat("Done.\n")
