#!/usr/bin/env bash
# Extract canonical k-mer vectors at k=7, 9, 10 (full HMP2 read depth, no
# truncation) for all 1,338 IBD samples, extending the existing k in
# {3,6,8} ablation (Table S9) to a wider, multi-split justification for k=8.
set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR=/alina-data0/jingbo-data
OUT_DIR=data/kmer_counts
SAMPLE_LIST=../_common/ibd_sample_ids.txt
SCRIPT=../_common/extract_kmers_generic.py
PARALLEL=16

for k in 7 9 10; do
  echo "=== k=$k ==="
  cat "$SAMPLE_LIST" | xargs -P "$PARALLEL" -I{} \
    nice -n 15 python3 "$SCRIPT" {} "$k" "$DATA_DIR" "$OUT_DIR"
done
echo "All k values done."
