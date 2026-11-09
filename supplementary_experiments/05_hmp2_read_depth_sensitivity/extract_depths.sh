#!/usr/bin/env bash
# Extract k=8 canonical k-mer vectors for all 1,338 IBD samples at three
# truncated read depths (50k / 100k / 200k reads per mate), matching the
# manuscript's cMD3 200,000-read protocol (Methods, Section 2.3), to test
# whether HMP2's untruncated extraction vs. cMD3's truncated extraction
# explains part of the HMP2-vs-cMD3 k-mer performance difference.
set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR=/alina-data0/jingbo-data
OUT_DIR=data/kmer_counts_depth
SAMPLE_LIST=../_common/ibd_sample_ids.txt
SCRIPT=../_common/extract_kmers_generic.py
PARALLEL=16

for depth in 50000 100000 200000; do
  echo "=== depth=$depth ==="
  cat "$SAMPLE_LIST" | xargs -P "$PARALLEL" -I{} \
    nice -n 15 python3 "$SCRIPT" {} 8 "$DATA_DIR" "$OUT_DIR" "$depth"
done
echo "All depths done."
