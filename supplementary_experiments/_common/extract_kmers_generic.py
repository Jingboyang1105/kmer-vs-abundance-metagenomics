#!/usr/bin/env python3
"""
Generic canonical k-mer extractor: supports arbitrary k AND an optional
read-count truncation, so it covers both supplementary tasks:

  - 06_kmer_k_ablation_extended: new k values (7, 9, 10) at full HMP2 depth
  - 05_hmp2_read_depth_sensitivity: k=8 at a truncated read depth, to match
    the cMD3 200,000-read protocol described in the manuscript's Methods
    (Section 2.3) and test whether the HMP2/cMD3 extraction-depth difference
    explains part of the
    HMP2-vs-cMD3 k-mer performance difference.

Adapted from MBEC/journal/ibd/extract_kmers.py (same canonicalisation, same
numba rolling-hash counter). The only functional addition is MAX_READS: when
set, `head` truncates the decompressed sequence-line stream before counting,
which (unlike truncating post-hoc) also makes extraction faster for small
depths since pigz can stop decompressing once `head` closes the pipe.

Usage:
    python3 extract_kmers_generic.py <sample_id> <k> <data_dir> <out_dir> [max_reads]

Output: <out_dir>/<sample_id>_k<k>[_r<max_reads>].npy
"""

import sys
import os
import subprocess
import numpy as np
from numba import njit
from pathlib import Path


def build_canonical_lookup(k: int):
    n = 4 ** k
    forward = np.arange(n, dtype=np.int64)
    rc = np.zeros(n, dtype=np.int64)
    temp = forward.copy()
    for _ in range(k):
        rc = (rc << 2) | (3 - (temp & 3))
        temp >>= 2
    canonical_raw = np.minimum(forward, rc)
    unique, lookup = np.unique(canonical_raw, return_inverse=True)
    return lookup.astype(np.int32), unique.astype(np.int32)


@njit(cache=True)
def _rolling_kmer_count(ints: np.ndarray, k: int, lookup: np.ndarray, counts: np.ndarray) -> None:
    mask = np.int64((1 << (2 * k)) - 1)
    kmer = np.int64(0)
    skip = k - 1
    for i in range(len(ints)):
        b = np.int32(ints[i])
        if b < 0:
            kmer = np.int64(0)
            skip = k - 1
        else:
            kmer = (kmer << np.int64(2) | np.int64(b)) & mask
            if skip == 0:
                counts[lookup[kmer]] += np.int64(1)
            else:
                skip -= 1


def _make_enc_table() -> np.ndarray:
    t = np.full(256, -1, dtype=np.int8)
    for b, v in [(65, 0), (67, 1), (71, 2), (84, 3), (97, 0), (99, 1), (103, 2), (116, 3)]:
        t[b] = v
    return t


_ENC = _make_enc_table()


def _accumulate_file(fpath: str, k: int, lookup: np.ndarray, counts: np.ndarray, max_reads: int | None) -> None:
    if max_reads:
        # `head` closes the pipe early, so pigz stops decompressing once
        # enough sequence lines have been emitted -- truncated extraction is
        # therefore *faster* than full extraction, not just a post-hoc slice.
        cmd = f"pigz -dc '{fpath}' | sed -n '2~4p' | head -n {max_reads}"
    else:
        cmd = f"pigz -dc '{fpath}' | sed -n '2~4p'"
    proc = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw = proc.stdout
    arr = _ENC[np.frombuffer(raw, dtype=np.uint8)]
    _rolling_kmer_count(arr, k, lookup, counts)


def main() -> None:
    if len(sys.argv) not in (5, 6):
        print(f"Usage: {sys.argv[0]} <sample_id> <k> <data_dir> <out_dir> [max_reads_per_mate]")
        sys.exit(1)

    sample_id, k_str, data_dir, out_dir = sys.argv[1:5]
    max_reads = int(sys.argv[5]) if len(sys.argv) == 6 else None
    k = int(k_str)

    suffix = f"_k{k}" + (f"_r{max_reads}" if max_reads else "")
    out_path = Path(out_dir) / f"{sample_id}{suffix}.npy"
    if out_path.exists():
        print(f"[skip] {sample_id}{suffix} already exists")
        return

    r1 = os.path.join(data_dir, f"{sample_id}_R1.fastq.gz")
    r2 = os.path.join(data_dir, f"{sample_id}_R2.fastq.gz")
    for fp in (r1, r2):
        if not os.path.exists(fp):
            print(f"[warn] missing: {fp}", file=sys.stderr)
            sys.exit(1)

    lookup, unique = build_canonical_lookup(k)
    n_canonical = len(unique)
    counts = np.zeros(n_canonical, dtype=np.int64)

    _warm = np.array([0, 1, 2, 3, -1, 0], dtype=np.int8)
    _c = np.zeros(n_canonical, dtype=np.int64)
    _rolling_kmer_count(_warm, k, lookup, _c)

    _accumulate_file(r1, k, lookup, counts, max_reads)
    _accumulate_file(r2, k, lookup, counts, max_reads)

    total = counts.sum()
    freq = (counts / float(max(total, 1))).astype(np.float32)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.save(out_path, freq)
    print(f"[done] {sample_id} k={k} max_reads={max_reads}: {n_canonical} canonical k-mers, {total:,} counts")


if __name__ == "__main__":
    main()
