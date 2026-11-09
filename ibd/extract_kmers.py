#!/usr/bin/env python3
"""
Step 1: Extract canonical k-mer frequency vector from a single sample (R1 + R2).

Fast implementation:
  - zcat + awk extracts only sequence lines (skips FASTQ headers/quality)
  - All reads concatenated into one int8 array (newlines act as invalid separators)
  - Numba JIT rolling hash loop counts k-mers at near-C speed
  - Canonical form: min(kmer_int, reverse_complement_int)

Output: <out_dir>/<sample_id>_k<k>.npy  — float32 relative-frequency vector
                                           length = n_canonical_kmers

Usage:
    python3 01_extract_kmers.py <sample_id> <k> <data_dir> <out_dir>
"""

import sys
import os
import subprocess
import numpy as np
from numba import njit
from pathlib import Path


# ─── Canonical lookup table ──────────────────────���───────────────────────────

def build_canonical_lookup(k: int):
    """
    Returns:
        lookup : int32 (4**k,)   lookup[kmer_int] → canonical_index
        unique : int32 (n_canon,) unique[ci] → representative kmer_int
    """
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


# ─── Numba JIT rolling-hash k-mer counter ────────────────────────────────────

@njit(cache=True)
def _rolling_kmer_count(ints: np.ndarray, k: int,
                         lookup: np.ndarray, counts: np.ndarray) -> None:
    """
    Rolling hash over a concatenated sequence array.
    Values < 0 (including newline=10 encoded as -1) act as separators that
    reset the window, preventing k-mers from spanning read boundaries.
    """
    mask = np.int64((1 << (2 * k)) - 1)
    kmer = np.int64(0)
    skip = k - 1          # need this many more valid bases before first k-mer

    for i in range(len(ints)):
        b = np.int32(ints[i])
        if b < 0:           # N or newline: reset window
            kmer = np.int64(0)
            skip = k - 1
        else:
            kmer = (kmer << np.int64(2) | np.int64(b)) & mask
            if skip == 0:
                counts[lookup[kmer]] += np.int64(1)
            else:
                skip -= 1


# ─── Encoding table ──────────────────────────────────────────────────────────

def _make_enc_table() -> np.ndarray:
    t = np.full(256, -1, dtype=np.int8)
    for b, v in [(65, 0), (67, 1), (71, 2), (84, 3),
                 (97, 0), (99, 1), (103, 2), (116, 3)]:
        t[b] = v
    return t


_ENC = _make_enc_table()


# ─── Per-file counting ────────────────────────────────────────────────────────

def _accumulate_file(fpath: str, k: int,
                     lookup: np.ndarray, counts: np.ndarray) -> None:
    """
    Extract sequence lines with awk (skips header/quality/+ lines),
    encode to int8, then count k-mers via JIT rolling hash.
    Newlines between reads are encoded as -1, naturally resetting windows.
    """
    proc = subprocess.run(
        ["bash", "-c", f"pigz -dc '{fpath}' | sed -n '2~4p'"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    raw = proc.stdout          # bytes: all sequences separated by newlines
    arr = _ENC[np.frombuffer(raw, dtype=np.uint8)]   # int8 array
    _rolling_kmer_count(arr, k, lookup, counts)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <sample_id> <k> <data_dir> <out_dir>")
        sys.exit(1)

    sample_id, k_str, data_dir, out_dir = sys.argv[1:]
    k = int(k_str)

    out_path = Path(out_dir) / f"{sample_id}_k{k}.npy"
    if out_path.exists():
        print(f"[skip] {sample_id}_k{k} already exists")
        sys.exit(0)

    r1 = os.path.join(data_dir, f"{sample_id}_R1.fastq.gz")
    r2 = os.path.join(data_dir, f"{sample_id}_R2.fastq.gz")
    for fp in (r1, r2):
        if not os.path.exists(fp):
            print(f"[warn] missing: {fp}", file=sys.stderr)
            sys.exit(1)

    lookup, unique = build_canonical_lookup(k)
    n_canonical = len(unique)
    counts = np.zeros(n_canonical, dtype=np.int64)

    # Warm-up: compile JIT function before timing (first sample only matters)
    _warm = np.array([0, 1, 2, 3, -1, 0], dtype=np.int8)
    _c    = np.zeros(n_canonical, dtype=np.int64)
    _rolling_kmer_count(_warm, k, lookup, _c)   # triggers compilation

    _accumulate_file(r1, k, lookup, counts)
    _accumulate_file(r2, k, lookup, counts)

    total = counts.sum()
    freq = (counts / float(max(total, 1))).astype(np.float32)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.save(out_path, freq)
    print(f"[done] {sample_id} k={k}: {n_canonical} canonical k-mers, "
          f"{total:,} counts")


if __name__ == "__main__":
    main()
