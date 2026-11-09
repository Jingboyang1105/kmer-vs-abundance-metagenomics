#!/usr/bin/env python3
"""
Reduced-scale k-mer extraction pilot for the geography study-held-out check.

Repeats the geographic-origin study-aware evaluation for k-mer as well as
abundance. The full cMD3 geography cohort
(n=22,397) has no cached k-mer vectors and was judged out of scope for a full
re-extraction. This script instead extracts k=8 k-mer vectors (same protocol
as the published cMD3 pipeline: 200,000 reads from the first available FASTQ
mate, archive/line3/geography/scripts/02b_kmer_pipeline.py) for exactly the
same 1,587-sample round-1 pool already used for the abundance study-held-out
result (geo_metadata_round1.csv), via SRA streaming (fastq-dump).

This is a bounded, clearly-scoped pilot (n=1,587, one round), not a
replacement for the full-cohort analysis -- reported as such.

Output: data/kmer_counts_pilot/<sample_id>_k8.npy
"""

import os
import sys
import subprocess
import shutil
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KMER_DIR = os.path.join(SCRIPT_DIR, "data", "kmer_counts_pilot")
FASTQ_TMP = "/tmp/geo_kmer_pilot_fastq"
os.makedirs(KMER_DIR, exist_ok=True)
os.makedirs(FASTQ_TMP, exist_ok=True)

# REPO_ROOT: supplementary_experiments/02_geography_study_confound -> up two levels -> repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_ROOT = os.environ.get("MBEC_DATA_ROOT", REPO_ROOT)

FASTQ_DUMP = "fastq-dump"   # update to full path if not on PATH
K = 8
MAX_READS = 200_000
WORKERS = 24

# Produced by geography/metadata_prep.py; see README "Execution order".
META_CSV = os.path.join(DATA_ROOT, "geography", "data", "geo_metadata_round1.csv")

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

_ENCODE = np.full(256, 255, dtype=np.uint8)
for _c, _i in zip(b"ACGTacgt", [0, 1, 2, 3, 0, 1, 2, 3]):
    _ENCODE[_c] = _i

if HAS_NUMBA:
    @njit(cache=True)
    def _count_kmers_numba(seq_array, k, encode, counts):
        L = len(seq_array)
        for pos in range(L - k + 1):
            fwd = np.int64(0)
            rev = np.int64(0)
            ok = True
            for j in range(k):
                b = encode[seq_array[pos + j]]
                if b == 255:
                    ok = False
                    break
                fwd = fwd * 4 + np.int64(b)
                b_rev = encode[seq_array[pos + k - 1 - j]]
                if b_rev == 255:
                    ok = False
                    break
                rev = rev * 4 + np.int64(3 - b_rev)
            if ok:
                counts[fwd if fwd < rev else rev] += 1.0


def count_kmers(fastq_path: str, k: int = 8, max_reads: int = 200_000) -> np.ndarray:
    n = 4 ** k
    counts = np.zeros(n, dtype=np.float64)
    opener = gzip.open if fastq_path.endswith(".gz") else open
    n_reads = 0
    with opener(fastq_path, "rt") as fh:
        for i, line in enumerate(fh):
            if i % 4 != 1:
                continue
            seq = np.frombuffer(line.strip().encode(), dtype=np.uint8)
            if HAS_NUMBA:
                _count_kmers_numba(seq, k, _ENCODE, counts)
            n_reads += 1
            if n_reads >= max_reads:
                break
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts.astype(np.float32)


def download_and_count(accession, sample_id: str) -> bool:
    cache = os.path.join(KMER_DIR, f"{sample_id}_k{K}.npy")
    if os.path.exists(cache):
        return True

    if not isinstance(accession, str) or not accession.strip():
        print(f"[fail] {sample_id}: missing NCBI_accession", flush=True)
        return False

    # Some samples list multiple semicolon-separated SRA runs (re-sequenced /
    # multi-lane samples); use the first run only, consistent with this
    # pilot's "first available mate/run" scoping.
    accession = accession.split(";")[0].strip()

    out_dir = os.path.join(FASTQ_TMP, accession)
    os.makedirs(out_dir, exist_ok=True)
    try:
        result = subprocess.run(
            [FASTQ_DUMP, accession, "--outdir", out_dir, "--split-files",
             "--maxSpotId", str(MAX_READS), "--skip-technical", "-Q", "0"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"[fail] {sample_id} ({accession}): {result.stderr[:150]}", flush=True)
            return False
        fastqs = sorted(f for f in os.listdir(out_dir) if f.endswith(".fastq"))
        if not fastqs:
            print(f"[fail] {sample_id} ({accession}): no fastq produced", flush=True)
            return False
        vec = count_kmers(os.path.join(out_dir, fastqs[0]), k=K, max_reads=MAX_READS)
        np.save(cache, vec)
        print(f"[done] {sample_id} ({accession})", flush=True)
        return True
    except Exception as e:
        print(f"[error] {sample_id} ({accession}): {e}", flush=True)
        return False
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def main():
    if not os.path.exists(META_CSV):
        print(f"Missing {META_CSV}")
        print(
            "Run geography/metadata_prep.py first to generate geo_metadata_round1.csv "
            "(see README 'Execution order'). Set MBEC_DATA_ROOT if it lives outside the repo."
        )
        raise SystemExit(1)
    meta = pd.read_csv(META_CSV)
    print(f"Round-1 pool: {len(meta)} samples")
    todo = [(r.NCBI_accession, r.sample_id) for r in meta.itertuples()
            if not os.path.exists(os.path.join(KMER_DIR, f"{r.sample_id}_k{K}.npy"))]
    print(f"To extract: {len(todo)} (already cached: {len(meta) - len(todo)})")

    n_ok = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(download_and_count, acc, sid): sid for acc, sid in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            if fut.result():
                n_ok += 1
            if i % 50 == 0:
                print(f"  progress: {i}/{len(todo)} ({n_ok} ok)", flush=True)

    print(f"\nDone. {n_ok}/{len(todo)} newly extracted.")
    have = sum(1 for r in meta.itertuples() if os.path.exists(os.path.join(KMER_DIR, f"{r.sample_id}_k{K}.npy")))
    print(f"Total cached for round-1 pool: {have}/{len(meta)}")


if __name__ == "__main__":
    main()
