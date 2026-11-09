#!/usr/bin/env python3
"""
Step 2: Aggregate per-sample k-mer vectors into a sample × k-mer matrix.
Merges with HMP2 metadata to attach IBD/nonIBD labels and participant IDs.

Outputs (all in <out_dir>/):
    kmer_matrix_k<k>_X.npy            float32  (n_samples, n_canonical_kmers)
    kmer_matrix_k<k>_labels.npy       str      (n_samples,)
    kmer_matrix_k<k>_participant.npy  str      (n_samples,)
    kmer_matrix_k<k>_sample_ids.npy   str      (n_samples,)
    kmer_matrix_k<k>_kmer_index.npy   int32    (n_canonical,) unique kmer ints

Usage:
    python3 02_build_matrix.py <k> <kmer_dir> <metadata_csv> <out_dir>
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def build_canonical_unique(k: int) -> np.ndarray:
    """Return unique[] array (canonical_index → representative kmer_int)."""
    n = 4 ** k
    forward = np.arange(n, dtype=np.int32)
    rc = np.zeros(n, dtype=np.int32)
    temp = forward.copy()
    for _ in range(k):
        base = temp & 3
        rc = (rc << 2) | (3 - base)
        temp >>= 2
    canonical_raw = np.minimum(forward, rc)
    unique = np.unique(canonical_raw)
    return unique.astype(np.int32)


def load_metadata(path: str) -> pd.DataFrame:
    meta = pd.read_csv(path, low_memory=False)
    meta = meta[meta["data_type"] == "metagenomics"].copy()
    meta = meta[meta["diagnosis"].isin(["CD", "UC", "nonIBD"])].copy()
    meta["diagnosis"] = meta["diagnosis"].replace({"CD": "IBD", "UC": "IBD"})
    meta = meta[["External ID", "Participant ID", "diagnosis"]].drop_duplicates("External ID")
    meta.rename(columns={"External ID": "sample_id"}, inplace=True)
    # Strip _P suffix so IDs match FASTQ filenames (e.g. CSM5FZ3N_P → CSM5FZ3N)
    meta["file_id"] = meta["sample_id"].str.replace(r"_P$", "", regex=True)
    return meta


def main() -> None:
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <k> <kmer_dir> <metadata_csv> <out_dir>")
        sys.exit(1)

    k_str, kmer_dir, metadata_csv, out_dir = sys.argv[1:]
    k = int(k_str)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ── Metadata ─────────────────────────────────────────────────────────────
    meta = load_metadata(metadata_csv)
    print(f"[Meta] {len(meta)} MGX samples: {meta['diagnosis'].value_counts().to_dict()}")

    # ── Available k-mer files ─────────────────────────────────────────────────
    kmer_files = {f.stem.replace(f"_k{k}", ""): f
                  for f in Path(kmer_dir).glob(f"*_k{k}.npy")}
    print(f"[Files] {len(kmer_files)} k-mer files found for k={k}")

    matched = meta[meta["file_id"].isin(kmer_files)].copy().reset_index(drop=True)
    print(f"[Match] {len(matched)} samples paired with k-mer files")
    print(f"        {matched['diagnosis'].value_counts().to_dict()}")

    if len(matched) == 0:
        print("ERROR: no samples matched. Check that 01_extract_kmers.py has been run.")
        sys.exit(1)

    # ── Load first vector to get dimensions ──────────────────────────────────
    first_sid = matched["file_id"].iloc[0]
    n_canonical = len(np.load(kmer_files[first_sid]))
    n_samples = len(matched)
    print(f"[Build] Matrix shape: {n_samples} × {n_canonical}")

    X = np.zeros((n_samples, n_canonical), dtype=np.float32)
    for i, fid in enumerate(matched["file_id"]):
        X[i] = np.load(kmer_files[fid])
        if (i + 1) % 200 == 0:
            print(f"  loaded {i+1}/{n_samples}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    prefix = os.path.join(out_dir, f"kmer_matrix_k{k}")
    np.save(f"{prefix}_X.npy",           X)
    np.save(f"{prefix}_labels.npy",      matched["diagnosis"].values.astype(str))
    np.save(f"{prefix}_participant.npy", matched["Participant ID"].values.astype(str))
    np.save(f"{prefix}_sample_ids.npy",  matched["file_id"].values.astype(str))
    np.save(f"{prefix}_kmer_index.npy",  build_canonical_unique(k))

    print(f"\n[Saved] {prefix}_X.npy  shape={X.shape}")
    print(f"        labels: {pd.Series(matched['diagnosis'].values).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
