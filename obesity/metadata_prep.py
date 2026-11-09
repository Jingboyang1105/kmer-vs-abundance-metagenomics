"""
Step 1 – Metadata preparation for obesity task: generate 10 random subsample rounds.
Input:  sampleMetadata.csv (cMD3, project root)
Output: splits/obesity/obesity_metadata_round{1..10}.csv  (per round)
        data/obesity_metadata_full.csv                    (all BMI-annotated samples)

Each round: ~1600 samples, natural 1:4 ratio (320 obese / 1280 non-obese),
stratified by study, participant-level 80/20 train/test split.
Matches IBD dataset scale (~1638 HMP2 samples).
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR   = os.path.join(os.path.dirname(__file__), "../data")
SPLITS_DIR = os.path.join(os.path.dirname(__file__), "../splits/obesity")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)

META_CSV   = os.path.join(ROOT, "sampleMetadata.csv")
N_ROUNDS   = 10
TARGET_N   = 1600
OBESE_FRAC = 0.20
BASE_SEED  = 42


def make_labels(smd: pd.DataFrame) -> pd.DataFrame:
    bmi = smd[smd["BMI"].notna()].copy()
    bmi["label"] = (bmi["BMI"] >= 30).astype(int)
    return bmi


def stratified_subsample(df: pd.DataFrame, n_total: int,
                          obese_frac: float, seed: int) -> pd.DataFrame:
    n_obese    = round(n_total * obese_frac)
    n_nonobese = n_total - n_obese
    rng = np.random.default_rng(seed)
    parts = []
    for label, n_target in [(1, n_obese), (0, n_nonobese)]:
        subset = df[df["label"] == label].copy()
        if len(subset) <= n_target:
            parts.append(subset)
            continue
        study_counts = subset["study_name"].value_counts()
        sampled, remaining = [], n_target
        for study, count in study_counts.items():
            n_from = max(1, round(n_target * count / len(subset)))
            n_from = min(n_from, count, remaining)
            sampled.append(
                subset[subset["study_name"] == study].sample(
                    n=n_from, random_state=int(rng.integers(1e6))
                )
            )
            remaining -= n_from
            if remaining <= 0:
                break
        parts.append(pd.concat(sampled))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def add_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    id_col = "subject_id" if "subject_id" in df.columns else "sample_id"
    df = df.copy()
    df[id_col] = df[id_col].fillna(df["sample_id"])
    subj_label = (
        df.groupby(id_col)["label"]
        .agg(lambda x: int(x.mode()[0]))
        .reset_index()
        .rename(columns={"label": "subj_label"})
    )
    train_subj, _ = train_test_split(
        subj_label[id_col].values,
        test_size=0.2, random_state=seed,
        stratify=subj_label["subj_label"].values,
    )
    train_set = set(train_subj)
    df["split"] = df[id_col].map(lambda s: "train" if s in train_set else "test")
    return df


def main():
    smd = pd.read_csv(META_CSV, low_memory=False)
    bmi = make_labels(smd)
    print(f"Full BMI samples: {len(bmi)}  "
          f"(obese={bmi['label'].sum()}, non-obese={(bmi['label']==0).sum()})")

    # Save full dataset (used as abundance upper-bound / supplementary)
    full = add_split(bmi, BASE_SEED)
    keep_cols = [c for c in
                 ["sample_id", "subject_id", "study_name", "BMI",
                  "label", "split", "NCBI_accession"]
                 if c in full.columns]
    full[keep_cols].to_csv(os.path.join(DATA_DIR, "obesity_metadata_full.csv"), index=False)
    print(f"Full metadata saved ({len(full)} samples)")

    # Generate N_ROUNDS subsampled rounds
    all_sub_ids = set()
    for r in range(1, N_ROUNDS + 1):
        seed = BASE_SEED + r * 100
        sub  = stratified_subsample(bmi, TARGET_N, OBESE_FRAC, seed)
        sub  = add_split(sub, seed)
        n_ob = sub["label"].sum()
        n_no = (sub["label"] == 0).sum()
        tr   = (sub["split"] == "train").sum()
        te   = (sub["split"] == "test").sum()
        print(f"Round {r} (seed={seed}): {len(sub)} samples | "
              f"obese={n_ob} non-obese={n_no} | train={tr} test={te} | "
              f"studies={sub['study_name'].nunique()}")
        sub[keep_cols].to_csv(
            os.path.join(SPLITS_DIR, f"obesity_metadata_round{r}.csv"), index=False
        )
        all_sub_ids.update(sub["sample_id"].tolist())

    print(f"\nUnique samples across all {N_ROUNDS} rounds: {len(all_sub_ids)}")
    print("(k-mer FASTQ needed for these samples only)")


if __name__ == "__main__":
    main()
