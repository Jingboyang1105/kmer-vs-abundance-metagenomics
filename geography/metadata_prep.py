"""
Step 1 – Metadata preparation for geography task: generate 10 random subsample rounds.
Input:  sampleMetadata.csv (cMD3, project root)
Output: splits/geography/geo_metadata_round{1..10}.csv  (per round)
        data/geo_metadata_full.csv                      (all geography-eligible samples)

30 countries (n>=50), 10 rounds x ~1600 samples.
Proportional-floor sampling: each country contributes min(count, cap) samples.
cap = round(TARGET_N / n_countries) so total ≈ TARGET_N.
80/20 participant-level train/test split, stratified by country.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR   = os.path.join(os.path.dirname(__file__), "../data")
SPLITS_DIR = os.path.join(os.path.dirname(__file__), "../splits/geography")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)

META_CSV   = os.path.join(ROOT, "sampleMetadata.csv")
N_ROUNDS   = 10
TARGET_N   = 1600
BASE_SEED  = 99
MIN_N      = 50  # country eligibility threshold


def make_labels(smd: pd.DataFrame) -> pd.DataFrame:
    geo = smd[smd["country"].notna()].copy()
    country_counts = geo["country"].value_counts()
    eligible = country_counts[country_counts >= MIN_N].index
    geo = geo[geo["country"].isin(eligible)].copy()
    geo["label"] = geo["country"]
    return geo


def stratified_subsample(df: pd.DataFrame, n_total: int, seed: int) -> pd.DataFrame:
    countries = df["label"].unique()
    n_per_country = round(n_total / len(countries))
    rng = np.random.default_rng(seed)
    parts = []
    for country in countries:
        subset = df[df["label"] == country].copy()
        n_take = min(len(subset), max(n_per_country, MIN_N))
        if len(subset) <= n_take:
            parts.append(subset)
        else:
            parts.append(subset.sample(n=n_take, random_state=int(rng.integers(1e6))))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def add_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    id_col = "subject_id" if "subject_id" in df.columns else "sample_id"
    df = df.copy()
    df[id_col] = df[id_col].fillna(df["sample_id"])
    subj_label = (
        df.groupby(id_col)["label"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
        .rename(columns={"label": "subj_label"})
    )
    # encode for stratification (train_test_split requires numeric)
    le_map = {c: i for i, c in enumerate(subj_label["subj_label"].unique())}
    strat = subj_label["subj_label"].map(le_map).values
    train_subj, _ = train_test_split(
        subj_label[id_col].values,
        test_size=0.2, random_state=seed,
        stratify=strat,
    )
    train_set = set(train_subj)
    df["split"] = df[id_col].map(lambda s: "train" if s in train_set else "test")
    return df


def main():
    smd = pd.read_csv(META_CSV, low_memory=False)
    geo = make_labels(smd)
    countries = sorted(geo["label"].unique())
    print(f"Geography samples: {len(geo)}  Countries: {len(countries)}")
    for c in countries:
        print(f"  {c}: {(geo['label']==c).sum()}")

    keep_base = ["sample_id", "subject_id", "study_name", "country", "label", "NCBI_accession"]

    # Full dataset
    full = add_split(geo, BASE_SEED)
    keep_cols = [c for c in keep_base + ["split"] if c in full.columns]
    full[keep_cols].to_csv(os.path.join(DATA_DIR, "geo_metadata_full.csv"), index=False)
    print(f"\nFull metadata saved ({len(full)} samples)")

    all_sub_ids = set()
    for r in range(1, N_ROUNDS + 1):
        seed = BASE_SEED + r * 100
        sub  = stratified_subsample(geo, TARGET_N, seed)
        sub  = add_split(sub, seed)
        keep_cols = [c for c in keep_base + ["split"] if c in sub.columns]
        n_tr = (sub["split"] == "train").sum()
        n_te = (sub["split"] == "test").sum()
        print(f"Round {r} (seed={seed}): {len(sub)} samples | "
              f"{sub['label'].nunique()} countries | train={n_tr} test={n_te}")
        sub[keep_cols].to_csv(
            os.path.join(SPLITS_DIR, f"geo_metadata_round{r}.csv"), index=False
        )
        all_sub_ids.update(sub["sample_id"].tolist())

    print(f"\nUnique samples across all {N_ROUNDS} rounds: {len(all_sub_ids)}")


if __name__ == "__main__":
    main()
