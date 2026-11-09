"""
Top discriminative k-mer → species attribution via correlation.

For IBD vs nonIBD (HMP2):
  1. Welch t-test on k-mer frequencies → top 100 IBD-enriched k-mers
  2. Decode k-mer index → ACGT sequence
  3. Spearman correlation: each top k-mer × each species abundance
  4. Heatmap: which species best explain discriminative k-mers?

Output: figures/kmer_attribution_heatmap.png
        figures/kmer_attribution_heatmap.pdf
        kmer_attribution_top.csv
"""

import os, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

ROOT        = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 6.5,
})

DECODE = {0: "A", 1: "C", 2: "G", 3: "T"}
K = 8

KNOWN_IBD = {
    "depleted":  ["prausnitzii", "Roseburia", "Akkermansia", "Lachnospiraceae",
                  "Ruminococcus", "Bifidobacterium"],
    "enriched":  ["Escherichia", "Fusobacterium", "Ruminococcus_gnavus",
                  "Bacteroides_fragilis", "Clostridium"],
}


def decode_kmer(idx, k=K):
    bases = []
    for _ in range(k):
        bases.append(DECODE[idx & 3])
        idx >>= 2
    return "".join(reversed(bases))


def revcomp(seq):
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(comp[b] for b in reversed(seq))


def main():
    # ── Load k-mer matrix ──────────────────────────────────────────────
    kmer_dir = os.path.join(ROOT, "../ibd/kmer_counts")
    X_km       = np.load(os.path.join(kmer_dir, "kmer_matrix_k8_X.npy"))
    kmer_index = np.load(os.path.join(kmer_dir, "kmer_matrix_k8_kmer_index.npy"))
    sample_ids = np.load(os.path.join(kmer_dir, "kmer_matrix_k8_sample_ids.npy"),
                         allow_pickle=True)
    labels     = np.load(os.path.join(kmer_dir, "kmer_matrix_k8_labels.npy"),
                         allow_pickle=True)

    # Normalise to relative frequency
    row_sums = X_km.sum(axis=1, keepdims=True)
    X_km_freq = X_km / np.where(row_sums == 0, 1, row_sums)

    # deduplicate sample_ids
    _, uniq_idx = np.unique(sample_ids, return_index=True)
    X_km_freq  = X_km_freq[uniq_idx]
    sample_ids = sample_ids[uniq_idx]
    labels     = labels[uniq_idx]
    y          = (labels == "IBD").astype(int)

    print(f"K-mer matrix: {X_km_freq.shape}, IBD={y.sum()}, nonIBD={(y==0).sum()}")

    # ── Load species abundance ─────────────────────────────────────────
    data_dir   = os.path.join(ROOT, "../ibd/data/raw")
    meta_path  = os.path.join(data_dir, "hmp2_metadata_2018-08-20.csv")
    taxon_path = os.path.join(data_dir, "taxonomic_profiles_3.tsv.gz")

    meta = pd.read_csv(meta_path, low_memory=False)
    meta = meta[meta["data_type"] == "metagenomics"]
    meta = meta[meta["diagnosis"].isin(["CD", "UC", "nonIBD"])]
    meta["diagnosis"] = meta["diagnosis"].replace({"CD": "IBD", "UC": "IBD"})
    meta = (meta[["External ID", "diagnosis"]]
            .drop_duplicates("External ID")
            .rename(columns={"External ID": "sample_id"})
            .set_index("sample_id"))

    print("Loading taxonomic profiles…")
    taxon = pd.read_csv(taxon_path, sep="\t", index_col=0)
    sp_mask = taxon.index.map(
        lambda x: x.count("|") == 6 and "s__" in x and "t__" not in x)
    species = taxon[sp_mask].copy()
    species.index = species.index.map(lambda x: x.split("|")[-1])
    species = species.T
    species.index = species.index.str.replace("_profile$", "", regex=True)
    species.index.name = "sample_id"
    species = species.astype(float)

    # prevalence filter
    prev = (species > 0).mean()
    species = species.loc[:, prev >= 0.10]
    print(f"Species (prevalence≥10%): {species.shape[1]}")

    # ── Match samples ──────────────────────────────────────────────────
    km_df = pd.DataFrame(X_km_freq, index=sample_ids)
    shared = km_df.index.intersection(species.index)
    km_shared  = km_df.loc[shared].values
    ab_shared  = species.loc[shared].values
    sp_names   = species.columns.tolist()
    y_shared   = pd.Series(y, index=sample_ids).loc[shared].values
    print(f"Shared samples: {len(shared)}")

    # ── Step 1: Welch t-test → top 100 IBD-enriched k-mers ───────────
    print("Running Welch t-test on k-mer frequencies…")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pvals = stats.ttest_ind(
            km_shared[y_shared == 0], km_shared[y_shared == 1],
            equal_var=False, axis=0)
    pvals = np.nan_to_num(pvals, nan=1.0)
    _, p_adj, _, _ = multipletests(pvals, method="fdr_bh")

    # mean difference (IBD - nonIBD): positive = enriched in IBD
    delta = km_shared[y_shared == 1].mean(axis=0) - km_shared[y_shared == 0].mean(axis=0)

    # top 50 enriched in IBD (delta > 0, smallest p-values)
    enriched_mask = delta > 0
    pvals_ibd = np.where(enriched_mask, p_adj, 1.0)
    top_idx   = np.argsort(pvals_ibd)[:50]

    # top 50 depleted in IBD (delta < 0, smallest p-values)
    depleted_mask = delta < 0
    pvals_nonIBD = np.where(depleted_mask, p_adj, 1.0)
    bot_idx = np.argsort(pvals_nonIBD)[:50]

    print(f"Top enriched BH-adjusted p-range:  {p_adj[top_idx[0]]:.2e} – {p_adj[top_idx[-1]]:.2e}")
    print(f"Top depleted BH-adjusted p-range:  {p_adj[bot_idx[0]]:.2e} – {p_adj[bot_idx[-1]]:.2e}")

    # ── Step 2: Decode k-mer sequences ────────────────────────────────
    def idx_to_seq(col_idx):
        ki = int(kmer_index[col_idx])
        seq = decode_kmer(ki)
        rc  = revcomp(seq)
        return seq if seq <= rc else rc  # canonical form

    top_seqs = [idx_to_seq(i) for i in top_idx]
    bot_seqs = [idx_to_seq(i) for i in bot_idx]

    # ── Step 3: Spearman correlation k-mer vs species ─────────────────
    print("Computing Spearman correlations (enriched k-mers × species)…")
    def compute_corr_matrix(km_cols_idx, seqs):
        rows = []
        for ci, seq in zip(km_cols_idx, seqs):
            kv = km_shared[:, ci]
            corrs = []
            for si in range(ab_shared.shape[1]):
                sv = ab_shared[:, si]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r, _ = spearmanr(kv, sv)
                corrs.append(r if not np.isnan(r) else 0.0)
            rows.append(corrs)
        return pd.DataFrame(rows, index=seqs, columns=sp_names)

    corr_top = compute_corr_matrix(top_idx, top_seqs)
    corr_bot = compute_corr_matrix(bot_idx, bot_seqs)

    # ── Step 4: Find top species per direction ────────────────────────
    # For IBD-enriched k-mers: species most positively correlated (same direction)
    mean_corr_top = corr_top.mean(axis=0).sort_values(ascending=False)
    mean_corr_bot = corr_bot.mean(axis=0).sort_values()          # most negatively correlated

    # Select top 12 species for each direction by mean absolute correlation
    sp_show_top = mean_corr_top.head(8).index.tolist() + mean_corr_top.tail(4).index.tolist()
    sp_show_bot = mean_corr_bot.head(4).index.tolist() + mean_corr_bot.tail(8).index.tolist()
    sp_show = list(dict.fromkeys(sp_show_top + sp_show_bot))  # deduplicate

    # ── Step 5: Save top results ──────────────────────────────────────
    result_rows = []
    for ci, seq in zip(top_idx, top_seqs):
        corr_row = corr_top.loc[seq]
        top_sp   = corr_row.abs().sort_values(ascending=False).head(3)
        result_rows.append({
            "kmer":        seq,
            "direction":   "IBD-enriched",
            "delta_freq":  round(float(delta[ci]), 6),
            "p_value":     float(pvals[ci]),
            "p_adj":       float(p_adj[ci]),
            "top_species_1": top_sp.index[0],
            "corr_1":      round(top_sp.iloc[0], 3),
            "top_species_2": top_sp.index[1],
            "corr_2":      round(top_sp.iloc[1], 3),
            "top_species_3": top_sp.index[2],
            "corr_3":      round(top_sp.iloc[2], 3),
        })
    for ci, seq in zip(bot_idx, bot_seqs):
        corr_row = corr_bot.loc[seq]
        top_sp   = corr_row.abs().sort_values(ascending=False).head(3)
        result_rows.append({
            "kmer":        seq,
            "direction":   "IBD-depleted",
            "delta_freq":  round(float(delta[ci]), 6),
            "p_value":     float(pvals[ci]),
            "p_adj":       float(p_adj[ci]),
            "top_species_1": top_sp.index[0],
            "corr_1":      round(top_sp.iloc[0], 3),
            "top_species_2": top_sp.index[1],
            "corr_2":      round(top_sp.iloc[1], 3),
            "top_species_3": top_sp.index[2],
            "corr_3":      round(top_sp.iloc[2], 3),
        })

    result_df = pd.DataFrame(result_rows)
    out_csv = os.path.join(ROOT, "kmer_attribution_top.csv")
    result_df.to_csv(out_csv, index=False)
    print(f"Attribution table → {out_csv}")

    # ── Step 6: Heatmap ───────────────────────────────────────────────
    # Combine representative enriched + depleted k-mers for a readable main figure.
    n_show = 20
    corr_combined = pd.concat([corr_top.iloc[:n_show], corr_bot.iloc[:n_show]])[sp_show]

    # Short species names for display
    def short_name(s):
        s = s.replace("s__", "")
        parts = s.split("_")
        if len(parts) >= 2:
            return f"{parts[0][0]}. {' '.join(parts[1:])}"
        return s
    corr_combined.columns = [short_name(c) for c in corr_combined.columns]

    # Row labels: seq + arrow
    row_labels = [f"↑ {s}" for s in top_seqs[:n_show]] + [f"↓ {s}" for s in bot_seqs[:n_show]]
    corr_combined.index = row_labels

    fig, ax = plt.subplots(figsize=(max(9.2, len(sp_show) * 0.72), 8.6))
    data_arr = corr_combined.values
    im = ax.imshow(data_arr, aspect="auto", cmap="RdBu_r", vmin=-0.4, vmax=0.4)
    plt.colorbar(im, ax=ax, label="Spearman r", shrink=0.55, pad=0.02)

    ax.set_xticks(range(len(corr_combined.columns)))
    ax.set_xticklabels(corr_combined.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_combined.index)))
    ax.set_yticklabels(corr_combined.index)
    for i, label in enumerate(ax.get_yticklabels()):
        label.set_color("#d55e00" if i < n_show else "#0072b2")

    # divider between enriched / depleted sections
    ax.axhline(n_show - 0.5, color="black", lw=1.4)

    ax.set_xlabel("Species")
    ax.set_ylabel("Discriminative 8-mer")
    plt.tight_layout()
    out_fig = os.path.join(FIGURES_DIR, "kmer_attribution_heatmap.png")
    out_pdf = os.path.join(FIGURES_DIR, "kmer_attribution_heatmap.pdf")
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Heatmap → {out_fig}")
    print(f"Heatmap → {out_pdf}")

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("TOP 10 IBD-ENRICHED k-mers + best correlated species:")
    print(f"{'='*65}")
    for _, row in result_df[result_df["direction"] == "IBD-enriched"].head(10).iterrows():
        print(f"  {row['kmer']}  Δ={row['delta_freq']:+.5f}  p_adj={row['p_adj']:.2e}"
              f"  → {row['top_species_1']} (r={row['corr_1']:+.3f})")

    print(f"\nTOP 10 IBD-DEPLETED k-mers + best correlated species:")
    print(f"{'='*65}")
    for _, row in result_df[result_df["direction"] == "IBD-depleted"].head(10).iterrows():
        print(f"  {row['kmer']}  Δ={row['delta_freq']:+.5f}  p_adj={row['p_adj']:.2e}"
              f"  → {row['top_species_1']} (r={row['corr_1']:+.3f})")


if __name__ == "__main__":
    main()
