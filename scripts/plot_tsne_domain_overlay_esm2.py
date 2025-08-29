# scripts/plot_tsne_domain_overlay_esm2.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from Bio import SeqIO
import pandas as pd
from matplotlib.lines import Line2D

# === Paths ===
REAL_EMB = "embeddings/esm2_embeddings.npy"
GEN_EMB = "embeddings/sampled_esm2_embeddings.npy"
CURATED_META = "data/curated_thioredoxin_reductase.fasta"
BLAST_META = "data/decoded_embeddiff_esm2.fasta"
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# === Load Embeddings ===
real_emb = np.load(REAL_EMB)
gen_emb = np.load(GEN_EMB)

# === Load FASTA Metadata ===
curated_records = list(SeqIO.parse(CURATED_META, "fasta"))
curated_df = pd.DataFrame({
    "sequence_id": [f"real_{i}" for i in range(len(curated_records))],
    "source": ["Real"] * len(curated_records),
    "description": [r.description for r in curated_records]
})

blast_records = list(SeqIO.parse(BLAST_META, "fasta"))
blast_df = pd.DataFrame({
    "sequence_id": [f"gen_{i}" for i in range(len(blast_records))],
    "source": ["Generated"] * len(blast_records),
    "description": [r.description for r in blast_records]
})

# === Infer domain from description ===
def infer_domain(description):
    desc = description.lower()
    if any(term in desc for term in ["bacteria", "eubacteria", "proteobacteria", "firmicutes"]):
        return "bacteria"
    elif any(term in desc for term in ["fungi", "ascomycota", "basidiomycota"]):
        return "fungi"
    elif "archaea" in desc or "archaeon" in desc:
        return "archaea"
    else:
        return "unknown"

curated_df["Domain"] = curated_df["description"].apply(infer_domain)
blast_df["Domain"] = "Generated"

# === Combine metadata and embeddings ===
all_emb = np.vstack([real_emb, gen_emb])
all_meta = pd.concat([
    curated_df[["sequence_id", "source", "Domain"]],
    blast_df[["sequence_id", "source", "Domain"]]
], ignore_index=True)

# === Match lengths and filter NaNs ===
min_len = min(len(all_emb), len(all_meta))
all_emb = all_emb[:min_len]
all_meta = all_meta.iloc[:min_len].reset_index(drop=True)

valid_mask = ~np.isnan(all_emb).any(axis=1)
all_emb = all_emb[valid_mask]
all_meta = all_meta.loc[valid_mask].reset_index(drop=True)

# === Run t-SNE ===
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
tsne_coords = tsne.fit_transform(all_emb)
all_meta["tSNE 1"] = tsne_coords[:, 0]
all_meta["tSNE 2"] = tsne_coords[:, 1]

# === Plot setup ===
palette = {
    "bacteria": "#2ca02c",
    "fungi": "#ff7f0e",
    "archaea": "#1f77b4"
}

sns.set(style="ticks", font="Arial")
plt.figure(figsize=(10, 7))

# Plot generated embeddings in black
generated = all_meta[all_meta["source"] == "Generated"]
sns.scatterplot(
    data=generated,
    x="tSNE 1", y="tSNE 2",
    color="black",
    alpha=1,
    s=90,
    edgecolor="black",
    linewidth=0.3
)

# Plot real embeddings by domain
real = all_meta[(all_meta["source"] == "Real") & (all_meta["Domain"].isin(palette.keys()))]
sns.scatterplot(
    data=real,
    x="tSNE 1", y="tSNE 2",
    hue="Domain",
    palette=palette,
    s=100,
    edgecolor="black"
)

# === Formatting ===
plt.xlabel("t-SNE 1", fontsize=14, fontweight="bold")
plt.ylabel("t-SNE 2", fontsize=14, fontweight="bold")
plt.xticks(fontsize=12, fontweight="bold")
plt.yticks(fontsize=12, fontweight="bold")
plt.grid(False)

# === Custom legend with circular "Generated" marker ===
handles, labels = plt.gca().get_legend_handles_labels()
circle_handle = Line2D(
    [], [], marker='o', color='black', label='Generated',
    markersize=9, linestyle='None', markeredgecolor='black'
)
handles.append(circle_handle)
plt.legend(handles=handles, title="Domain (ESM-2)", title_fontsize=13, fontsize=12, frameon=True)

# === Save ===
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig5f_tsne_domain_overlay_esm2.png"), dpi=300)
plt.savefig(os.path.join(OUT_DIR, "fig5f_tsne_domain_overlay_esm2.svg"))
plt.close()

print("✅ ESM-2 t-SNE domain overlay saved.")
