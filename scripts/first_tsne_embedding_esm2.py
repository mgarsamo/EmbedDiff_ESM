# scripts/first_tsne_embedding_esm2.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from Bio import SeqIO
from matplotlib.font_manager import FontProperties
import os

# === Paths ===
embedding_path = "embeddings/esm2_embeddings.npy"
fasta_path = "data/curated_thioredoxin_reductase.fasta"
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

# === Load embeddings ===
embeddings = np.load(embedding_path)

# === Parse domains from FASTA ===
domains = []

def infer_domain(description):
    desc = description.lower()
    if any(word in desc for word in ["bacteria", "eubacteria", "proteobacteria", "firmicutes"]):
        return "bacteria"
    elif any(word in desc for word in ["fungi", "ascomycota", "basidiomycota"]):
        return "fungi"
    elif "archaea" in desc or "archaeon" in desc:
        return "archaea"
    else:
        return "unknown"

records = list(SeqIO.parse(fasta_path, "fasta"))
for record in records:
    domains.append(infer_domain(record.description))

domains = np.array(domains)

# === Run t-SNE ===
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
tsne_result = tsne.fit_transform(embeddings)

# === Create DataFrame for plotting ===
df = pd.DataFrame({
    "tSNE 1": tsne_result[:, 0],
    "tSNE 2": tsne_result[:, 1],
    "Domain": domains
})

# === Plot settings ===
sns.set(style="ticks", font="Arial")
palette = {
    "bacteria": "#2ca02c",  # green
    "fungi": "#ff7f0e",     # orange
    "archaea": "#1f77b4",   # blue
    "unknown": "#d62728"    # red
}

plt.figure(figsize=(10, 7))
ax = sns.scatterplot(
    data=df,
    x="tSNE 1", y="tSNE 2",
    hue="Domain",
    palette=palette,
    s=100,
    edgecolor="black"
)

# === Axes formatting ===
plt.xlim(-70, 70)
plt.ylim(-70, 70)
plt.xlabel("t-SNE 1", fontsize=18, fontweight="bold")
plt.ylabel("t-SNE 2", fontsize=18, fontweight="bold")
plt.xticks(fontsize=12, fontweight="bold")
plt.yticks(fontsize=12, fontweight="bold")
plt.grid(False)

# === Legend formatting with bold title ===
font_props = FontProperties(weight='bold', family='Arial', size=14)
legend = ax.legend(
    title="Thioredoxin Reductase (ESM-2)",
    title_fontsize=14,
    prop=font_props,
    frameon=True,
    edgecolor="black",
    loc='upper left'
)

legend.get_title().set_fontweight('bold')

# === Save ===
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig_tsne_by_domain_esm2.png"), dpi=300)
plt.savefig(os.path.join(output_dir, "fig_tsne_by_domain_esm2.svg"))
plt.close()

print("✅ Saved ESM-2 t-SNE plot to figures/")
