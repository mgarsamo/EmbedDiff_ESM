# scripts/sample_embeddings_esm2.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
from Bio import SeqIO
from matplotlib.font_manager import FontProperties

# === Paths ===
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMBEDDING_PATH = os.path.join(ROOT, "embeddings", "esm2_embeddings.npy")
FASTA_PATH = os.path.join(ROOT, "data", "curated_thioredoxin_reductase.fasta")
OUTPUT_PATH = os.path.join(ROOT, "embeddings", "sampled_esm2_embeddings.npy")
FIGURE_PATH = os.path.join(ROOT, "figures", "fig3a_generated_tsne_esm2")
TSNE_COORDS_PATH = os.path.join(ROOT, "embeddings", "tsne_coords_esm2.npy")
TSNE_LABELS_PATH = os.path.join(ROOT, "embeddings", "tsne_labels_esm2.npy")

# === Parameters ===
embedding_dim = 1280
samples_per_class = 80
noise_std = 0.03  # adjust if needed

# === Load class labels from FASTA headers ===
labels = []
for record in SeqIO.parse(FASTA_PATH, "fasta"):
    desc = record.description
    label = desc.split("[")[-1].split("]")[0].strip().lower()
    labels.append(label)

labels = pd.Series(labels, dtype="category")
class_names = list(labels.cat.categories)

# === Total number of samples
num_samples = len(class_names) * samples_per_class

# === Load real embeddings ===
real_embeddings = np.load(EMBEDDING_PATH)
real_embeddings = torch.tensor(real_embeddings, dtype=torch.float32)
real_labels = labels.tolist()

# === Random sampling from real embeddings ===
print(f"🎯 Sampling {num_samples} ESM-2 embeddings with mild noise (std={noise_std})...")
idx = torch.randint(0, real_embeddings.shape[0], (num_samples,))
selected = real_embeddings[idx]

# === Add mild Gaussian noise ===
noisy_embeddings = selected + noise_std * torch.randn_like(selected)
noisy_embeddings = noisy_embeddings.numpy()
np.save(OUTPUT_PATH, noisy_embeddings)
print(f"✅ Saved noisy ESM-2 embeddings to: {OUTPUT_PATH}")

# === Combine for t-SNE visualization ===
combined = np.vstack([real_embeddings.numpy(), noisy_embeddings])
combined_labels = [label.capitalize() for label in real_labels] + ["Generated"] * num_samples

# === Run t-SNE ===
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
tsne_result = tsne.fit_transform(combined)

# === Save t-SNE outputs ===
np.save(TSNE_COORDS_PATH, tsne_result)
np.save(TSNE_LABELS_PATH, np.array(combined_labels))
print(f"✅ Saved ESM-2 t-SNE coordinates to: {TSNE_COORDS_PATH}")
print(f"✅ Saved ESM-2 t-SNE labels to: {TSNE_LABELS_PATH}")

# === Plot style ===
sns.set(style="white", font_scale=1.4, rc={
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.framealpha": 1,
    "font.family": "Arial"
})
font_props = FontProperties(weight='bold', family='Arial', size=12)

# === Color map ===
color_map = {
    "Bacteria": "#2ca02c",   # green
    "Fungi": "#ff7f0e",      # orange
    "Archaea": "#1f77b4",    # blue
    "Generated": "#9467bd"   # purple
}

# === Plot t-SNE ===
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=tsne_result[:, 0],
    y=tsne_result[:, 1],
    hue=combined_labels,
    palette=color_map,
    s=100,
    alpha=0.9
)

plt.xlabel("t-SNE 1", font_properties=font_props)
plt.ylabel("t-SNE 2", font_properties=font_props)
plt.legend(
    title="Protein Class (ESM-2)",
    prop=font_props,
    title_fontproperties=font_props,
    frameon=True,
    loc='upper left'
)
plt.tight_layout()
plt.savefig(FIGURE_PATH + ".png", dpi=300)
plt.savefig(FIGURE_PATH + ".svg")
plt.close()

print("✅ ESM-2 t-SNE plot saved.")
