# scripts/cosine_similarity_esm2.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from matplotlib.font_manager import FontProperties

# === Paths ===
REAL_EMB = "embeddings/esm2_embeddings.npy"
GEN_EMB = "embeddings/sampled_esm2_embeddings.npy"
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# === Global Style ===
sns.set(style="white", font="Arial")
plt.rcParams.update({
    "font.size": 14,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "legend.title_fontsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.grid": False
})

bold_font = FontProperties(weight='bold', family='Arial', size=12)

# === Utility Functions ===
def load_embeddings(path):
    emb = np.load(path)
    if np.isnan(emb).any():
        print(f"⚠️ NaNs detected in {path}. Removing affected rows.")
        emb = emb[~np.isnan(emb).any(axis=1)]
    return emb

def plot_heatmap(matrix, x_label, y_label, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=False, yticklabels=False,
                cmap="coolwarm", annot=False, cbar_kws={"label": "Cosine Similarity"})
    plt.xlabel(x_label, weight="bold")
    plt.ylabel(y_label, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=300)
    plt.close()

def plot_mds(matrix, labels, title, filename):
    dist = 1 - matrix
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=4)
    coords = mds.fit_transform(dist)
    plt.figure(figsize=(10, 7))
    plt.scatter(coords[:, 0], coords[:, 1], color='black', s=80)
    for i, label in enumerate(labels):
        plt.text(coords[i, 0], coords[i, 1], label, fontsize=12, weight="bold", ha='center', va='center')
    plt.xlabel("MDS 1", weight="bold")
    plt.ylabel("MDS 2", weight="bold")
    plt.xticks(weight="bold")
    plt.yticks(weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=300)
    plt.close()

def plot_cosine_histograms(sim_matrices, labels, title, filename, bins=20):
    plt.figure(figsize=(10, 7))
    for matrix, label in zip(sim_matrices, labels):
        if matrix.shape[0] == matrix.shape[1]:
            values = matrix[~np.eye(matrix.shape[0], dtype=bool)]
        else:
            values = matrix.flatten()
        sns.histplot(values, bins=bins, kde=True, label=label, stat="density", edgecolor=None)

    plt.xlabel("Cosine Similarity", weight="bold")
    plt.ylabel("Density", weight="bold")
    plt.xticks(weight="bold")
    plt.yticks(weight="bold")
    plt.legend(
        title="Pair Type (ESM-2)",
        prop=bold_font,
        title_fontproperties=bold_font,
        frameon=True
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=300)
    plt.close()

# === Main Workflow ===
def main():
    real_emb = load_embeddings(REAL_EMB)
    gen_emb = load_embeddings(GEN_EMB)

    # Label sequences
    real_labels = [f"real_{i}" for i in range(len(real_emb))]
    gen_labels = [f"gen_{i}" for i in range(len(gen_emb))]

    # === Cosine Similarity Calculations ===
    rr_sim = cosine_similarity(real_emb)
    plot_heatmap(rr_sim, "Natural Sequences", "Natural Sequences", "fig5a_real_real_cosine_esm2.png")
    plot_mds(rr_sim, real_labels, "", "fig5a_real_real_mds_esm2.png")

    gg_sim = cosine_similarity(gen_emb)
    plot_heatmap(gg_sim, "Generated Sequences", "Generated Sequences", "fig5b_gen_gen_cosine_esm2.png")
    plot_mds(gg_sim, gen_labels, "", "fig5b_gen_gen_mds_esm2.png")

    rg_sim = cosine_similarity(real_emb, gen_emb)
    plot_heatmap(rg_sim, "Generated Sequences", "Natural Sequences", "fig5c_real_gen_cosine_esm2.png")
    print("✅ All ESM-2 cosine similarity figures saved to:", OUT_DIR)

    plot_cosine_histograms(
        sim_matrices=[rr_sim, gg_sim, rg_sim],
        labels=["Natural–Natural", "Gen–Gen", "Natural–Gen"],
        title="",
        filename="fig5d_all_histograms_esm2.png"
    )

if __name__ == "__main__":
    main()
