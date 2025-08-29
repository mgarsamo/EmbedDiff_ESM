# scripts/plot_entropy_identity_esm2.py

import os
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from collections import Counter
import math
import pandas as pd

# === Config ===
MAX_LEN = 256
ENTROPY_THRESHOLD = 2.8
MIN_IDENTITY = 30.0
MAX_IDENTITY = 85.0
FASTA_PATH = "data/decoded_embeddiff_esm2.fasta"

# === Helper: Compute entropy ===
def sequence_entropy(seq):
    counter = Counter(seq)
    probs = [v / len(seq) for v in counter.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

# === Load sequences & compute ===
data = []
for record in SeqIO.parse(FASTA_PATH, "fasta"):
    seq = str(record.seq)
    entropy = sequence_entropy(seq)
    try:
        identity = float(record.description.split("id")[1])
    except:
        identity = None
    data.append({"seq": seq, "entropy": entropy, "identity": identity})

df = pd.DataFrame(data)
df["status"] = df.apply(
    lambda row: "accepted" if row["entropy"] >= ENTROPY_THRESHOLD and 
                              row["identity"] is not None and 
                              MIN_IDENTITY <= row["identity"] <= MAX_IDENTITY 
                 else "filtered_out",
    axis=1
)

# === Plot ===
plt.figure(figsize=(8, 6))
accepted = df[df["status"] == "accepted"]
rejected = df[df["status"] == "filtered_out"]

plt.scatter(accepted["identity"], accepted["entropy"], alpha=0.7, label="Accepted")
# Optional: visualize filtered-out
# plt.scatter(rejected["identity"], rejected["entropy"], alpha=0.5, color="red", marker="x", label="Filtered Out")

plt.axhline(ENTROPY_THRESHOLD, linestyle='--', color='gray', label="Entropy Threshold")
plt.axvline(MIN_IDENTITY, linestyle='--', color='blue', label="Min Identity")
plt.axvline(MAX_IDENTITY, linestyle='--', color='blue', label="Max Identity")
plt.xlabel("Percent Identity", weight="bold")
plt.ylabel("Shannon Entropy", weight="bold")
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("figures/fig5c_entropy_scatter_esm2.png", dpi=300)
plt.close()

print("✅ Entropy vs Identity plot saved for ESM-2 embeddings.")
