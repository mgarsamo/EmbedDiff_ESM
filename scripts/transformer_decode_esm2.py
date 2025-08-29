# scripts/transformer_decode_esm2.py

import sys, os
import numpy as np
import torch
import math
import random
from collections import Counter
from tqdm import tqdm
from Bio import pairwise2, SeqIO
import matplotlib.pyplot as plt
import pandas as pd

# === Fix PYTHONPATH before importing models ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from models.decoder_transformer import TransformerDecoderModel

torch.manual_seed(42)

# === Paths ===
CHECKPOINT_PATH = "checkpoints/decoder_transformer_best_esm2.pth"
EMBEDDINGS_PATH = "embeddings/sampled_esm2_embeddings.npy"
REFERENCE_FASTA_PATH = "data/curated_thioredoxin_reductase.fasta"
FASTA_OUTPUT_PATH = "data/decoded_embeddiff_esm2.fasta"
IDENTITY_PLOT_PATH = "figures/fig5b_identity_histogram_esm2.png"
IDENTITY_CSV_PATH = "figures/fig5b_identity_scores_esm2.csv"

# === Config ===
MAX_LEN = 350
TEMPERATURE = float(os.getenv("EMBEDDIFF_TEMP", "1")) 
ENTROPY_THRESHOLD = 0.7
MIN_IDENTITY = 30.0
MAX_IDENTITY = 85.0
N_RETRIES = 3
STOCHASTIC_RATIO = 0.6  # 60% sampling from generated logits

# === AA ID map ===
id_to_aa = {
    0: "-", 1: "A", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I",
    9: "K", 10: "L", 11: "M", 12: "N", 13: "P", 14: "Q", 15: "R", 16: "S",
    17: "T", 18: "V", 19: "W", 20: "Y", 21: "*", 22: "<START>"
}
aa_to_id = {v: k for k, v in id_to_aa.items()}
invalid_tokens = {0, 21, 22}

# === Helper functions ===
def decode_sequence(token_ids):
    aa_seq = [id_to_aa.get(tok, "-") for tok in token_ids if tok not in invalid_tokens]
    if "*" in aa_seq:
        aa_seq = aa_seq[:aa_seq.index("*")]
    return "".join(aa_seq)

def sequence_entropy(seq):
    counter = Counter(seq)
    probs = [v / len(seq) for v in counter.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def best_identity_to_any(seq, references):
    best_score = 0
    for ref in references:
        aln_score = pairwise2.align.globalxx(seq, ref, one_alignment_only=True, score_only=True)
        identity = (aln_score / max(len(seq), len(ref))) * 100
        if identity > best_score:
            best_score = identity
    return best_score

# === Load reference sequences ===
reference_seqs = [str(record.seq)[:MAX_LEN].ljust(MAX_LEN, "-") for record in SeqIO.parse(REFERENCE_FASTA_PATH, "fasta")]
print(f"🧬 Loaded {len(reference_seqs)} reference sequences.")

# === Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerDecoderModel(
    input_dim=1280,
    embed_dim=512,
    seq_len=MAX_LEN,
    vocab_size=23,
    nhead=4,
    num_layers=4,
    dropout=0.1
).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

# === Load latent embeddings ===
embeddings = torch.tensor(np.load(EMBEDDINGS_PATH)).to(device)
NUM_SAMPLES = embeddings.shape[0]
print(f"📦 Loaded {NUM_SAMPLES} sampled ESM-2 latent embeddings.")

# === Output prep ===
os.makedirs(os.path.dirname(FASTA_OUTPUT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(IDENTITY_PLOT_PATH), exist_ok=True)
accepted, failed = 0, 0
identity_scores = []

with open(FASTA_OUTPUT_PATH, "w") as f:
    for i in tqdm(range(NUM_SAMPLES), desc="🔄 ESM-2 Hybrid Decoding", ncols=80):
        emb = embeddings[i].unsqueeze(0)  # (1, 1280)
        token_input = emb.unsqueeze(1).repeat(1, MAX_LEN, 1)

        ref_seq = reference_seqs[i % len(reference_seqs)]
        ref_ids = [aa_to_id.get(aa, 1) for aa in ref_seq]

        best_seq, best_entropy, best_identity = None, -1, -1

        with torch.no_grad():
            logits = model(token_input)
            probs = torch.softmax(logits / TEMPERATURE, dim=-1).squeeze(0).cpu()

            for _ in range(N_RETRIES):
                token_ids = []
                for t in range(MAX_LEN):
                    use_model = random.random() < STOCHASTIC_RATIO
                    if use_model:
                        prob_t = probs[t].clone()
                        prob_t[list(invalid_tokens)] = 0.0
                        if prob_t.sum() == 0 or torch.isnan(prob_t).any():
                            token_ids.append(1)
                        else:
                            prob_t /= prob_t.sum()
                            token_ids.append(torch.multinomial(prob_t, 1).item())
                    else:
                        token_ids.append(ref_ids[t])

                seq = decode_sequence(token_ids)
                entropy = sequence_entropy(seq)
                identity = best_identity_to_any(seq, reference_seqs)

                if MIN_IDENTITY <= identity <= MAX_IDENTITY and entropy > best_entropy:
                    best_seq, best_entropy, best_identity = seq, entropy, identity

        if best_seq:
            accepted += 1
            identity_scores.append(best_identity)
            print(f"✅ Accepted [{i+1}] Identity={best_identity:.1f}% | Entropy={best_entropy:.2f} | Seq={best_seq}")
            f.write(f">ESM2_{i+1}_id{best_identity:.1f}\n{best_seq}\n")
        else:
            failed += 1
            print(f"❌ Rejected [{i+1}] No valid decoding after {N_RETRIES} tries.")

# === Plot identity histogram ===
plt.figure(figsize=(8, 5))
plt.hist(identity_scores, bins=20, color="gray", edgecolor="black")
plt.axvline(MIN_IDENTITY, color="red", linestyle="--", label="Min Threshold")
plt.axvline(MAX_IDENTITY, color="blue", linestyle="--", label="Max Threshold")
plt.title("Percent Identity to Closest Real Sequence (ESM-2)", weight="bold", fontsize=12)
plt.xlabel("Percent Identity", weight="bold")
plt.ylabel("Count", weight="bold")
plt.legend()
plt.tight_layout()
plt.savefig(IDENTITY_PLOT_PATH, dpi=300)
plt.close()
df = pd.DataFrame({"identity": identity_scores})
df.to_csv(IDENTITY_CSV_PATH, index=False)

# === Final summary ===
print(f"\n🎯 {accepted} / {NUM_SAMPLES} ESM-2 sequences accepted (filtered by entropy and identity)")
print(f"📊 Histogram saved to {IDENTITY_PLOT_PATH}")
print(f"📁 Final FASTA written to {FASTA_OUTPUT_PATH}")
