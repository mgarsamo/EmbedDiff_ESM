import numpy as np
import torch
from tqdm import tqdm
from Bio import SeqIO
from transformers import AutoTokenizer, EsmModel

def embed_sequences(fasta_file, output_npy, batch_size=8):
    """
    Embed protein sequences from a FASTA file using ESM-2 and save to a .npy file.

    Args:
        fasta_file (str): Path to the input FASTA file.
        output_npy (str): Path to save the output NumPy (.npy) file.
        batch_size (int): Number of sequences to embed in parallel.
    """
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]

    model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model.eval()

    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="🔬 Embedding sequences"):
            batch_seqs = sequences[i:i+batch_size]
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            all_embeddings.extend(embeddings)

    np.save(output_npy, np.array(all_embeddings))
    print(f"✅ Saved embeddings to {output_npy}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Embed protein sequences using ESM-2 and save to .npy file.")
    parser.add_argument(
        '--input',
        default='data/curated_thioredoxin_reductase.fasta',
        help='Path to the input FASTA file (default: data/curated_thioredoxin_reductase.fasta)'
    )
    parser.add_argument(
        '--output',
        default='embeddings/esm2_embeddings.npy',
        help='Path to save the output .npy file (default: embeddings/esm2_embeddings.npy)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for parallel embedding (default: 8)'
    )
    args = parser.parse_args()

    embed_sequences(args.input, args.output, args.batch_size)
