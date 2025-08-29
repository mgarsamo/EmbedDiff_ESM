# Activate your .venv
#source .venv/bin/activate

import os
import subprocess
import argparse
import sys

def run_command(command, description, skip_steps, step_key):
    if step_key in skip_steps:
        print(f"⏭️ Skipping: {description}")
        return
    print(f"\n✅ {description}")
    print(f"🔧 Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"❌ Failed at: {description}")

def main():
    parser = argparse.ArgumentParser(description="Run the full EmbedDiff-ESM2 pipeline.")
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        help="List of step keys to skip (e.g. esm2 logreg tsne diffusion decoder blast html)"
    )
    args = parser.parse_args()
    skip_steps = set(args.skip)

    fasta_path = "data/curated_thioredoxin_reductase.fasta"
    print(f"\n✅ Dataset prepared at: {fasta_path}")

    run_command(
        f"python utils/esm2_embedder.py --input {fasta_path} --output embeddings/esm2_embeddings.npy",
        "Step 2a: ESM-2 embedding of real sequences",
        skip_steps, "esm2"
    )

    # Logistic regression probe to analyze embedding quality and domain separation
    run_command(
        "python scripts/logistic_regression_probe_esm2.py",
        "Step 2b: Logistic Regression Probe on ESM-2 embeddings",
        skip_steps, "logreg"
    )

    run_command(
        "python scripts/first_tsne_embedding_esm2.py",
        "Step 2c: Plot t-SNE of real ESM-2 embeddings",
        skip_steps, "tsne"
    )

    run_command(
        "python scripts/train_embeddiff_esm2.py",
        "Step 3: Train EmbedDiff-ESM2 latent diffusion model",
        skip_steps, "diffusion"
    )

    run_command(
        "python scripts/sample_embeddings_esm2.py",
        "Step 4: Sample synthetic embeddings from EmbedDiff-ESM2",
        skip_steps, "sample"
    )

    run_command(
        "python scripts/build_decoder_dataset_esm2.py",
        "Step 5a: Build decoder dataset from real ESM-2 embeddings",
        skip_steps, "decoder_data"
    )

    run_command(
        "python scripts/train_transformer_esm2.py",
        "Step 5b: Train Transformer decoder (ESM-2)",
        skip_steps, "decoder_train"
    )

    run_command(
        "python scripts/transformer_decode_esm2.py",
        "Step 6: Decode synthetic embeddings to amino acid sequences (ESM-2)",
        skip_steps, "decode"
    )

    run_command(
        "python scripts/plot_tsne_domain_overlay_esm2.py",
        "Step 7a: Overlay real vs. generated embeddings via t-SNE (ESM-2)",
        skip_steps, "tsne_overlay"
    )

    run_command(
        "python scripts/cosine_similarity_esm2.py",
        "Step 7b: Plot cosine similarity histogram (ESM-2)",
        skip_steps, "cosine"
    )

    run_command(
        "python scripts/plot_entropy_identity_esm2.py",
        "Step 7c: Plot entropy vs. sequence identity (ESM-2)",
        skip_steps, "entropy"
    )

    run_command(
        "python scripts/blastlocal_esm2.py",
        "Step 7d: Run local BLAST and summarize results (ESM-2)",
        skip_steps, "blast"
    )

    # === Final HTML Report ===
    if "html" in skip_steps:
        print("⏭️ Skipping: Step 8 - Generate HTML Summary Report (ESM-2)")
    else:
        print("\n✅ Step 8: Generate HTML Summary Report (ESM-2)")
        run_command(
            "python scripts/generate_esm2_report.py",
            "Step 8: Generate HTML Summary Report (ESM-2)",
            skip_steps, "html"
        )

    print("\n🎉 All steps in the EmbedDiff-ESM2 pipeline completed successfully!")

if __name__ == "__main__":
    main()
