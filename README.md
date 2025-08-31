<!--
EmbedDiff is a deep learning pipeline for de novo protein design using latent diffusion models and ESM2 embeddings. It generates novel, biologically plausible protein sequences and includes decoding, BLAST validation, entropy filtering, and structure prediction using ESMFold or AlphaFold2. Ideal for machine learning in bioinformatics, protein engineering, and generative biology.
-->
# 🧬 EmbedDiff-ESM: Latent Diffusion Pipeline for De Novo Protein Sequence Generation

[![HTML Report](https://img.shields.io/badge/View%20Report-📊-orange)](https://mgarsamo.github.io/EmbedDiff_ESM/embeddiff_esm2_summary_report.html)
[![Run EmbedDiff](https://img.shields.io/badge/🚀-Run%20Pipeline-blue)](#-quick-start-1-liner)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/mgarsamo/EmbedDiff_ESM/blob/main/LICENSE)

**EmbedDiff-ESM2** is a comprehensive protein sequence generation pipeline that combines large-scale pretrained protein embeddings (ESM-2) with a latent diffusion model to explore and sample from the vast protein sequence space. It generates novel sequences that preserve semantic and evolutionary properties without relying on explicit structural data, and evaluates them through a suite of biologically meaningful analyses including logistic regression classification.

---

## 🚀 Quick Start (1-liner)

To run the entire EmbedDiff pipeline from end to end:

```bash
python run_embeddiff_pipeline.py
```

---

## 🔍 What Is EmbedDiff-ESM2?

EmbedDiff-ESM2 uses ESM-2 (Evolutionary Scale Modeling v2) to project protein sequences into a high-dimensional latent space rich in evolutionary and functional priors. A denoising latent diffusion model is trained to learn the distribution of these embeddings and generate new ones from random noise. These latent vectors represent plausible protein-like states and are decoded into sequences using a Transformer decoder with configurable stochastic sampling ratios.

The pipeline includes **logistic regression analysis** to evaluate embedding quality and domain separation, followed by comprehensive sequence validation via entropy analysis, cosine similarity, BLAST alignment, and embedding visualization (t-SNE, MDS). A final HTML report presents all figures and results in an interactive format.

---

## 📌 Pipeline Overview

The full EmbedDiff-ESM2 pipeline is modular and proceeds through the following stages:

### **Step 1: Input Dataset**
- Format: A curated FASTA file of real protein sequences (e.g., Thioredoxin reductases from different domains).
- Used as the basis for learning a latent protein representation and decoder training.

---

### **Step 2a: ESM-2 Embedding**
- The curated sequences are embedded using the `esm2_t33_650M_UR50D` model.
- This transforms each protein into a 1280-dimensional latent vector.
- These embeddings capture functional and evolutionary constraints without any structural input.

---

### **Step 2b: Logistic Regression Probe Analysis**
- **NEW**: Logistic regression classifier is trained on ESM-2 embeddings to evaluate domain separation.
- Analyzes how well embeddings can distinguish between different protein domains (e.g., archaea, bacteria, fungi).
- Generates confusion matrices and per-class recall plots to assess embedding quality.
- Provides quantitative metrics for embedding discriminative power.

---

### **Step 2c: t-SNE of Real Embeddings**
- t-SNE is applied to the real ESM-2 embeddings to visualize the structure of protein space.
- Serves as a baseline to later compare generated (synthetic) embeddings.

---

### **Step 3: Train EmbedDiff-ESM2 Latent Diffusion Model**

**Architecture Details:**
- **Noise Predictor**: Multi-layer perceptron (MLP) with **4 hidden layers** (1024→1024→512→1280)
- **Input Dimension**: 1280 (ESM-2 embedding size) + conditional domain labels + timestep embedding
- **Activation**: ReLU with LayerNorm and Dropout (0.2) for regularization
- **Conditional Input**: Domain-specific labels (archaea, bacteria, fungi) as one-hot encoded vectors

**Diffusion Process:**
- **Timesteps**: **1000 diffusion steps** for smooth noise scheduling
- **Noise Schedule**: **Cosine beta schedule** with improved stability (β ∈ [0.0001, 0.9999])
- **Forward Process**: Gradual addition of Gaussian noise following q(xₜ|x₀) = √ᾱₜx₀ + √(1-ᾱₜ)ε
- **Reverse Process**: Learned denoising using p_θ(xₜ₋₁|xₜ) with noise prediction

**Training Configuration:**
- **Batch Size**: 32 (optimized for stability)
- **Learning Rate**: 1e-4 (Adam optimizer)
- **Epochs**: 300 with early stopping
- **Data Split**: 80/10/10 (train/val/test) with stratified sampling by domain
- **Normalization**: ESM-2 embeddings scaled to [-1, 1] range using tanh scaling

**Loss Function**: Mean squared error (MSE) between predicted and actual noise: L = ||ε - ε_θ(xₜ, t)||²

This architecture enables the model to learn the complex distribution of protein embeddings and generate novel, biologically plausible latent representations through iterative denoising.

---

### **Step 4: Sample Synthetic Embeddings**
- Starting from pure Gaussian noise, the trained diffusion model is used to generate new latent vectors that resemble real protein embeddings.
- These latent samples are biologically plausible but unseen — representing de novo candidates.

---

### **Step 5a: Build Decoder Dataset**
- Real ESM-2 embeddings are paired with their corresponding amino acid sequences.
- This dataset is used to train a decoder to translate from embedding → sequence.

---

### **Step 5b: Train Transformer Decoder**
- A Transformer model is trained to autoregressively generate amino acid sequences from input embeddings.
- Label smoothing and entropy filtering are used to improve sequence diversity and biological plausibility.

---

### **Step 6: Decode Synthetic Sequences**

The synthetic embeddings from Step 4 are decoded into amino acid sequences using a **hybrid decoding strategy** that balances biological realism with diversity.

**Current Configuration:**
- **60%** of amino acid positions are generated **stochastically**, sampled from the decoder's output distribution.
- **40%** are **reference-guided**, biased toward residues from the closest matching natural sequence.

This configuration produces sequences with approximately **30-55% sequence identity** to known proteins—striking a practical balance between novelty and plausibility.

#### 💡 Modular and Adjustable
This decoding step is fully configurable:
- Setting the stochastic ratio to **100%** yields **fully de novo sequences**, maximizing novelty.
- Lower stochastic ratios (e.g., **20–30%**) increase similarity to natural proteins.
- The ratio can be adjusted in `scripts/transformer_decode_esm2.py`.

---

### **Step 7a: t-SNE Overlay**
- A combined t-SNE plot compares the distribution of real and generated embeddings.
- Useful for assessing whether synthetic proteins fall within plausible latent regions.

---

### **Step 7b: Cosine Similarity Analysis**
- Pairwise cosine distances are computed between:
  - Natural vs. Natural sequences
  - Natural vs. generated sequences
  - Generated vs. generated sequences
- This helps evaluate diversity and proximity to known protein embeddings.

---

### **Step 7c: Entropy vs. Identity Analysis**
- Each decoded protein sequence is evaluated using two key metrics:
  - **Shannon Entropy**: Quantifies amino acid diversity across the sequence.
  - **Sequence Identity (via BLAST)**: Measures similarity to known natural proteins.
- Sequences are filtered based on configurable entropy and identity thresholds.

---

### **Step 7d: Local BLAST Validation**
- Generated sequences are validated by aligning them against a **locally downloaded SwissProt database** using `blastp`.
- Outputs a CSV summary with percent identity, E-value, bit score, and alignment details.

---

### **Step 8: HTML Summary Report**
- All visualizations, metrics, and links to output files are compiled into an interactive HTML report.
- Includes logistic regression results, cosine plots, entropy scatter, identity histograms, and t-SNE projections.
- Allows easy inspection and sharing of results.

---

## 📂 Project Structure

```
EmbedDiff_ESM/
├── README.md                                    # 📘 Project overview and documentation
├── requirements.txt                             # 📦 Python dependencies
├── run_embeddiff_pipeline.py                   # 🚀 Master pipeline script
│
├── data/                                        # 📁 Input and output biological data
│   ├── curated_thioredoxin_reductase.fasta     # Input protein sequences
│   ├── decoded_embeddiff_esm2.fasta            # Generated sequences
│   ├── decoder_dataset_esm2.pt                 # Decoder training dataset
│   └── blast_results/                          # BLAST analysis results
│       ├── blast_summary_local_esm2.csv        # BLAST summary
│       └── [individual BLAST XML and FASTA files]
│
├── embeddings/                                  # 📁 Latent vector representations
│   ├── esm2_embeddings.npy                     # Real sequence embeddings
│   ├── esm2_stats.npz                          # Embedding statistics
│   ├── sampled_esm2_embeddings.npy             # Generated embeddings
│   ├── tsne_coords_esm2.npy                    # t-SNE coordinates
│   └── tsne_labels_esm2.npy                    # t-SNE labels
│
├── figures/                                     # 📁 All generated plots and reports
│   ├── fig_tsne_by_domain_esm2.png            # t-SNE by domain
│   ├── logreg_per_class_recall_esm2.png       # Logistic regression recall
│   ├── logreg_confusion_matrix_esm2.png        # Logistic regression confusion matrix
│   ├── fig2b_loss_esm2.png                    # Diffusion training loss
│   ├── fig3a_generated_tsne_esm2.png          # Generated embeddings t-SNE
│   ├── fig5a_decoder_loss_esm2.png            # Decoder training loss
│   ├── fig5a_real_real_cosine_esm2.png        # Real-Real cosine similarity
│   ├── fig5b_gen_gen_cosine_esm2.png          # Generated-Generated cosine similarity
│   ├── fig5c_real_gen_cosine_esm2.png         # Real-Generated cosine similarity
│   ├── fig5b_identity_histogram_esm2.png      # Identity histogram
│   ├── fig5c_entropy_scatter_esm2.png         # Entropy vs Identity scatter
│   ├── fig5d_all_histograms_esm2.png          # All histograms
│   ├── fig5f_tsne_domain_overlay_esm2.png     # t-SNE domain overlay
│   ├── logreg_classification_results_esm2.csv  # Logistic regression results
│   └── embeddiff_esm2_summary_report.html     # Final HTML report
│
├── scripts/                                     # 📁 Core processing scripts
│   ├── esm2_embedder.py                       # Step 2a: ESM-2 embedding
│   ├── logistic_regression_probe_esm2.py      # Step 2b: Logistic regression analysis
│   ├── first_tsne_embedding_esm2.py           # Step 2c: t-SNE of real embeddings
│   ├── train_embeddiff_esm2.py                # Step 3: Train latent diffusion model
│   ├── sample_embeddings_esm2.py              # Step 4: Sample new embeddings
│   ├── build_decoder_dataset_esm2.py          # Step 5a: Build decoder training set
│   ├── train_transformer_esm2.py              # Step 5b: Train decoder
│   ├── transformer_decode_esm2.py             # Step 6: Decode embeddings to sequences
│   ├── plot_tsne_domain_overlay_esm2.py       # Step 7a: t-SNE comparison
│   ├── cosine_similarity_esm2.py              # Step 7b: Cosine similarity plots
│   ├── plot_entropy_identity_esm2.py          # Step 7c: Entropy vs. identity filter
│   ├── blastlocal_esm2.py                     # Step 7d: Local BLAST alignment
│   └── generate_esm2_report.py                # Step 8: Generate final HTML report
│
├── models/                                      # 📁 ML model architectures
│   ├── latent_diffusion.py                     # EmbedDiff-ESM2 diffusion model
│   └── decoder_transformer.py                  # Transformer decoder
│
├── utils/                                       # 📁 Utility and helper functions
│   └── esm2_embedder.py                       # ESM-2 embedding utilities
│
└── checkpoints/                                # 📁 Model checkpoints
    ├── best_embeddiff_mlp_esm2.pth            # Best diffusion model
    ├── decoder_transformer_best_esm2.pth       # Best decoder model
    └── decoder_transformer_last_esm2.pth       # Last decoder checkpoint
```

---

## 🚀 Quick Start

### 1. **Setup Environment**
```bash
# Clone the repository
git clone <repository-url>
cd EmbedDiff_ESM

# Install dependencies
pip install -r requirements.txt
```

### 2. **Prepare Data**
- Place your curated protein sequences in `data/curated_thioredoxin_reductase.fasta`
- Ensure sequences are in FASTA format with domain information in descriptions

### 3. **Run Full Pipeline**
```bash
# Run complete pipeline
python run_embeddiff_pipeline.py

# Or skip specific steps
python run_embeddiff_pipeline.py --skip esm2 logreg tsne diffusion
```

### 4. **View Results**
- Check generated sequences in `data/decoded_embeddiff_esm2.fasta`
- View all visualizations in the `figures/` directory
- Open `embeddiff_esm2_summary_report.html` for comprehensive results

---

## 🔧 Configuration Options

### **Stochastic Ratio Adjustment**
Edit `scripts/transformer_decode_esm2.py`:
```python
STOCHASTIC_RATIO = 0.6  # 60% stochastic, 40% reference-guided
```

### **Pipeline Step Control**
Use the `--skip` flag to skip specific steps:
```bash
python run_embeddiff_pipeline.py --skip esm2 logreg tsne diffusion sample decoder_data decoder_train decode tsne_overlay cosine entropy blast html
```

---

## 📊 Key Features

- **🔍 Logistic Regression Analysis**: Evaluates embedding quality and domain separation
- **🎯 Configurable Sampling**: Adjustable stochastic ratios for sequence generation
- **📈 Comprehensive Analysis**: Multiple validation metrics and visualizations
- **🔄 Modular Pipeline**: Easy to skip steps or modify individual components
- **📱 Interactive Reports**: HTML reports with downloadable results
- **🧬 Biological Validation**: BLAST analysis against SwissProt database

---

## 🧪 Optional: Structural Validation

Generated sequences can be assessed for structural plausibility using:

- **[ESMFold](https://github.com/facebookresearch/esm)**: Fast structure prediction
- **[AlphaFold2](https://github.com/deepmind/alphafold)**: High-accuracy prediction

---

## 📊 Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Generated Sequences** | **240** | High-quality synthetic proteins with domain-specific conditioning |
| **Sequence Identity** | **37-49%** | Range of similarity to real sequences (BLAST validation) |
| **Training Epochs** | **300** | Diffusion model training with early stopping |
| **Batch Size** | **32** | Optimized for training stability |
| **Learning Rate** | **1e-4** | Adam optimizer configuration |
| **Timesteps** | **1000** | Diffusion process steps for smooth noise scheduling |
| **Embedding Dimension** | **1280** | ESM-2 latent space size |
| **Data Split** | **80/10/10** | Train/validation/test ratio with stratified sampling |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
