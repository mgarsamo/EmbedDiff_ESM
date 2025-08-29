# scripts/generate_esm2_report.py

import os
import base64
from datetime import datetime
import imghdr  # To validate image type

def generate_html_report(output_path="embeddiff_esm2_summary_report.html"):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    ordered_filenames = [
        "fig_tsne_by_domain_esm2.png",
        "logreg_per_class_recall_esm2.png",
        "logreg_confusion_matrix_esm2.png",
        "fig2b_loss_esm2.png",
        "fig3a_generated_tsne_esm2.png",
        "fig5a_decoder_loss_esm2.png",
        "fig5a_real_real_cosine_esm2.png",
        "fig5b_gen_gen_cosine_esm2.png",
        "fig5c_real_gen_cosine_esm2.png",
        "fig5b_identity_histogram_esm2.png",
        "fig5c_entropy_scatter_esm2.png",
        "fig5d_all_histograms_esm2.png",
        "fig5f_tsne_domain_overlay_esm2.png"
    ]

    figures_dir = "figures"
    plots = [fname for fname in ordered_filenames if os.path.exists(os.path.join(figures_dir, fname))]
    if not plots:
        print(f"⚠️ No valid ESM-2 image files found in {figures_dir}. Check file paths and extensions.")
        return

    blast_csv = "data/blast_results/blast_summary_local_esm2.csv"
    identity_csv = os.path.join(figures_dir, "fig5b_identity_scores_esm2.csv")
    decoded_fasta = "data/decoded_embeddiff_esm2.fasta"
    logreg_csv = os.path.join(figures_dir, "logreg_classification_results_esm2.csv")

    with open(output_path, "w") as f:
        f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EmbedDiff (ESM-2) Summary Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', sans-serif;
            background-color: #f4f6f8;
            color: #333;
            margin: 0;
            padding: 2em;
        }}
        h1 {{
            font-size: 2.2em;
            color: #1e3a8a;
            margin-bottom: 0.2em;
        }}
        h2 {{
            font-size: 1.5em;
            margin-top: 2em;
            border-bottom: 2px solid #ccc;
            padding-bottom: 0.3em;
        }}
        .grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 24px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 1em;
            flex: 1 1 45%;
            max-width: 45%;
        }}
        .card img {{
            width: 100%;
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            border: 1px solid #ddd;
            display: block;
        }}
        .figure-title {{
            font-weight: 600;
            margin-top: 0.5em;
            margin-bottom: 1em;
            font-size: 1em;
            color: #111827;
        }}
        a.download {{
            display: inline-block;
            margin: 10px 10px 0 0;
            font-weight: bold;
            color: #2563eb;
            text-decoration: none;
            border-bottom: 1px solid transparent;
        }}
        a.download:hover {{
            border-color: #2563eb;
        }}
    </style>
</head>
<body>
    <h1>🧬 EmbedDiff (ESM-2) Summary Report</h1>
    <p><strong>Date Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    <h2>🔍 Embedding Analysis & Classification</h2>
    <p>This section shows the analysis of ESM-2 embeddings including domain clustering and logistic regression classification results.</p>
    <div class="grid">
""")

        for plot in plots:
            full_path = os.path.join(figures_dir, plot)
            img_type = imghdr.what(full_path)
            if img_type:
                # Create more descriptive titles
                title_map = {
                    "fig_tsne_by_domain_esm2.png": "Figure 1: t-SNE by Domain (ESM-2)",
                    "logreg_per_class_recall_esm2.png": "Figure 2: Logistic Regression Per-Class Recall (ESM-2)",
                    "logreg_confusion_matrix_esm2.png": "Figure 3: Logistic Regression Confusion Matrix (ESM-2)",
                    "fig2b_loss_esm2.png": "Figure 4: Diffusion Training Loss (ESM-2)",
                    "fig3a_generated_tsne_esm2.png": "Figure 5: Generated Embeddings t-SNE (ESM-2)",
                    "fig5a_decoder_loss_esm2.png": "Figure 6: Transformer Decoder Loss (ESM-2)",
                    "fig5a_real_real_cosine_esm2.png": "Figure 7: Real-Real Cosine Similarity (ESM-2)",
                    "fig5b_gen_gen_cosine_esm2.png": "Figure 8: Generated-Generated Cosine Similarity (ESM-2)",
                    "fig5c_real_gen_cosine_esm2.png": "Figure 9: Real-Generated Cosine Similarity (ESM-2)",
                    "fig5b_identity_histogram_esm2.png": "Figure 10: Identity Histogram (ESM-2)",
                    "fig5c_entropy_scatter_esm2.png": "Figure 11: Entropy vs Identity Scatter (ESM-2)",
                    "fig5d_all_histograms_esm2.png": "Figure 12: All Histograms (ESM-2)",
                    "fig5f_tsne_domain_overlay_esm2.png": "Figure 13: t-SNE Domain Overlay (ESM-2)"
                }
                
                title = title_map.get(plot, plot.replace("fig", "Figure ").replace("_", " ").replace(".png", "").title())
                
                # Add section break after embedding analysis (first 3 figures)
                if plot == "logreg_confusion_matrix_esm2.png":
                    f.write("""
    </div>
    
    <h2>🚀 Diffusion & Generation Results</h2>
    <p>This section shows the results of the EmbedDiff latent diffusion model training and synthetic sequence generation.</p>
    <div class="grid">
""")
                
                f.write(f"""
        <div class="card">
            <div class="figure-title">{title}</div>
            <img src="figures/{plot}" alt="{title}">
        </div>
""")
            else:
                print(f"⚠️ Skipping {plot} due to invalid image format.")

        f.write("""
    </div>

    <h2>📥 Downloads</h2>
""")
        if os.path.exists(blast_csv):
            f.write(f"<a class='download' href='{blast_csv}' download>Download BLAST Summary CSV (ESM-2)</a>\n")
        if os.path.exists(identity_csv):
            f.write(f"<a class='download' href='{identity_csv}' download>Download Identity Scores CSV (ESM-2)</a>\n")
        if os.path.exists(decoded_fasta):
            f.write(f"<a class='download' href='{decoded_fasta}' download>Download Final FASTA (ESM-2)</a>\n")
        if os.path.exists(logreg_csv):
            f.write(f"<a class='download' href='{logreg_csv}' download>Download Logistic Regression Classification Results (ESM-2)</a>\n")

        f.write("""
</body>
</html>
""")

    print(f"✅ Final ESM-2 HTML report saved to {output_path}")

if __name__ == "__main__":
    generate_html_report()