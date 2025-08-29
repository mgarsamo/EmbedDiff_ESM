# scripts/blastlocal_esm2.py

import os
import csv
import subprocess
from Bio import SeqIO
from Bio.Blast import NCBIXML

# === Config ===
fasta_file = "data/decoded_embeddiff_esm2.fasta"
output_dir = "data/blast_results"
csv_output = os.path.join(output_dir, "blast_summary_local_esm2.csv")
blast_db = "/Users/melaku/blastdb/swissprot"  # ✅ Your Swiss-Prot BLAST DB path
os.makedirs(output_dir, exist_ok=True)

# === Parameters ===
MAX_QUERIES = 240
IDENTITY_THRESHOLD = 70.0

def fmt(val):
    return f"{val:.2f}" if isinstance(val, (float, int)) else ""

# === Read sequences ===
sequences = list(SeqIO.parse(fasta_file, "fasta"))
if MAX_QUERIES:
    sequences = sequences[:MAX_QUERIES]

# === Run BLAST and collect results ===
blast_data = []

for i, seq_record in enumerate(sequences):
    seq_str = str(seq_record.seq).strip()
    if not seq_str:
        print(f"⚠️ Skipping empty sequence {i+1}/{len(sequences)}: {seq_record.id}")
        continue

    print(f"\n🔬 [ESM-2] BLASTing sequence {i+1}/{len(sequences)}: {seq_record.id}")

    # Save temporary FASTA
    temp_fasta = os.path.join(output_dir, f"{seq_record.id}.fasta")
    SeqIO.write(seq_record, temp_fasta, "fasta")

    # Output XML path
    xml_path = os.path.join(output_dir, f"{seq_record.id}_blast.xml")

    # Run BLAST
    blast_cmd = [
        "blastp",
        "-query", temp_fasta,
        "-db", blast_db,
        "-out", xml_path,
        "-outfmt", "5",
        "-evalue", "0.001",
        "-max_target_seqs", "5"
    ]

    try:
        subprocess.run(blast_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ BLAST failed for {seq_record.id}: {e}")
        continue

    # Parse result
    try:
        with open(xml_path) as result_handle:
            blast_record = NCBIXML.read(result_handle)
    except Exception as e:
        print(f"❌ Failed to parse BLAST XML for {seq_record.id}: {e}")
        continue

    top_hits = []
    for alignment in blast_record.alignments[:2]:
        hsp = alignment.hsps[0]
        identity = (hsp.identities / hsp.align_length) * 100
        coverage = (hsp.align_length / blast_record.query_length) * 100
        top_hits.append((alignment.hit_def, identity, coverage, hsp.expect))

    is_novel = "Yes"
    if top_hits and isinstance(top_hits[0][1], (float, int)) and top_hits[0][1] >= IDENTITY_THRESHOLD:
        is_novel = "No"

    while len(top_hits) < 2:
        top_hits.append(("None", "", "", ""))

    score = 0.0
    if isinstance(top_hits[0][1], (float, int)) and isinstance(top_hits[0][2], (float, int)):
        score = top_hits[0][1] + top_hits[0][2]

    blast_data.append([
        score,
        seq_record.id, is_novel,
        top_hits[0][0], fmt(top_hits[0][1]), fmt(top_hits[0][2]), top_hits[0][3],
        top_hits[1][0], fmt(top_hits[1][1]), top_hits[1][3],
        seq_str
    ])

# === Sort results by score (identity + coverage) descending ===
blast_data.sort(key=lambda x: x[0], reverse=True)

# === Write results to CSV ===
with open(csv_output, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "Score", "Sequence_ID", "Is_Novel",
        "Top_Hit_1_Definition", "Top_Hit_1_Identity(%)", "Top_Hit_1_Query_Coverage(%)", "Top_Hit_1_E-value",
        "Top_Hit_2_Definition", "Top_Hit_2_Identity(%)", "Top_Hit_2_E-value",
        "ESM2_Sequence"
    ])
    for row in blast_data:
        writer.writerow(row)

print(f"\n✅ Done! [ESM-2] Sorted BLAST results saved to:\n{csv_output}")
