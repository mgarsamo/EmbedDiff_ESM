#scripts/logistic_regression_probe_esm2.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns
import os


# === Paths ===
EMBED_PATH = "embeddings/esm2_embeddings.npy"   # changed from dayhoff
FASTA_PATH = "data/curated_thioredoxin_reductase.fasta"
FIGURE_DIR = "figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# === Load embeddings ===
embeddings = np.load(EMBED_PATH)
print(f"[INFO] Embedding shape: {embeddings.shape}")

# === Load labels ===
labels = []
for record in SeqIO.parse(FASTA_PATH, "fasta"):
    desc = record.description
    label = desc.split("[")[-1].split("]")[0].strip().lower()
    labels.append(label)
labels = pd.Series(labels, dtype="category")
label_ids = labels.cat.codes.values
print(f"[INFO] Number of classes: {len(labels.cat.categories)} | Labels: {list(labels.cat.categories)}")

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, label_ids, test_size=0.2, stratify=label_ids, random_state=42
)

# === Train logistic regression probe ===
clf = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"[RESULT] Logistic Regression accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=labels.cat.categories))

# === Create detailed results table ===
report = classification_report(y_test, y_pred, target_names=labels.cat.categories, output_dict=True)
results_df = pd.DataFrame(report).transpose()
results_df = results_df.round(4)
results_df = results_df.drop('accuracy', errors='ignore')  # Remove accuracy row for cleaner table

# Save results table to CSV
table_path = os.path.join(FIGURE_DIR, "logreg_classification_results_esm2.csv")
results_df.to_csv(table_path)
print(f"[TABLE] Classification results saved to {table_path}")

# === Confusion Matrix (Counts) ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))

# Create subplot for counts
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels.cat.categories,
            yticklabels=labels.cat.categories)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Counts)")

# === Confusion Matrix (Percentages) ===
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
plt.subplot(1, 2, 2)
sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=labels.cat.categories,
            yticklabels=labels.cat.categories)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Percentages)")

plt.tight_layout()
cm_path = os.path.join(FIGURE_DIR, "logreg_confusion_matrix_esm2.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"[FIGURE] Confusion matrices saved to {cm_path}")
plt.close()

# === Accuracy Bar Plot (Publication Quality) ===
accs = [report[c]['recall'] for c in labels.cat.categories]

# Set up publication-quality figure
plt.figure(figsize=(6, 4))
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# Create bar plot with same colors as before
bars = plt.bar(range(len(labels.cat.categories)), accs, 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
               edgecolor='black', linewidth=0.8, alpha=0.8)

# Customize axes
plt.ylim(0, 1.05)
plt.ylabel("Recall", fontsize=14, fontweight='bold')
plt.xlabel("Class", fontsize=14, fontweight='bold')
plt.title("Logistic Regression Per-Class Recall (ESM-2)", fontsize=16, fontweight='bold')
plt.xticks(range(len(labels.cat.categories)), labels.cat.categories, 
           fontsize=12, rotation=0)

# Add value labels on bars with proper formatting
for i, (bar, acc) in enumerate(zip(bars, accs)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{acc:.3f}', ha='center', va='bottom', 
             fontsize=11, fontweight='bold')

# Customize grid and spines
plt.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Adjust layout and save
plt.tight_layout()
bar_path = os.path.join(FIGURE_DIR, "logreg_per_class_recall_esm2.png")
plt.savefig(bar_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[FIGURE] Publication-quality bar plot saved to {bar_path}")
plt.close()

# === Print summary statistics ===
print("\n" + "="*60)
print("CLASSIFICATION RESULTS SUMMARY")
print("="*60)
print(f"Overall Accuracy: {acc:.4f}")
print(f"Macro Average F1-Score: {results_df.loc['macro avg', 'f1-score']:.4f}")
print(f"Weighted Average F1-Score: {results_df.loc['weighted avg', 'f1-score']:.4f}")
print("="*60)
