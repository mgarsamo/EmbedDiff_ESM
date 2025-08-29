# train_embeddiff.py

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import scipy.ndimage
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from Bio import SeqIO
from torch.utils.data import DataLoader, TensorDataset
from models.latent_diffusion import MLPNoisePredictor, GaussianDiffusion
from matplotlib.font_manager import FontProperties
import pandas as pd

# === Paths ===
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMBED_PATH = os.path.join(ROOT, "embeddings", "esm2_embeddings.npy")
FASTA_PATH = os.path.join(ROOT, "data", "curated_thioredoxin_reductase.fasta")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
FIGURE_PATH = os.path.join(ROOT, "figures", "fig2b_loss_esm2")

# === Seed for reproducibility ===
torch.manual_seed(42)
np.random.seed(42)

# === Hyperparameters ===
embedding_dim = 1280   # ESM-2 hidden_size is 1280
timesteps = 1000       # Increased for better noise scheduling
batch_size = 32        # Reduced for stability
epochs = 300
lr = 1e-4             # Increased learning rate for diffusion

# === Load embeddings and class labels ===
x = np.load(EMBED_PATH)
print(f"[INFO] ESM-2 embedding shape: {x.shape}")

# Normalize ESM-2 latents for diffusion (scale to reasonable range)
mu = x.mean(axis=0, keepdims=True)
sigma = x.std(axis=0, keepdims=True) + 1e-6
np.savez(os.path.join(ROOT, "embeddings", "esm2_stats.npz"), mu=mu, sigma=sigma)

# Scale to [-1, 1] range for better diffusion training
x = (x - mu) / sigma
x = np.tanh(x * 0.5)  # Scale to [-1, 1] range
print(f"[INFO] Applied scaled normalization to ESM-2 embeddings (range: [{x.min():.3f}, {x.max():.3f}]).")

labels = []
for record in SeqIO.parse(FASTA_PATH, "fasta"):
    desc = record.description
    label = desc.split("[")[-1].split("]")[0].strip().lower()
    labels.append(label)

labels = pd.Series(labels, dtype="category")
one_hot = np.eye(len(labels.cat.categories))[labels.cat.codes]

# === Stratified 80/10/10 split ===
x_train, x_temp, y_train, y_temp = train_test_split(
    x, one_hot, test_size=0.2, stratify=labels, random_state=42
)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=42
)

def make_loader(x, y):
    data = np.hstack([x, y])
    return DataLoader(
        TensorDataset(torch.tensor(data, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True
    )

train_loader = make_loader(x_train, y_train)
val_loader = make_loader(x_val, y_val)
test_loader = make_loader(x_test, y_test)

# === Model ===
class ImprovedConditionalMLP(MLPNoisePredictor):
    def __init__(self, dim, cond_dim):
        super().__init__(dim + cond_dim)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + cond_dim + 1, 1024),
            torch.nn.LayerNorm(1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, dim)
        )

cond_dim = y_train.shape[1]
model = ImprovedConditionalMLP(dim=embedding_dim, cond_dim=cond_dim)
diffusion = GaussianDiffusion(model, timesteps=timesteps)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# === Training Loop ===
train_losses, val_losses = [], []
best_val_loss = float("inf")
best_model_path = os.path.join(CHECKPOINT_DIR, "best_embeddiff_mlp_esm2.pth")

for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0
    for batch in train_loader:
        x_batch = batch[0]
        t = torch.randint(0, timesteps, (x_batch.size(0),), device=x_batch.device)
        embed = x_batch[:, :embedding_dim]
        cond = x_batch[:, embedding_dim:]
        loss = diffusion.p_losses(embed, t, cond=cond)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    train_losses.append(epoch_train_loss / len(train_loader))

    # === Validation Loss ===
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x_batch = batch[0]
            t = torch.randint(0, timesteps, (x_batch.size(0),), device=x_batch.device)
            embed = x_batch[:, :embedding_dim]
            cond = x_batch[:, embedding_dim:]
            loss = diffusion.p_losses(embed, t, cond=cond)
            epoch_val_loss += loss.item()

    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"[ESM-2] Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(model.state_dict(), best_model_path)

# === Final Evaluation on Test Set ===
model.eval()
test_loss = 0
with torch.no_grad():
    for batch in test_loader:
        x_batch = batch[0]
        t = torch.randint(0, timesteps, (x_batch.size(0),), device=x_batch.device)
        embed = x_batch[:, :embedding_dim]
        cond = x_batch[:, embedding_dim:]
        loss = diffusion.p_losses(embed, t, cond=cond)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f"[ESM-2] 📊 Final Test Loss: {test_loss:.4f}")

# === Save loss curves ===
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
np.save(os.path.join(ROOT, "figures", "train_losses_esm2.npy"), np.array(train_losses))
np.save(os.path.join(ROOT, "figures", "val_losses_esm2.npy"), np.array(val_losses))

# === Plot loss curves ===
train_smooth = scipy.ndimage.gaussian_filter1d(train_losses, sigma=2)
val_smooth = scipy.ndimage.gaussian_filter1d(val_losses, sigma=2)

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

plt.figure(figsize=(8, 5))
plt.plot([], [], label="ESM-2", alpha = 0.0)
plt.plot(train_losses, label="Train (raw)", alpha=0.4)
plt.plot(val_losses, label="Val (raw)", alpha=0.4)
plt.plot(train_smooth, label="Train (smooth)", linewidth=2)
plt.plot(val_smooth, label="Val (smooth)", linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
font_props = FontProperties(weight='bold', family='Arial', size=12)
plt.legend(loc="best", frameon=True, prop=font_props)
plt.tight_layout()
plt.savefig(FIGURE_PATH + ".png", dpi=300)
plt.savefig(FIGURE_PATH + ".svg")
plt.close()

print(f"✅ Training complete. Best ESM-2 model saved to {best_model_path}. Loss curves plotted. Test loss reported.")
