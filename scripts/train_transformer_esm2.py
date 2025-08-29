# scripts/train_transformer_esm2.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from models.decoder_transformer import TransformerDecoderModel
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

# === Load dataset ===
data = torch.load("data/decoder_dataset_esm2.pt")
token_embeddings = data["token_embeddings"]    # (B, L, 1280)
sequences = data["sequences"]                  # (B, L)
esm2_logits = data["esm2_logits"]             # (B, L, V)

# === Dataset split (80/10/10) ===
dataset = TensorDataset(token_embeddings, sequences, esm2_logits)
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_set, val_set, test_set = random_split(
    dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8)

# === Device setup (MPS or CPU) ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === Initialize model ===
seq_len = sequences.shape[1]
model = TransformerDecoderModel(
    input_dim=1280,
    embed_dim=512,
    seq_len=seq_len,
    vocab_size=23,   # 20 AAs + PAD/EOS/UNK style tokens
    nhead=4,
    num_layers=4,
    dropout=0.1
).to(device)

# === Optimizer and Losses ===
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
ce_loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

# === Training Parameters ===
EPOCHS = 40
ALPHA = 0.01  # 🔧 L2-like penalty on embedding norm
PATIENCE = 3
best_val_loss = float("inf")
patience_counter = 0
train_losses, val_losses = [], []
output_dim = model.vocab_size

# === Training Loop ===
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for token_embed, y_seq, _ in train_loader:   # ignore logits during training
        token_embed, y_seq = token_embed.to(device), y_seq.to(device)

        optimizer.zero_grad()
        logits = model(token_embed)  # (B, L, V)
        ce_loss = ce_loss_fn(logits.view(-1, output_dim), y_seq.view(-1))
        norm_loss = torch.mean((torch.norm(token_embed, dim=-1) - 1) ** 2)
        total_loss = ce_loss + ALPHA * norm_loss
        total_loss.backward()
        optimizer.step()
        train_loss += total_loss.item()

    train_losses.append(train_loss / len(train_loader))

    # === Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for token_embed, y_seq, _ in val_loader:
            token_embed, y_seq = token_embed.to(device), y_seq.to(device)
            logits = model(token_embed)
            ce = ce_loss_fn(logits.view(-1, output_dim), y_seq.view(-1))
            norm = torch.mean((torch.norm(token_embed, dim=-1) - 1) ** 2)
            val_loss += (ce + ALPHA * norm).item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"[ESM-2][Epoch {epoch+1:>2}] Train Loss: {train_losses[-1]:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/decoder_transformer_best_esm2.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
            break

# === Save final model
torch.save(model.state_dict(), "checkpoints/decoder_transformer_last_esm2.pth")
print("✅ Final ESM-2 decoder model saved!")

# === Plot loss curves
font_props = FontProperties(weight='bold', family='Arial', size=12)
plt.figure(figsize=(8, 5))
# Add ESM-2 label with alpha=0.0 to avoid showing dash
plt.plot([], [], label="ESM-2", alpha=0.0)
plt.plot(train_losses, label="Train", linewidth=2)
plt.plot(val_losses, label="Val", linewidth=2)
plt.xlabel("Epoch", fontproperties=font_props)
plt.ylabel("Total Loss", fontproperties=font_props)
plt.legend(prop=font_props, frameon=True)
plt.xticks(fontproperties=font_props)
plt.yticks(fontproperties=font_props)
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/fig5a_decoder_loss_esm2.png", dpi=300)
plt.savefig("figures/fig5a_decoder_loss_esm2.svg")
plt.close()
