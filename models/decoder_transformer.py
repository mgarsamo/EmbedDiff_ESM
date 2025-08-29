import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]  # batch_first shape


class TransformerDecoderModel(nn.Module):
    def __init__(
        self,
        input_dim=1280,     # ESM2 token embedding size
        embed_dim=512,      # internal dimension
        seq_len=350,
        vocab_size=23,
        nhead=4,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size

        # === Layers ===
        self.input_proj = nn.Linear(input_dim, embed_dim)
        nn.init.xavier_uniform_(self.input_proj.weight, gain=0.5)  # scaled init

        self.dropout = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len)
        self.norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",           # 👍 better than relu for protein decoding
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_embeddings):
        """
        token_embeddings: (batch_size, seq_len, input_dim)
        returns logits: (batch_size, seq_len, vocab_size)
        """
        x = self.input_proj(token_embeddings)     # (B, L, embed_dim)
        x = self.dropout(x)
        x = self.pos_encoder(x)                   # (B, L, embed_dim)
        x = self.transformer(x)                   # (B, L, embed_dim)
        x = self.norm(x)
        logits = self.output_fc(x)                # (B, L, vocab_size)
        return logits
