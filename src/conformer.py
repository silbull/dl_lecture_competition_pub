import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ConformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        ff_expansion=4,
        conv_expansion=2,
        conv_kernel_size=31,
        dropout=0.1,
    ):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_expansion, dim),
            nn.Dropout(dropout),
        )
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.conv = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose(-1, -2),
            nn.Conv1d(dim, dim * conv_expansion, 1),
            nn.GLU(dim=1),
            nn.Conv1d(
                dim, dim, conv_kernel_size, padding=conv_kernel_size // 2, groups=dim
            ),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1),
            Transpose(-1, -2),
            nn.Dropout(dropout),
        )
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_expansion, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x, x, x)[0]
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class BrainWaveConformer(nn.Module):
    def __init__(
        self,
        num_classes,
        seq_len,
        num_channels,
        dim=128,
        depth=4,
        num_heads=4,
        num_subjects=4,
    ):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(num_channels, dim, kernel_size=3, padding=1),
            Transpose(-1, -2),
            nn.LayerNorm(dim),
        )
        self.subject_layers = nn.ModuleList(
            [nn.Conv1d(dim, dim, kernel_size=1) for _ in range(num_subjects)]
        )
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, dim))
        self.blocks = nn.ModuleList(
            [ConformerBlock(dim, num_heads) for _ in range(depth)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x, subject_idxs):
        x = self.embed(x)  # (batch_size, seq_len, dim)

        # Apply subject-specific layer
        x = x.transpose(-1, -2)  # (batch_size, dim, seq_len)
        x = torch.stack(
            [
                self.subject_layers[s](x[i : i + 1]).squeeze(0)
                for i, s in enumerate(subject_idxs)
            ]
        )
        x = x.transpose(-1, -2)  # (batch_size, seq_len, dim)

        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = x.transpose(-1, -2)
        x = self.pool(x).squeeze(-1)
        x = self.to_latent(x)
        return self.mlp_head(x)
