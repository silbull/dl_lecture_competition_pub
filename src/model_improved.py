import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifierWithSubject(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        num_subjects: int,
        hid_dim: int = 128,
        subject_embedding_dim: int = 16,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        self.spatial_attention = SpatialAttention(in_channels, hid_dim)
        self.subject_layers = nn.ModuleList(
            [nn.Conv1d(hid_dim, hid_dim, kernel_size=1) for _ in range(num_subjects)]
        )

        self.blocks = nn.Sequential(
            ConvBlock(hid_dim, hid_dim),
            ConvBlock(hid_dim, hid_dim * 2),
            ConvBlock(hid_dim * 2, hid_dim * 2),
        )

        self.subject_embedding = nn.Embedding(num_subjects, subject_embedding_dim)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim * 2 + subject_embedding_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor, subject_idxs: torch.Tensor) -> torch.Tensor:
        # print(f"Input shape: {X.shape}")
        X = self.spatial_attention(X)
        # print(f"After spatial attention: {X.shape}")

        # Apply subject-specific layer
        X = torch.stack(
            [
                self.subject_layers[s.item()](X[i : i + 1].squeeze(0))
                for i, s in enumerate(subject_idxs)
            ]
        )
        # print(f"After subject layers: {X.shape}")

        X = self.blocks(X)
        # print(f"After blocks: {X.shape}")
        X = self.head[0](X)
        X = self.head[1](X)
        # print(f"After head[1]: {X.shape}")
        subject_emb = self.subject_embedding(subject_idxs)
        X = torch.cat([X, subject_emb], dim=1)
        # print(f"After concatenation: {X.shape}")
        return self.head[2:](X)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.attention = nn.Parameter(torch.randn(in_channels, in_channels))

    def forward(self, x):
        # x shape: (batch_size, in_channels, seq_len)
        attn = F.softmax(self.attention, dim=1)
        x = torch.einsum("bct,ic->bit", x, attn)
        return self.conv(x)  # 出力は (batch_size, out_channels, seq_len) であるべき


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm2 = nn.BatchNorm1d(num_features=out_dim)
        self.dropout = nn.Dropout(p_drop)
        self.skip = (
            nn.Conv1d(in_dim, out_dim, kernel_size=1)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # print(f"ConvBlock input shape: {X.shape}")
        residual = self.skip(X)
        X = F.gelu(self.batchnorm1(self.conv1(X)))
        X = self.batchnorm2(self.conv2(X))
        X = F.gelu(X + residual)
        return self.dropout(X)
