import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, p_drop=0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_dim, out_dim, kernel_size, stride, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        return self.dropout(self.activation(self.bn(self.conv(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class ImprovedBrainwaveClassifier(nn.Module):
    def __init__(
        self, num_classes, seq_len, in_channels, num_subjects, hid_dim=128, num_layers=4
    ):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.subject_embedding = nn.Embedding(num_subjects, hid_dim)

        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hid_dim,
                nhead=8,
                dim_feedforward=hid_dim * 4,
                dropout=0.1,
                activation="gelu",
            ),
            num_layers=num_layers,
        )

        self.classifier = nn.Linear(hid_dim * 2, num_classes)

    def forward(self, x, subject_idx):
        # x shape: (batch_size, channels, seq_len)
        x = self.feature_extractor(x)
        x = x.permute(2, 0, 1)  # (seq_len, batch_size, hid_dim)
        x = self.transformer_layers(x)
        x = x.permute(1, 2, 0)  # (batch_size, hid_dim, seq_len)

        # Global average pooling
        x = x.mean(dim=2)  # (batch_size, hid_dim)

        subject_embed = self.subject_embedding(subject_idx)  # (batch_size, hid_dim)
        x = torch.cat([x, subject_embed], dim=1)  # (batch_size, hid_dim * 2)

        return self.classifier(x)


# class BrainwaveClassifier(nn.Module):
#     def __init__(
#         self, num_classes, seq_len, num_channels, hidden_size=768, num_hidden_layers=12
#     ):
#         super().__init__()
#         self.num_channels = num_channels
#         self.config = Wav2Vec2Config(
#             hidden_size=hidden_size,
#             num_hidden_layers=num_hidden_layers,
#             num_attention_heads=12,
#             intermediate_size=3072,
#             hidden_act="gelu",
#             hidden_dropout=0.1,
#             activation_dropout=0.1,
#             attention_dropout=0.1,
#             feat_proj_dropout=0.0,
#             layerdrop=0.1,
#             initializer_range=0.02,
#             layer_norm_eps=1e-5,
#             feat_extract_norm="group",
#             feat_extract_activation="gelu",
#             conv_dim=(512, 512, 512, 512, 512, 512, 512),
#             conv_stride=(5, 2, 2, 2, 2, 2, 2),
#             conv_kernel=(10, 3, 3, 3, 3, 2, 2),
#             conv_bias=False,
#             num_conv_pos_embeddings=128,
#             num_conv_pos_embedding_groups=16,
#             gradient_checkpointing=False,  # 勾配チェックポイントを無効化
#         )
#         self.channel_proj = nn.Linear(num_channels, 1)
#         self.wav2vec2 = Wav2Vec2Model(self.config)
#         self.classifier = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         # x shape: (batch_size, channels, seq_len)
#         # Project channels to a single dimension
#         x = x.transpose(1, 2)  # (batch_size, seq_len, channels)
#         x = self.channel_proj(x).squeeze(-1)  # (batch_size, seq_len)

#         outputs = self.wav2vec2(x)
#         pooled_output = outputs.last_hidden_state.mean(dim=1)
#         return self.classifier(pooled_output)


# # モデルの初期化
# num_classes = 1854  # クラス数
# seq_len = 281
# num_channels = 271
# model = BrainwaveClassifier(num_classes, seq_len, num_channels)

# # 使用例
# model = BrainwaveClassifier(
#     train_set.num_classes, train_set.seq_len, train_set.num_channels
# )

# # テスト用の入力
# batch_size = 32
# dummy_input = torch.randn(batch_size, train_set.num_channels, train_set.seq_len)
# output = model(dummy_input)
# print(output.shape)  # Expected:
