import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMClassifier(nn.Module):
    def __init__(
        self, num_classes, seq_len, num_channels, num_subjects, hidden_dim=128
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.pool = nn.MaxPool1d(2)

        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True
        )

        self.subject_embedding = nn.Embedding(num_subjects, hidden_dim)

        self.fc = nn.Linear(hidden_dim * 2 + hidden_dim, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x, subject_idx):
        # x shape: (batch_size, num_channels, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.permute(0, 2, 1)  # (batch_size, seq_len, hidden_dim)

        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 最後の時間ステップの出力を使用

        subject_embed = self.subject_embedding(subject_idx)
        x = torch.cat([x, subject_embed], dim=1)

        x = self.dropout(x)
        x = self.fc(x)

        return x


# # モデルの初期化
# model = CNNLSTMClassifier(
#     num_classes=test_set.num_classes,
#     seq_len=test_set.seq_len,
#     num_channels=test_set.num_channels,
#     num_subjects=test_set.num_subjects,
# )
