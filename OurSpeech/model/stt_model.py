import torch
import torch.nn as nn


class OurSTTModel(nn.Module):
    def __init__(self, mel_dim: int = 80, hidden_dim: int = 256, vocab_size: int = 29):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(mel_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(
            hidden_dim, hidden_dim // 2,
            num_layers=2, batch_first=True, bidirectional=True, dropout=0.1,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, mel):
        x = self.conv(mel)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        logits = self.fc(x)
        return logits


CHAR_MAP = {c: i for i, c in enumerate("_' abcdefghijklmnopqrstuvwxyz")}
INV_CHAR_MAP = {i: c for c, i in CHAR_MAP.items()}


def decode_ctc(logits):
    pred = logits.argmax(dim=-1)
    decoded = []
    for seq in pred:
        chars = []
        prev = -1
        for idx in seq:
            idx = idx.item()
            if idx != prev and idx != 0:
                chars.append(INV_CHAR_MAP.get(idx, ""))
            prev = idx
        decoded.append("".join(chars))
    return decoded
