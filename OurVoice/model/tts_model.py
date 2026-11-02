import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 128, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, _ = self.rnn(embedded)
        return outputs


class Attention(nn.Module):
    def __init__(self, enc_dim: int = 256, dec_dim: int = 256):
        super().__init__()
        self.W = nn.Linear(enc_dim + dec_dim, dec_dim)
        self.v = nn.Linear(dec_dim, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        dec_expanded = decoder_state.unsqueeze(1).expand(-1, seq_len, -1)
        energy = torch.tanh(self.W(torch.cat([dec_expanded, encoder_outputs], dim=2)))
        weights = torch.softmax(self.v(energy).squeeze(2), dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class MelDecoder(nn.Module):
    def __init__(self, mel_dim: int = 80, hidden_dim: int = 256, max_steps: int = 200):
        super().__init__()
        self.mel_dim = mel_dim
        self.max_steps = max_steps
        self.prenet = nn.Sequential(
            nn.Linear(mel_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attention = Attention(hidden_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim + hidden_dim, hidden_dim)
        self.mel_proj = nn.Linear(hidden_dim + hidden_dim, mel_dim)
        self.stop_proj = nn.Linear(hidden_dim + hidden_dim, 1)

    def forward(self, encoder_outputs, max_steps=None):
        batch = encoder_outputs.size(0)
        device = encoder_outputs.device
        max_steps = max_steps or self.max_steps

        mel_input = torch.zeros(batch, self.mel_dim, device=device)
        hidden = torch.zeros(batch, 256, device=device)

        mel_outputs = []
        for _ in range(max_steps):
            prenet_out = self.prenet(mel_input)
            context, _ = self.attention(hidden, encoder_outputs)
            rnn_input = torch.cat([prenet_out, context], dim=1)
            hidden = self.rnn(rnn_input, hidden)
            output = torch.cat([hidden, context], dim=1)
            mel_frame = self.mel_proj(output)
            mel_outputs.append(mel_frame)
            mel_input = mel_frame

        return torch.stack(mel_outputs, dim=1)


class OurTTSModel(nn.Module):
    def __init__(self, vocab_size: int = 128, mel_dim: int = 80):
        super().__init__()
        self.encoder = TextEncoder(vocab_size=vocab_size)
        self.decoder = MelDecoder(mel_dim=mel_dim)

    def forward(self, text_ids, max_steps=200):
        encoder_out = self.encoder(text_ids)
        mel_out = self.decoder(encoder_out, max_steps=max_steps)
        return mel_out
