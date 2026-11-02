import torch
import torch.nn as nn


class FaceEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.fc = nn.Linear(256 * 4 * 4, embed_dim)

    def forward(self, face):
        h = self.conv(face)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class AudioEncoder(nn.Module):
    def __init__(self, mel_dim: int = 80, embed_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(mel_dim, 128, 3, 1, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, 1, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.fc = nn.Linear(256, embed_dim)

    def forward(self, mel):
        x = mel.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return self.fc(x)


class MotionGenerator(nn.Module):
    def __init__(self, embed_dim: int = 256, motion_dim: int = 512):
        super().__init__()
        self.rnn = nn.GRU(embed_dim * 2, motion_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(motion_dim, motion_dim)

    def forward(self, face_embed, audio_embed):
        seq_len = audio_embed.size(1)
        face_expanded = face_embed.unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([face_expanded, audio_embed], dim=2)
        motion, _ = self.rnn(combined)
        return self.fc(motion)


class FrameDecoder(nn.Module):
    def __init__(self, motion_dim: int = 512):
        super().__init__()
        self.fc = nn.Linear(motion_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, motion):
        batch, seq_len, dim = motion.shape
        motion_flat = motion.reshape(batch * seq_len, dim)
        h = self.fc(motion_flat)
        h = h.view(-1, 256, 4, 4)
        frames = self.deconv(h)
        return frames.view(batch, seq_len, 3, 64, 64)


class AvatarNet(nn.Module):
    def __init__(self, mel_dim: int = 80, embed_dim: int = 256, motion_dim: int = 512):
        super().__init__()
        self.face_encoder = FaceEncoder(embed_dim)
        self.audio_encoder = AudioEncoder(mel_dim, embed_dim)
        self.motion_gen = MotionGenerator(embed_dim, motion_dim)
        self.frame_decoder = FrameDecoder(motion_dim)

    def forward(self, face_image, audio_features):
        face_embed = self.face_encoder(face_image)
        audio_embed = self.audio_encoder(audio_features)
        motion = self.motion_gen(face_embed, audio_embed)
        frames = self.frame_decoder(motion)
        return frames
