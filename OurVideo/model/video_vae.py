import torch
import torch.nn as nn


class Encoder3D(nn.Module):
    def __init__(self, z_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256 * 1 * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(256 * 1 * 4 * 4, z_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder3D(nn.Module):
    def __init__(self, z_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256 * 1 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 256, 1, 4, 4)
        return self.deconv(h)


class VideoVAE(nn.Module):
    def __init__(self, z_dim: int = 128):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder3D(z_dim)
        self.decoder = Decoder3D(z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def generate(self, num_videos: int = 1, device: str = "cpu"):
        z = torch.randn(num_videos, self.z_dim, device=device)
        with torch.no_grad():
            return self.decoder(z)


def vae_loss(recon, target, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon, target, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss
