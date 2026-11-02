import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from OurImage.model.dcgan import Generator, Discriminator


def train(epochs: int = 50, batch_size: int = 64, lr: float = 0.0002, latent_dim: int = 100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    data_root = Path(__file__).parent.parent / "datasets" / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    dataset = datasets.CIFAR10(root=str(data_root), download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    print(f"Dataset: {len(dataset)} images")

    gen = Generator(latent_dim=latent_dim).to(device)
    disc = Discriminator().to(device)

    gen.apply(_weights_init)
    disc.apply(_weights_init)

    criterion = nn.BCELoss()
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

    checkpoint_dir = Path(__file__).parent.parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch = real_images.size(0)
            real_label = torch.ones(batch, 1, device=device)
            fake_label = torch.zeros(batch, 1, device=device)

            opt_d.zero_grad()
            output_real = disc(real_images)
            loss_real = criterion(output_real, real_label)

            z = torch.randn(batch, latent_dim, device=device)
            fake_images = gen(z)
            output_fake = disc(fake_images.detach())
            loss_fake = criterion(output_fake, fake_label)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            opt_d.step()

            opt_g.zero_grad()
            output = disc(fake_images)
            loss_g = criterion(output, real_label)
            loss_g.backward()
            opt_g.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{i}] D_loss: {loss_d.item():.4f} G_loss: {loss_g.item():.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save({
                "generator_state_dict": gen.state_dict(),
                "discriminator_state_dict": disc.state_dict(),
                "epoch": epoch + 1,
                "latent_dim": latent_dim,
            }, checkpoint_dir / f"ourimage_epoch_{epoch+1}.pt")

    torch.save({
        "generator_state_dict": gen.state_dict(),
        "epoch": epochs,
        "latent_dim": latent_dim,
    }, checkpoint_dir / "ourimage_generator.pt")
    print(f"Saved: checkpoints/ourimage_generator.pt")


def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
