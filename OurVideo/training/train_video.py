import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from OurVideo.model.video_vae import VideoVAE, vae_loss


def create_synthetic_dataset(num_samples: int = 500, frames: int = 16, size: int = 64):
    data = torch.rand(num_samples, 3, frames, size, size)
    return TensorDataset(data)


def train(epochs: int = 30, batch_size: int = 8, lr: float = 1e-3, z_dim: int = 128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = create_synthetic_dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Dataset: {len(dataset)} synthetic video clips")

    model = VideoVAE(z_dim=z_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count:,}")

    checkpoint_dir = Path(__file__).parent.parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for (batch,) in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "z_dim": z_dim,
            }, checkpoint_dir / f"ourvideo_epoch_{epoch+1}.pt")

    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epochs,
        "z_dim": z_dim,
    }, checkpoint_dir / "ourvideo_vae.pt")
    print(f"Saved: checkpoints/ourvideo_vae.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
