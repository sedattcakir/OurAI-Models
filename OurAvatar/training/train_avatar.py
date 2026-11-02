import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from OurAvatar.model.avatar_net import AvatarNet


def create_synthetic_dataset(num_samples: int = 300, seq_len: int = 20):
    faces = torch.rand(num_samples, 3, 64, 64)
    audio = torch.rand(num_samples, seq_len, 80)
    target_frames = torch.rand(num_samples, seq_len, 3, 64, 64)
    return TensorDataset(faces, audio, target_frames)


def train(epochs: int = 30, batch_size: int = 4, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = create_synthetic_dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Dataset: {len(dataset)} synthetic samples")

    model = AvatarNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count:,}")

    checkpoint_dir = Path(__file__).parent.parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for face, audio, target in dataloader:
            face, audio, target = face.to(device), audio.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(face, audio)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * face.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
            }, checkpoint_dir / f"ouravatar_epoch_{epoch+1}.pt")

    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epochs,
    }, checkpoint_dir / "ouravatar_net.pt")
    print(f"Saved: checkpoints/ouravatar_net.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
