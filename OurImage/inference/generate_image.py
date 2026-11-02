import argparse
import torch
from pathlib import Path
from torchvision.utils import save_image
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from OurImage.model.dcgan import Generator


def generate(num_images: int = 1, output_dir: str = "generated"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(__file__).parent.parent.parent / "checkpoints" / "ourimage_generator.pt"

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return None

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    latent_dim = checkpoint.get("latent_dim", 100)

    gen = Generator(latent_dim=latent_dim).to(device)
    gen.load_state_dict(checkpoint["generator_state_dict"])
    gen.eval()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        z = torch.randn(num_images, latent_dim, device=device)
        images = gen(z)
        images = (images + 1) / 2

    paths = []
    for i in range(num_images):
        p = out_path / f"generated_{i}.png"
        save_image(images[i], str(p))
        paths.append(str(p))
        print(f"Saved: {p}")

    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=4)
    parser.add_argument("--output", type=str, default="generated")
    args = parser.parse_args()
    generate(num_images=args.num, output_dir=args.output)
