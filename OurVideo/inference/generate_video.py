import argparse
import torch
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from OurVideo.model.video_vae import VideoVAE


def tensor_to_pil_frames(video_tensor):
    frames = []
    for t in range(video_tensor.size(1)):
        frame = video_tensor[0, :, t]
        frame = frame.clamp(0, 1).mul(255).byte()
        frame = frame.permute(1, 2, 0).cpu().numpy()
        frames.append(Image.fromarray(frame))
    return frames


def generate(num_videos: int = 1, output_dir: str = "generated_videos"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(__file__).parent.parent.parent / "checkpoints" / "ourvideo_vae.pt"

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return None

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    z_dim = checkpoint.get("z_dim", 128)

    model = VideoVAE(z_dim=z_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    paths = []
    for i in range(num_videos):
        video = model.generate(num_videos=1, device=str(device))
        frames = tensor_to_pil_frames(video)
        gif_path = out_path / f"video_{i}.gif"
        frames[0].save(
            str(gif_path),
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )
        paths.append(str(gif_path))
        print(f"Saved: {gif_path}")

    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=2)
    parser.add_argument("--output", type=str, default="generated_videos")
    args = parser.parse_args()
    generate(num_videos=args.num, output_dir=args.output)
