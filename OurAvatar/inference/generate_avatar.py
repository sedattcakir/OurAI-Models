import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from OurAvatar.model.avatar_net import AvatarNet


def generate(face_image_path: str = None, num_frames: int = 20, output_dir: str = "generated_avatars"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(__file__).parent.parent.parent / "checkpoints" / "ouravatar_net.pt"

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return None

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    model = AvatarNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if face_image_path and Path(face_image_path).exists():
        img = Image.open(face_image_path).convert("RGB").resize((64, 64))
        face_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        face_tensor = face_tensor.unsqueeze(0).to(device)
    else:
        face_tensor = torch.rand(1, 3, 64, 64, device=device)

    audio_features = torch.rand(1, num_frames, 80, device=device)

    with torch.no_grad():
        frames = model(face_tensor, audio_features)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pil_frames = []
    for t in range(frames.size(1)):
        frame = frames[0, t].clamp(0, 1).mul(255).byte()
        frame = frame.permute(1, 2, 0).cpu().numpy()
        pil_frames.append(Image.fromarray(frame))

    gif_path = out_path / "avatar_animation.gif"
    pil_frames[0].save(
        str(gif_path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=100,
        loop=0,
    )
    print(f"Saved: {gif_path}")
    return [str(gif_path)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--face", type=str, default=None)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--output", type=str, default="generated_avatars")
    args = parser.parse_args()
    generate(face_image_path=args.face, num_frames=args.frames, output_dir=args.output)
