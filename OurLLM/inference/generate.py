"""
Text generation with trained OurLLM model.
"""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from OurLLM.model.transformer import OurLLMModel
from OurLLM.tokenizer.bpe_tokenizer import OurBPETokenizer, PAD_ID, BOS_ID, EOS_ID


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    tokenizer_path = Path(__file__).parent.parent / "tokenizer" / "trained" / "ourllm_tokenizer.json"
    tokenizer = OurBPETokenizer()
    tokenizer.load(str(tokenizer_path))

    model = OurLLMModel(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=config["model"]["embedding_dim"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        ff_dim=config["model"]["feedforward_dim"],
        max_seq_length=config["model"]["max_seq_length"],
        dropout=0.0,
        pad_id=PAD_ID,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded from {checkpoint_path} (step {checkpoint['step']})")
    print(f"Parameters: {model.count_parameters():,}")

    return model, tokenizer


def generate_text(
    model: OurLLMModel,
    tokenizer: OurBPETokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 50,
    device: torch.device = torch.device("cpu"),
) -> str:
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    output_ids = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=EOS_ID,
    )

    generated_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    return generated_text


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    project_root = Path(__file__).parent.parent.parent
    checkpoint_path = project_root / "checkpoints" / "ourllm_final.pt"

    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}")
        print("Train the model first with: python OurLLM/training/train.py")
        return

    model, tokenizer = load_model(str(checkpoint_path), device)

    prompts = [
        "The sun",
        "Artificial intelligence",
        "Python is",
        "The Earth",
        "Yapay zeka",
    ]

    print("\n=== Text Generation ===")
    for prompt in prompts:
        generated = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=30,
            temperature=0.8,
            top_k=50,
            device=device,
        )
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated}")

    print("\n=== Interactive Mode ===")
    print("Type a prompt and press Enter (type 'quit' to exit)")
    while True:
        try:
            prompt = input("\nPrompt> ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                break
            if not prompt:
                continue

            generated = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=50,
                temperature=0.8,
                top_k=50,
                device=device,
            )
            print(f"Generated: {generated}")
        except (KeyboardInterrupt, EOFError):
            break

    print("\nDone!")


if __name__ == "__main__":
    main()
