"""Train the BPE tokenizer on prepared data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from OurLLM.tokenizer.bpe_tokenizer import OurBPETokenizer


def main():
    data_dir = Path(__file__).parent.parent / "datasets" / "raw"
    tokenizer_dir = Path(__file__).parent / "trained"

    train_files = sorted(data_dir.glob("*.txt"))
    if not train_files:
        print(f"No training files found in {data_dir}")
        print("Run prepare_data.py first:")
        print("  python OurLLM/datasets/prepare_data.py")
        return

    total_size = sum(f.stat().st_size for f in train_files)
    print(f"Training files: {[f.name for f in train_files]}")
    print(f"Total data size: {total_size / 1024 / 1024:.1f} MB")

    tokenizer = OurBPETokenizer(vocab_size=16000)
    tokenizer.train([str(f) for f in train_files], min_frequency=2)

    save_path = str(tokenizer_dir / "ourllm_tokenizer.json")
    tokenizer.save(save_path)

    print("\n=== Tokenizer Test ===")
    test_texts = [
        "Hello, how are you?",
        "The sun rises in the east.",
        "Yapay zeka dünyayı değiştiriyor.",
        "Python is great for AI.",
    ]

    for text in test_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        print(f"\nOriginal:  {text}")
        print(f"Token IDs: {ids}")
        print(f"Decoded:   {decoded}")
        print(f"Num tokens: {len(ids)}")

    print(f"\nVocab size: {tokenizer.get_vocab_size()}")
    print(f"Special tokens: <pad>={tokenizer.token_to_id('<pad>')}, "
          f"<unk>={tokenizer.token_to_id('<unk>')}, "
          f"<bos>={tokenizer.token_to_id('<bos>')}, "
          f"<eos>={tokenizer.token_to_id('<eos>')}")


if __name__ == "__main__":
    main()
