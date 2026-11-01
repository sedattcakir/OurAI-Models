"""
Prepare training data for OurLLM tokenizer and model training.
Downloads small public domain text samples for initial prototyping.
"""
import os
import json
import sys
from pathlib import Path
from datasets import load_dataset


def prepare_wiki_data(output_dir: str, num_samples: int = 5000, lang: str = "en"):
    """Download and prepare Wikipedia samples."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Wikipedia ({lang}) samples...")
    dataset = load_dataset(
        "wikipedia",
        "20220301.en",
        split=f"train[:{num_samples}]",
        trust_remote_code=True,
    )

    texts = []
    for item in dataset:
        text = item["text"].strip()
        if len(text) > 100:
            texts.append(text)

    output_file = output_path / f"wiki_{lang}_{len(texts)}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n\n")

    print(f"Saved {len(texts)} articles to {output_file}")
    print(f"Total characters: {sum(len(t) for t in texts):,}")
    return str(output_file)


def prepare_simple_data(output_dir: str):
    """Create a small synthetic dataset for initial testing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    texts = [
        "The sun rises in the east and sets in the west.",
        "Artificial intelligence is transforming how we live and work.",
        "Python is a popular programming language for machine learning.",
        "The ocean covers more than seventy percent of the Earth's surface.",
        "Music has been an important part of human culture for thousands of years.",
        "Trees produce oxygen and absorb carbon dioxide from the atmosphere.",
        "The human brain contains approximately one hundred billion neurons.",
        "Water boils at one hundred degrees Celsius at sea level.",
        "Light travels at approximately three hundred thousand kilometers per second.",
        "The Earth orbits the Sun once every three hundred sixty five days.",
        "Mathematics is the foundation of all natural sciences.",
        "Books have been a primary source of knowledge for centuries.",
        "The moon affects the tides of the oceans on Earth.",
        "Computers process information using binary code of zeros and ones.",
        "The heart pumps blood throughout the entire human body.",
        "Mountains are formed by the movement of tectonic plates.",
        "Stars are massive balls of hot gas that produce light and heat.",
        "The internet connects billions of devices around the world.",
        "Plants convert sunlight into energy through photosynthesis.",
        "History teaches us about the successes and failures of past civilizations.",
        "Türkiye, Avrupa ve Asya kıtaları arasında yer alan bir ülkedir.",
        "Yapay zeka, makinelerin insan benzeri görevleri yerine getirmesini sağlar.",
        "İstanbul, dünyada iki kıtada yer alan tek şehirdir.",
        "Bilim insanları evrenin yaklaşık on dört milyar yıl önce oluştuğunu düşünmektedir.",
        "Türk mutfağı, zengin lezzetleri ve çeşitliliği ile dünyaca ünlüdür.",
        "Matematik, bilimin temel yapı taşlarından biridir.",
        "Güneş sistemi sekiz gezegenden oluşmaktadır.",
        "Su, canlılar için en temel ihtiyaçlardan biridir.",
        "Teknoloji, günlük hayatımızı büyük ölçüde değiştirmektedir.",
        "Eğitim, toplumların gelişmesinde en önemli faktörlerden biridir.",
    ]

    output_file = output_path / "simple_train.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for _ in range(100):
            for text in texts:
                f.write(text + "\n")

    print(f"Saved simple training data to {output_file}")
    print(f"Total lines: {len(texts) * 100}")
    return str(output_file)


if __name__ == "__main__":
    data_dir = str(Path(__file__).parent / "raw")
    print("=== Preparing simple training data ===")
    simple_file = prepare_simple_data(data_dir)
    print(f"\nSimple data ready: {simple_file}")
    print("\nTo also download Wikipedia data, run:")
    print("  python prepare_data.py --wiki")

    if "--wiki" in sys.argv:
        print("\n=== Downloading Wikipedia data ===")
        wiki_file = prepare_wiki_data(data_dir, num_samples=5000)
        print(f"Wiki data ready: {wiki_file}")
