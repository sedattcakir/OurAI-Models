"""
Prepare training data for OurLLM.
Downloads Wikipedia articles and cleans them for language model training.
All data sources are public domain / permissive license.
"""
import re
import sys
from pathlib import Path
from datasets import load_dataset


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if len(line) < 20:
            continue
        if line.startswith(("==", "Category:", "[[", "{{", "|")):
            continue
        if re.match(r"^\d+\.\s*$", line):
            continue
        lines.append(line)
    return "\n".join(lines)


def prepare_wiki_data(output_dir: str, num_samples: int = 10000, lang: str = "en"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Wikipedia ({lang}), {num_samples} articles...")
    dataset = load_dataset(
        "wikimedia/wikipedia",
        f"20231101.{lang}",
        split=f"train[:{num_samples}]",
    )

    texts = []
    total_chars = 0
    for item in dataset:
        cleaned = clean_text(item["text"])
        if len(cleaned) > 200:
            texts.append(cleaned)
            total_chars += len(cleaned)

    output_file = output_path / f"wiki_{lang}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n\n")

    print(f"Saved {len(texts)} articles to {output_file}")
    print(f"Total characters: {total_chars:,}")
    return str(output_file)


def prepare_simple_data(output_dir: str):
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

    if "--wiki" in sys.argv:
        lang = "en"
        num = 10000
        for i, arg in enumerate(sys.argv):
            if arg == "--lang" and i + 1 < len(sys.argv):
                lang = sys.argv[i + 1]
            if arg == "--num" and i + 1 < len(sys.argv):
                num = int(sys.argv[i + 1])

        print(f"=== Downloading Wikipedia ({lang}) ===")
        wiki_file = prepare_wiki_data(data_dir, num_samples=num, lang=lang)
        print(f"Data ready: {wiki_file}")
    else:
        print("=== Preparing simple training data ===")
        simple_file = prepare_simple_data(data_dir)
        print(f"\nSimple data ready: {simple_file}")
        print("\nTo download Wikipedia data:")
        print("  python prepare_data.py --wiki --lang en --num 10000")
        print("  python prepare_data.py --wiki --lang tr --num 5000")
