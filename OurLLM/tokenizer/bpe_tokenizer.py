"""
BPE Tokenizer for OurLLM.
Built from scratch using the tokenizers library for BPE training,
with a custom wrapper for our model's needs.
"""
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


class OurBPETokenizer:
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.tokenizer = None
        self._build_base_tokenizer()

    def _build_base_tokenizer(self):
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    def train(self, files: list[str], min_frequency: int = 2):
        """Train the BPE tokenizer on text files."""
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
        )
        self.tokenizer.train(files, trainer)
        print(f"Tokenizer trained. Vocab size: {self.tokenizer.get_vocab_size()}")

    def save(self, path: str):
        """Save tokenizer to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(path)
        print(f"Tokenizer saved to {path}")

    def load(self, path: str):
        """Load tokenizer from file."""
        self.tokenizer = Tokenizer.from_file(path)
        print(f"Tokenizer loaded from {path}. Vocab size: {self.tokenizer.get_vocab_size()}")

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs."""
        ids = self.tokenizer.encode(text).ids
        if add_special_tokens:
            ids = [BOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if skip_special_tokens:
            ids = [i for i in ids if i not in (PAD_ID, UNK_ID, BOS_ID, EOS_ID)]
        return self.tokenizer.decode(ids)

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> int | None:
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, id: int) -> str | None:
        return self.tokenizer.id_to_token(id)
