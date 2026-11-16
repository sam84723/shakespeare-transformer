"""Character-level Shakespeare dataset and tokenizer."""

import numpy as np
import torch
from torch.utils.data import Dataset


class CharTokenizer:
    """Maps each character of the training text to an integer id and back."""

    def __init__(self, text: str):
        self.vocab = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


class ShakespeareDataset(Dataset):
    """Every window of `block_size` characters, paired with the same window shifted by one."""

    def __init__(self, text: str, block_size: int, tokenizer: CharTokenizer):
        self.block_size = block_size
        self.data = torch.from_numpy(np.array(tokenizer.encode(text), dtype=np.int64))

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y


def load_datasets(train_path: str, test_path: str, block_size: int):
    """Build the tokenizer from the training text and return (tokenizer, train_set, test_set)."""
    with open(train_path, encoding="utf-8") as f:
        train_text = f.read()
    with open(test_path, encoding="utf-8") as f:
        test_text = f.read()

    tokenizer = CharTokenizer(train_text)
    return (
        tokenizer,
        ShakespeareDataset(train_text, block_size, tokenizer),
        ShakespeareDataset(test_text, block_size, tokenizer),
    )
