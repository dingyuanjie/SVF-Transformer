from __future__ import annotations

import re
from dataclasses import dataclass


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass
class WordTokenizer:
    stoi: dict[str, int]
    itos: dict[int, str]
    unk_token: str = "<unk>"
    pad_token: str = "<pad>"
    num_token: str = "<num>"

    @classmethod
    def train(cls, text: str, min_freq: int = 1, max_vocab_size: int | None = None) -> "WordTokenizer":
        counts: dict[str, int] = {}
        for token in basic_tokenize(text):
            token = normalize_token(token)
            counts[token] = counts.get(token, 0) + 1

        ordered = sorted(
            (item for item in counts.items() if item[1] >= min_freq),
            key=lambda item: (-item[1], item[0].lower()),
        )
        if max_vocab_size is not None:
            ordered = ordered[: max(0, max_vocab_size - 3)]

        vocab = ["<pad>", "<unk>", "<num>"] + [
            token for token, _ in ordered if token not in {"<pad>", "<unk>", "<num>"}
        ]
        stoi = {token: idx for idx, token in enumerate(vocab)}
        itos = {idx: token for token, idx in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(normalize_token(token), self.unk_id) for token in basic_tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        tokens = [self.itos.get(int(idx), self.unk_token) for idx in ids]
        return detokenize(tokens)


def basic_tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def normalize_token(token: str) -> str:
    if token.isdigit():
        return "<num>"
    return token


def detokenize(tokens: list[str]) -> str:
    text = ""
    no_space_before = {".", ",", ":", ";", "!", "?", ")", "]", "}", "%"}
    no_space_after = {"(", "[", "{", "$"}
    for token in tokens:
        if token in {"<pad>", "<unk>", "<num>"}:
            continue
        if not text:
            text = token
        elif token in no_space_before:
            text += token
        elif text[-1] in no_space_after:
            text += token
        else:
            text += " " + token
    return text
