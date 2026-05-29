from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from models import SVFTransformer, SVFTransformerConfig
from tokenizer import WordTokenizer
from train import DEFAULT_TEXT


class WordDataset(Dataset):
    def __init__(
        self,
        text: str,
        seq_len: int,
        min_freq: int = 1,
        max_vocab_size: Optional[int] = None,
    ) -> None:
        self.tokenizer = WordTokenizer.train(text, min_freq=min_freq, max_vocab_size=max_vocab_size)
        ids = self.tokenizer.encode(text)
        if len(ids) < seq_len + 2:
            repeats = (seq_len + 2) // max(len(ids), 1) + 1
            ids = ids * repeats
        self.seq_len = seq_len
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return max(1, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def load_text(path: Optional[str]) -> str:
    if path is None:
        return DEFAULT_TEXT
    return Path(path).read_text(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a word-level SVF-Transformer.")
    parser.add_argument("--data", type=str, default=None, help="Optional UTF-8 text file.")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=12000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default="svf_word_transformer.pt")
    parser.add_argument("--sample-prompt", type=str, default="Attractor dynamics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    dataset = WordDataset(
        load_text(args.data),
        seq_len=args.seq_len,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    config = SVFTransformerConfig(
        vocab_size=dataset.tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        max_seq_len=args.seq_len,
    )
    model = SVFTransformer(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"word vocab size: {dataset.tokenizer.vocab_size}")
    model.train()
    step = 0
    core_state = None
    while step < args.steps:
        for x, y in loader:
            x = x.to(args.device)
            y = y.to(args.device)
            if core_state is not None and core_state.size(0) != x.size(0):
                core_state = None

            out = model(x, targets=y, core_state=core_state, write_memory=True)
            core_state = out.core_state.detach()
            assert out.loss is not None

            optimizer.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if step % 20 == 0:
                print(
                    f"step={step:04d} "
                    f"loss={out.loss.item():.4f} "
                    f"energy={out.structural_energy.item():.6f} "
                    f"conservation={out.conservation_loss.item():.6f}"
                )

            step += 1
            if step >= args.steps:
                break

    torch.save(
        {
            "model": model.state_dict(),
            "config": config.__dict__,
            "tokenizer": {
                "stoi": dataset.tokenizer.stoi,
                "itos": dataset.tokenizer.itos,
            },
        },
        args.checkpoint,
    )
    print(f"saved checkpoint to {args.checkpoint}")

    prompt = torch.tensor([dataset.tokenizer.encode(args.sample_prompt)], dtype=torch.long, device=args.device)
    sample = model.generate(prompt, max_new_tokens=80, temperature=0.8, top_k=40)[0].cpu()
    print("sample:")
    print(dataset.tokenizer.decode(sample.tolist()))


if __name__ == "__main__":
    main()
