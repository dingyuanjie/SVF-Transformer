from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from models import SVFTransformer, SVFTransformerConfig


DEFAULT_TEXT = """
SVF-Transformer is a stable baseline for structural dynamics.
The model keeps a persistent core, bounded structural drift, attractor dynamics,
and a compressed ring-buffer memory. The first goal is not complexity. The first
goal is to train without exploding.
""".strip()


class CharDataset(Dataset):
    def __init__(self, text: str, seq_len: int) -> None:
        if len(text) < seq_len + 2:
            repeats = (seq_len + 2) // max(len(text), 1) + 1
            text = (text + "\n") * repeats
        self.seq_len = seq_len
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def __len__(self) -> int:
        return max(1, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]

    def decode(self, ids: torch.Tensor) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def load_text(path: Optional[str]) -> str:
    if path is None:
        return DEFAULT_TEXT
    return Path(path).read_text(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal SVF-Transformer.")
    parser.add_argument("--data", type=str, default=None, help="Optional UTF-8 text file.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default="svf_transformer.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    dataset = CharDataset(load_text(args.data), seq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    config = SVFTransformerConfig(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        max_seq_len=args.seq_len,
    )
    model = SVFTransformer(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

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
            "stoi": dataset.stoi,
            "itos": dataset.itos,
        },
        args.checkpoint,
    )
    print(f"saved checkpoint to {args.checkpoint}")

    prompt = torch.tensor([[dataset.stoi[dataset.chars[0]]]], dtype=torch.long, device=args.device)
    sample = model.generate(prompt, max_new_tokens=80, temperature=0.9, top_k=20)[0].cpu()
    print("sample:")
    print(dataset.decode(sample))


if __name__ == "__main__":
    main()
