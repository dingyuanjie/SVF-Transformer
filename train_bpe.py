from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader, Dataset

from models import SVFTransformer, SVFTransformerConfig
from train import DEFAULT_TEXT


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class BPEDataset(Dataset):
    def __init__(self, ids: list[int], seq_len: int) -> None:
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


def train_tokenizer(text: str, vocab_size: int, min_frequency: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator([text], trainer=trainer)
    return tokenizer


def split_ids(ids: list[int], val_fraction: float) -> tuple[list[int], list[int]]:
    if not 0 < val_fraction < 0.5:
        raise ValueError("--val-fraction must be greater than 0 and less than 0.5")
    split = max(2, int(len(ids) * (1.0 - val_fraction)))
    split = min(split, len(ids) - 2)
    return ids[:split], ids[split:]


@torch.no_grad()
def evaluate_loss(
    model: SVFTransformer,
    loader: DataLoader,
    device: str,
    max_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    core_state = None
    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        if core_state is not None and core_state.size(0) != x.size(0):
            core_state = None
        out = model(x, targets=y, core_state=core_state, write_memory=False)
        core_state = out.core_state.detach()
        assert out.loss is not None
        losses.append(float(out.loss.item()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def save_checkpoint(
    path: str,
    model: SVFTransformer,
    config: SVFTransformerConfig,
    tokenizer: Tokenizer,
    step: int,
    val_loss: float | None,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.__dict__,
            "tokenizer_json": tokenizer.to_str(),
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BPE/subword SVF-Transformer.")
    parser.add_argument("--data", type=str, default=None, help="Optional UTF-8 text file.")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default="svf_bpe_transformer.pt")
    parser.add_argument("--save-best", type=str, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--sample-prompt", type=str, default="Attractor dynamics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    text = load_text(args.data)
    tokenizer = train_tokenizer(text, vocab_size=args.vocab_size, min_frequency=args.min_frequency)
    all_ids = tokenizer.encode(text).ids
    train_ids, val_ids = split_ids(all_ids, args.val_fraction)
    train_dataset = BPEDataset(train_ids, seq_len=args.seq_len)
    val_dataset = BPEDataset(val_ids, seq_len=args.seq_len)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    config = SVFTransformerConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        max_seq_len=args.seq_len,
    )
    model = SVFTransformer(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"BPE vocab size: {tokenizer.get_vocab_size()}")
    print(f"BPE train token count: {len(train_dataset.data)}")
    print(f"BPE val token count: {len(val_dataset.data)}")

    model.train()
    step = 0
    core_state = None
    best_val_loss = float("inf")
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
            if args.eval_interval > 0 and step % args.eval_interval == 0:
                val_loss = evaluate_loss(model, val_loader, args.device, args.eval_batches)
                print(f"eval step={step:04d} val_loss={val_loss:.4f}")
                if args.save_best is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(args.save_best, model, config, tokenizer, step, val_loss)
                    print(f"saved best checkpoint to {args.save_best} val_loss={val_loss:.4f}")

            if step >= args.steps:
                break

    final_val_loss = evaluate_loss(model, val_loader, args.device, args.eval_batches)
    save_checkpoint(args.checkpoint, model, config, tokenizer, step, final_val_loss)
    print(f"saved checkpoint to {args.checkpoint}")
    print(f"final val_loss={final_val_loss:.4f}")

    prompt_ids = tokenizer.encode(args.sample_prompt).ids
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
    sample = model.generate(prompt, max_new_tokens=80, temperature=0.8, top_k=40)[0].cpu()
    print("sample:")
    print(tokenizer.decode(sample.tolist()))


if __name__ == "__main__":
    main()
