from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from models import SVFTransformer, SVFTransformerConfig


def load_checkpoint(path: Path, device: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def normalize_itos(itos: dict[Any, str]) -> dict[int, str]:
    return {int(k): v for k, v in itos.items()}


def encode_prompt(prompt: str, stoi: dict[str, int], fallback_id: int) -> torch.Tensor:
    if not prompt:
        prompt = "SVF-Transformer"
    ids = [stoi.get(ch, fallback_id) for ch in prompt]
    return torch.tensor([ids], dtype=torch.long)


def decode_tokens(tokens: torch.Tensor, itos: dict[int, str]) -> str:
    return "".join(itos.get(int(token), "") for token in tokens)


def apply_repetition_penalty(logits: torch.Tensor, tokens: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty <= 1.0:
        return logits

    logits = logits.clone()
    for batch_idx in range(tokens.size(0)):
        seen = torch.unique(tokens[batch_idx])
        scores = logits[batch_idx, seen]
        logits[batch_idx, seen] = torch.where(scores < 0, scores * penalty, scores / penalty)
    return logits


def apply_no_repeat_ngram(logits: torch.Tensor, tokens: torch.Tensor, ngram_size: int) -> torch.Tensor:
    if ngram_size <= 1 or tokens.size(1) < ngram_size - 1:
        return logits

    logits = logits.clone()
    prefix_len = ngram_size - 1
    for batch_idx in range(tokens.size(0)):
        sequence = tokens[batch_idx].tolist()
        current_prefix = tuple(sequence[-prefix_len:])
        banned: set[int] = set()
        for i in range(len(sequence) - ngram_size + 1):
            ngram = sequence[i : i + ngram_size]
            if tuple(ngram[:-1]) == current_prefix:
                banned.add(ngram[-1])
        if banned:
            logits[batch_idx, list(banned)] = float("-inf")
    return logits


@torch.no_grad()
def generate_with_controls(
    model: SVFTransformer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> torch.Tensor:
    model.eval()
    core_state = None
    tokens = input_ids
    for _ in range(max_new_tokens):
        window = tokens[:, -model.config.max_seq_len :]
        out = model(window, core_state=core_state, use_memory=True, write_memory=False)
        core_state = out.core_state
        logits = out.logits[:, -1, :]
        logits = apply_repetition_penalty(logits, tokens, repetition_penalty)
        logits = apply_no_repeat_ngram(logits, tokens, no_repeat_ngram_size)
        logits = logits / max(temperature, 1e-6)

        if top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained SVF-Transformer checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("svf_transformer.pt"))
    parser.add_argument("--prompt", type=str, default="SVF-Transformer")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    checkpoint = load_checkpoint(args.checkpoint, args.device)
    config = SVFTransformerConfig(**checkpoint["config"])
    stoi: dict[str, int] = checkpoint["stoi"]
    itos = normalize_itos(checkpoint["itos"])

    model = SVFTransformer(config).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    fallback_id = stoi.get(" ", 0)
    input_ids = encode_prompt(args.prompt, stoi, fallback_id).to(args.device)
    output = generate_with_controls(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    print(decode_tokens(output[0].cpu(), itos))


if __name__ == "__main__":
    main()
