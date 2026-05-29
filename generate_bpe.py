from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from generate import apply_no_repeat_ngram, apply_repetition_penalty
from models import SVFTransformer, SVFTransformerConfig


def load_checkpoint(path: Path, device: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def generate_bpe(
    model: SVFTransformer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    frequency_penalty: float,
    presence_penalty: float,
    max_consecutive_token: int,
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
        logits = apply_token_penalties(logits, tokens, frequency_penalty, presence_penalty)
        logits = apply_consecutive_token_ban(logits, tokens, max_consecutive_token)
        logits = logits / max(temperature, 1e-6)

        if top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens


def apply_token_penalties(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    frequency_penalty: float,
    presence_penalty: float,
) -> torch.Tensor:
    if frequency_penalty <= 0 and presence_penalty <= 0:
        return logits

    logits = logits.clone()
    for batch_idx in range(tokens.size(0)):
        counts = torch.bincount(tokens[batch_idx], minlength=logits.size(-1)).to(logits.device)
        if frequency_penalty > 0:
            logits[batch_idx] -= counts * frequency_penalty
        if presence_penalty > 0:
            logits[batch_idx] -= (counts > 0).to(logits.dtype) * presence_penalty
    return logits


def apply_consecutive_token_ban(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    max_consecutive_token: int,
) -> torch.Tensor:
    if max_consecutive_token <= 0 or tokens.size(1) < max_consecutive_token:
        return logits

    logits = logits.clone()
    for batch_idx in range(tokens.size(0)):
        tail = tokens[batch_idx, -max_consecutive_token:]
        if torch.all(tail == tail[0]):
            logits[batch_idx, int(tail[0])] = float("-inf")
    return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a BPE/subword SVF-Transformer checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("svf_bpe_transformer.pt"))
    parser.add_argument("--prompt", type=str, default="Attractor dynamics")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--frequency-penalty", type=float, default=0.35)
    parser.add_argument("--presence-penalty", type=float, default=0.08)
    parser.add_argument("--max-consecutive-token", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    checkpoint = load_checkpoint(args.checkpoint, args.device)
    config = SVFTransformerConfig(**checkpoint["config"])
    tokenizer = Tokenizer.from_str(checkpoint["tokenizer_json"])

    model = SVFTransformer(config).to(args.device)
    model.load_state_dict(checkpoint["model"])

    prompt_ids = tokenizer.encode(args.prompt).ids
    if not prompt_ids:
        prompt_ids = [tokenizer.token_to_id("<unk>") or 0]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
    output = generate_bpe(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        max_consecutive_token=args.max_consecutive_token,
    )
    print(tokenizer.decode(output[0].cpu().tolist()))


if __name__ == "__main__":
    main()
