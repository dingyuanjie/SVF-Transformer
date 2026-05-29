from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from generate import apply_no_repeat_ngram, apply_repetition_penalty
from models import SVFTransformer, SVFTransformerConfig
from tokenizer import WordTokenizer


def load_checkpoint(path: Path, device: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def normalize_itos(itos: dict[Any, str]) -> dict[int, str]:
    return {int(k): v for k, v in itos.items()}


@torch.no_grad()
def generate_words(
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
    parser = argparse.ArgumentParser(description="Generate text from a word-level SVF-Transformer checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("svf_word_transformer.pt"))
    parser.add_argument("--prompt", type=str, default="Attractor dynamics")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    checkpoint = load_checkpoint(args.checkpoint, args.device)
    config = SVFTransformerConfig(**checkpoint["config"])
    tok_data = checkpoint["tokenizer"]
    tokenizer = WordTokenizer(stoi=tok_data["stoi"], itos=normalize_itos(tok_data["itos"]))

    model = SVFTransformer(config).to(args.device)
    model.load_state_dict(checkpoint["model"])

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.unk_id]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
    output = generate_words(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    print(tokenizer.decode(output[0].cpu().tolist()))


if __name__ == "__main__":
    main()
