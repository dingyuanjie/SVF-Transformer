from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from generate_bpe import generate_bpe
from models import SVFTransformer, SVFTransformerConfig


DEFAULT_PROMPTS = [
    "Memory Compression",
    "Attractor Dynamics",
    "In this experiment, structural conservation is evaluated by",
    "A stable SVF model should",
    "The persistent core is designed to",
]


def load_checkpoint(path: Path, device: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a BPE SVF-Transformer checkpoint on fixed prompts.")
    parser.add_argument("--checkpoint", type=Path, default=Path("svf_bpe_transformer.pt"))
    parser.add_argument("--prompts", type=Path, default=None, help="Optional prompt file, one prompt per line.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--frequency-penalty", type=float, default=0.35)
    parser.add_argument("--presence-penalty", type=float, default=0.08)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--max-consecutive-token", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts)
    checkpoint = load_checkpoint(args.checkpoint, args.device)
    config = SVFTransformerConfig(**checkpoint["config"])
    tokenizer = Tokenizer.from_str(checkpoint["tokenizer_json"])

    model = SVFTransformer(config).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out_dir / f"eval_bpe_{timestamp}.txt"

    sections: list[str] = []
    header = [
        "SVF-Transformer BPE Evaluation",
        f"checkpoint: {args.checkpoint}",
        f"device: {args.device}",
        f"seed: {args.seed}",
        f"max_new_tokens: {args.max_new_tokens}",
        f"temperature: {args.temperature}",
        f"top_k: {args.top_k}",
        f"repetition_penalty: {args.repetition_penalty}",
        f"frequency_penalty: {args.frequency_penalty}",
        f"presence_penalty: {args.presence_penalty}",
        f"no_repeat_ngram_size: {args.no_repeat_ngram_size}",
        f"max_consecutive_token: {args.max_consecutive_token}",
        "",
    ]
    sections.append("\n".join(header))

    for idx, prompt in enumerate(prompts, start=1):
        torch.manual_seed(args.seed + idx)
        prompt_ids = tokenizer.encode(prompt).ids
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
        text = tokenizer.decode(output[0].cpu().tolist())
        section = [
            "=" * 80,
            f"Prompt {idx}: {prompt}",
            "-" * 80,
            text,
            "",
        ]
        sections.append("\n".join(section))
        print(f"\nPrompt {idx}: {prompt}\n{text}\n")

    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"saved evaluation to {out_path}")


if __name__ == "__main__":
    main()
