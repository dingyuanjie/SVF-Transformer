from __future__ import annotations

import argparse
import random
from pathlib import Path


def read_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    if not blocks:
        raise ValueError(f"No text blocks found in {path}")
    return blocks


def weighted_pick(rng: random.Random, corpora: list[list[str]], weights: list[float]) -> str:
    index = rng.choices(range(len(corpora)), weights=weights, k=1)[0]
    return rng.choice(corpora[index])


def build_mixed_corpus(
    inputs: list[Path],
    weights: list[float],
    target_kb: int,
    seed: int,
) -> str:
    if len(inputs) != len(weights):
        raise ValueError("--inputs and --weights must have the same length")

    corpora = [read_blocks(path) for path in inputs]
    rng = random.Random(seed)
    target_bytes = target_kb * 1024
    blocks: list[str] = []
    current_bytes = 0
    while current_bytes < target_bytes:
        block = weighted_pick(rng, corpora, weights)
        blocks.append(block)
        current_bytes += len(block.encode("utf-8")) + 2
    return "\n\n".join(blocks) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mix multiple SVF corpora with sampling weights.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Input text files.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        required=True,
        help="Sampling weights, one per input.",
    )
    parser.add_argument("--out", type=Path, default=Path("data/svf_mixed_bpe_20mb.txt"))
    parser.add_argument("--target-kb", type=int, default=20480)
    parser.add_argument("--seed", type=int, default=2029)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = build_mixed_corpus(args.inputs, args.weights, args.target_kb, args.seed)
    args.out.write_text(text, encoding="utf-8")
    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
