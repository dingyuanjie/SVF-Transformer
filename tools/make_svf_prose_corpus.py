from __future__ import annotations

import argparse
import random
from pathlib import Path


OPENINGS = [
    "SVF-Transformer can be understood as a compact research baseline for structural dynamics in sequence models.",
    "A stable SVF model starts from ordinary language modeling and adds a small number of measurable structural mechanisms.",
    "The main design goal is not to simulate consciousness, but to keep long-term state bounded, inspectable, and useful.",
    "The architecture is easiest to study when each mechanism has a clear metric and a clear ablation.",
]

CORE_PARAGRAPHS = [
    [
        "The persistent core is designed to preserve a compact structural summary across batches.",
        "It updates more slowly than token activations, which allows it to carry continuity without replacing attention.",
        "A small core is easier to inspect during debugging and easier to remove during ablation.",
        "When the core helps, generated text tends to maintain a topic across section boundaries.",
        "When the core hurts, the model may show stable energy while language quality becomes less coherent.",
    ],
    [
        "Attractor dynamics provide a learned stabilizing reference for the evolving core state.",
        "The attractor should pull gently enough that the state can still adapt to the current sequence.",
        "If the pull is too weak, structural energy may drift upward during long training runs.",
        "If the pull is too strong, the state can become rigid and stop carrying useful context.",
        "The useful range is visible when loss decreases while conservation remains small.",
    ],
    [
        "Structural conservation is evaluated by tracking changes in core energy across updates.",
        "The conservation term is not meant to freeze the model or make every state identical.",
        "Its purpose is to discourage abrupt jumps that make the structural path difficult to interpret.",
        "A good report places conservation next to prediction loss, generated samples, and memory usage.",
        "This makes training failures easier to diagnose before a long run consumes too much time.",
    ],
    [
        "Memory compression should summarize useful history rather than replay every activation.",
        "A ring buffer keeps the memory bounded by replacing old summaries with newer summaries.",
        "Attention over compressed memory can recover relevant traces when they help generation.",
        "The memory path should remain auxiliary, because the Transformer backbone must still learn local patterns.",
        "A memory ablation is useful when it compares the same prompts with memory enabled and disabled.",
    ],
    [
        "A practical training curriculum begins with short sequences and modest batch sizes.",
        "After the baseline is stable, the context window can increase to test longer dependencies.",
        "The batch size should grow only after GPU memory usage is understood.",
        "Fixed prompts make it easier to compare checkpoints across training runs.",
        "The best checkpoint should be selected by validation loss rather than by the final training step alone.",
    ],
    [
        "BPE tokenization is a useful middle ground between character-level and word-level modeling.",
        "Character-level models are easy to test, but they often break spelling and long phrases.",
        "Word-level models improve spelling, but they handle unfamiliar phrase composition poorly.",
        "BPE keeps frequent words intact while still allowing new terms to be represented by subword pieces.",
        "For SVF-Transformer, BPE gives cleaner samples without making the vocabulary unnecessarily large.",
    ],
]

CONNECTORS = [
    "This matters because every structural mechanism should be measured rather than assumed useful.",
    "The same idea also appears in generation, where fixed prompts reveal repetition and topic drift.",
    "The engineering value comes from being able to remove the mechanism and compare the result.",
    "The model should stay close enough to a plain Transformer that a baseline comparison remains meaningful.",
    "A controlled experiment is more useful than a larger system whose behavior cannot be isolated.",
]

CONCLUSIONS = [
    "In this sense, SVF-Transformer is best treated as an experimental scaffold for structural sequence modeling.",
    "The next useful improvement is to compare checkpoints with the same prompts and the same sampling parameters.",
    "A stronger result would show lower validation loss, bounded structural energy, and cleaner prompt-conditioned samples.",
    "The system becomes more credible when each added component improves a measured behavior.",
]


def build_paragraph(rng: random.Random) -> str:
    paragraph = []
    if rng.random() < 0.35:
        paragraph.append(rng.choice(OPENINGS))
    blocks = rng.sample(CORE_PARAGRAPHS, k=rng.randint(1, 3))
    for block in blocks:
        selected = rng.sample(block, k=rng.randint(3, len(block)))
        paragraph.extend(selected)
        if rng.random() < 0.7:
            paragraph.append(rng.choice(CONNECTORS))
    if rng.random() < 0.55:
        paragraph.append(rng.choice(CONCLUSIONS))
    return " ".join(paragraph)


def build_corpus(target_kb: int, seed: int) -> str:
    rng = random.Random(seed)
    target_bytes = target_kb * 1024
    paragraphs: list[str] = []
    current_bytes = 0
    while current_bytes < target_bytes:
        paragraph = build_paragraph(rng)
        paragraphs.append(paragraph)
        current_bytes += len(paragraph.encode("utf-8")) + 2
    return "\n\n".join(paragraphs) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate continuous prose SVF training corpus.")
    parser.add_argument("--out", type=Path, default=Path("data/svf_prose_20mb.txt"))
    parser.add_argument("--target-kb", type=int, default=20480)
    parser.add_argument("--seed", type=int, default=2030)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = build_corpus(args.target_kb, args.seed)
    args.out.write_text(text, encoding="utf-8")
    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
