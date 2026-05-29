from __future__ import annotations

import argparse
import random
from pathlib import Path


QUESTIONS = [
    (
        "What is SVF-Transformer?",
        [
            "SVF-Transformer is a Transformer research baseline with a persistent structural state.",
            "The name SVF refers to Structural Vital Force, but the implementation treats the idea as an engineering constraint.",
            "The model combines token prediction with bounded structural dynamics, compressed memory, and attractor stabilization.",
            "A stable SVF model should be trainable, measurable, and easy to compare against a plain Transformer baseline.",
        ],
    ),
    (
        "Why does the persistent core exist?",
        [
            "The persistent core is designed to carry a compact summary across batches without storing the full history.",
            "It gives the model a slow state channel that can preserve structural context beyond a single sequence window.",
            "The core should remain small enough to inspect, reset, and ablate during experiments.",
            "It should help continuity while leaving ordinary language modeling to the Transformer backbone.",
        ],
    ),
    (
        "How does attractor dynamics improve stability?",
        [
            "Attractor dynamics pull the evolving core state toward a learned reference point.",
            "The attractor does not force a fixed answer; it provides a gentle restoring force after each update.",
            "If the pull is too weak, structural energy may drift upward during long runs.",
            "If the pull is too strong, the core can become rigid and stop carrying useful context.",
        ],
    ),
    (
        "How is structural conservation evaluated?",
        [
            "Structural conservation is evaluated by comparing the energy of consecutive core states.",
            "The conservation loss should stay small while prediction loss decreases.",
            "A useful evaluation report includes language loss, structural energy, conservation loss, and generated samples.",
            "The goal is not perfectly constant energy, but controlled and interpretable state movement.",
        ],
    ),
    (
        "What should memory compression do?",
        [
            "Memory compression should summarize useful history without replaying every activation.",
            "The ring buffer keeps memory bounded by replacing old summaries with newer summaries.",
            "Attention over memory allows the model to recover relevant traces when they help generation.",
            "The memory path should remain auxiliary so the backbone still learns local token patterns.",
        ],
    ),
    (
        "What makes a good SVF experiment?",
        [
            "A good SVF experiment changes one mechanism at a time and records its effect on stability.",
            "The experiment should compare checkpoints using fixed prompts and fixed generation settings.",
            "It should report speed, memory usage, prediction loss, structural energy, and sample quality.",
            "The strongest conclusions come from ablations rather than from a single successful run.",
        ],
    ),
]

EXPERIMENT_REPORTS = [
    [
        "Experiment: persistent core ablation.",
        "The baseline model is trained with the persistent core enabled and then trained again with the core removed.",
        "The comparison checks whether the core improves continuity across generated paragraphs.",
        "If the core helps, fixed prompts should produce fewer abrupt topic changes.",
        "If the core hurts, structural energy may look stable while language loss or sample quality becomes worse.",
    ],
    [
        "Experiment: conservation weight sweep.",
        "The run compares several conservation weights while keeping the dataset, optimizer, and model size fixed.",
        "A low weight may allow useful movement but can also permit energy drift.",
        "A high weight can reduce drift but may make the core too passive.",
        "The best setting keeps conservation visible without dominating token prediction.",
    ],
    [
        "Experiment: memory-enabled generation.",
        "The model generates samples with memory enabled and then repeats the same prompts with memory disabled.",
        "The report compares topic continuity, repeated phrases, and transitions between sections.",
        "Memory is useful only if it improves the sample without adding stale summaries.",
        "A bounded ring buffer makes the experiment practical on a single GPU.",
    ],
    [
        "Experiment: tokenizer comparison.",
        "The character-level model is useful for smoke tests but often breaks spelling.",
        "The word-level model improves spelling but struggles with flexible phrase composition.",
        "The BPE model is the best default because it balances spelling, vocabulary size, and generalization.",
        "The same prompts should be evaluated across all tokenizers to make the comparison fair.",
    ],
]

PROMPT_COMPLETIONS = [
    (
        "A stable SVF model should",
        [
            "A stable SVF model should reduce prediction loss while keeping structural energy bounded.",
            "A stable SVF model should generate coherent technical paragraphs without repeating one sentence too often.",
            "A stable SVF model should make its internal dynamics measurable through energy and conservation metrics.",
            "A stable SVF model should remain simple enough that every added mechanism can be tested by ablation.",
        ],
    ),
    (
        "The persistent core is designed to",
        [
            "The persistent core is designed to preserve a compact structural summary across batches.",
            "The persistent core is designed to update slowly, so it can carry context without destabilizing the backbone.",
            "The persistent core is designed to be small, bounded, and easy to inspect during debugging.",
            "The persistent core is designed to support continuity rather than replace attention.",
        ],
    ),
    (
        "In this experiment, structural conservation is evaluated by",
        [
            "In this experiment, structural conservation is evaluated by tracking energy changes between core states.",
            "In this experiment, structural conservation is evaluated by comparing loss curves with energy traces.",
            "In this experiment, structural conservation is evaluated by checking whether the core moves gradually during training.",
            "In this experiment, structural conservation is evaluated by pairing scalar metrics with fixed prompt samples.",
        ],
    ),
]

BRIDGE_SENTENCES = [
    "This design choice keeps the system close to ordinary engineering practice.",
    "The important point is that every mechanism should be measurable.",
    "The report should describe both successful behavior and failure cases.",
    "The model should be improved by controlled experiments rather than by adding untested modules.",
    "Clear prompts make it easier to compare checkpoints across training runs.",
]


def build_question_answer(rng: random.Random) -> str:
    question, answers = rng.choice(QUESTIONS)
    selected = rng.sample(answers, k=len(answers))
    return "Question: " + question + "\nAnswer: " + " ".join(selected)


def build_report(rng: random.Random) -> str:
    report = rng.choice(EXPERIMENT_REPORTS)
    lines = report[:]
    rng.shuffle(lines[1:])
    lines.append(rng.choice(BRIDGE_SENTENCES))
    return " ".join(lines)


def build_prompt_completion(rng: random.Random) -> str:
    prompt, completions = rng.choice(PROMPT_COMPLETIONS)
    lines = [rng.choice(completions)]
    lines.extend(rng.sample(completions, k=2))
    lines.append(rng.choice(BRIDGE_SENTENCES))
    return " ".join(dict.fromkeys(lines))


def build_corpus(target_kb: int, seed: int) -> str:
    rng = random.Random(seed)
    target_bytes = target_kb * 1024
    sections: list[str] = []
    builders = [build_question_answer, build_report, build_prompt_completion]
    while len("\n\n".join(sections).encode("utf-8")) < target_bytes:
        sections.append(rng.choice(builders)(rng))
    return "\n\n".join(sections) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an instruction-style SVF training corpus.")
    parser.add_argument("--out", type=Path, default=Path("data/svf_instruction_10mb.txt"))
    parser.add_argument("--target-kb", type=int, default=10240)
    parser.add_argument("--seed", type=int, default=2028)
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
