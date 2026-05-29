from __future__ import annotations

import argparse
import random
from pathlib import Path


TOPICS = [
    "SVF-Transformer",
    "structural dynamics",
    "persistent core",
    "attractor dynamics",
    "ring buffer memory",
    "attention compression",
    "structural conservation",
    "world modeling",
    "causal structure",
    "stable sequence learning",
    "hierarchical objective",
    "latent world model",
    "gradient stability",
    "sequence memory",
    "energy regularization",
    "state compression",
    "long horizon prediction",
    "modular routing",
    "training curriculum",
    "evaluation protocol",
]

SENTENCE_TEMPLATES = [
    "{topic} keeps the hidden state bounded while the sequence model learns local patterns.",
    "A stable model should reduce prediction loss without allowing structural energy to explode.",
    "The persistent core stores a compressed trace of previous batches and returns a small context vector.",
    "Attractor dynamics pull the core state toward a learnable stable point after each update.",
    "Ring buffer memory avoids unbounded growth by replacing old summaries with newer summaries.",
    "Attention compression reads only the useful memory traces instead of replaying the full history.",
    "The conservation term measures how much structural energy changes between two core states.",
    "A useful baseline must be simple enough to train and explicit enough to extend.",
    "The model is a research scaffold for testing structural drift, memory, and sequence prediction.",
    "When loss decreases and conservation remains small, the training run is behaving normally.",
    "Short experiments should verify forward pass, backward pass, checkpoint saving, and generation.",
    "Longer experiments can compare the baseline Transformer against the SVF-enhanced model.",
    "The structural state should help the network maintain continuity across batches.",
    "The first version favors stability over philosophical complexity.",
    "A bounded system is easier to debug than an unconstrained recursive system.",
    "During training, {topic} should improve continuity without dominating token prediction.",
    "The checkpoint records model weights, configuration, and the character vocabulary used by the corpus.",
    "A lower temperature usually produces safer samples, while a higher temperature explores more variants.",
    "Evaluation should compare loss, structural energy, conservation drift, and generated text quality.",
    "If the generated text becomes repetitive, the corpus should include broader phrasing and longer contexts.",
    "The baseline can be extended with sparse routing after the stable version is measured carefully.",
    "The memory path should remain auxiliary, so the Transformer backbone can still learn ordinary language patterns.",
    "A practical research system needs small tests before larger experiments.",
    "The attractor is not a fixed answer; it is a learned stabilizing reference for the structural state.",
    "The model can be trained on synthetic notes first, then tested on real papers, logs, or documentation.",
    "Structural energy is useful only when it helps diagnose the behavior of the training run.",
    "A clear ablation removes one module at a time and measures whether stability or prediction improves.",
    "Long horizon prediction requires both local token accuracy and a persistent summary of context.",
    "A good corpus mixes definitions, observations, hypotheses, failure cases, and implementation notes.",
    "The system should prefer gradual state updates instead of sudden uncontrolled jumps.",
    "In this experiment, {topic} acts as a measurable component rather than a vague metaphor.",
    "The training curriculum begins with short sequences and then increases the context window.",
    "A stable checkpoint should reload on CPU and produce similar samples from the same prompt and seed.",
    "Research notes are useful because they contain repeated concepts with enough linguistic variation.",
]

PARAGRAPH_OPENERS = [
    "Experiment note",
    "Training observation",
    "Model design",
    "Ablation idea",
    "Research log",
    "Implementation detail",
    "Stability check",
    "Memory behavior",
    "Failure case",
    "Comparison",
    "Hypothesis",
    "Metric note",
    "Curriculum step",
    "Generation sample",
    "Checkpoint note",
    "Architecture note",
]

CONNECTORS = [
    "Therefore",
    "However",
    "In practice",
    "For this reason",
    "During evaluation",
    "In a small run",
    "After several updates",
    "When the corpus grows",
]

CLAUSES = [
    "the model should keep loss decreasing smoothly",
    "the core state should avoid uncontrolled energy growth",
    "the memory should summarize rather than copy the full sequence",
    "the generated sample should remain close to the prompt topic",
    "the conservation term should stay small but not force the state to freeze",
    "the training loop should remain easy to inspect",
    "the baseline should remain simple enough for ablation",
    "the checkpoint should be reusable for generation and comparison",
]

TECHNICAL_SECTIONS = [
    (
        "Persistent Core Design",
        [
            "The persistent core is a compact state that carries information across training batches.",
            "It should not replace the Transformer backbone; it only adds a slow structural channel.",
            "The update rule combines a batch summary with the previous core state, then applies bounded drift.",
            "This design gives the model continuity without storing an unlimited history of tokens.",
            "A useful implementation keeps the core small enough to inspect during experiments.",
        ],
    ),
    (
        "Attractor Dynamics",
        [
            "Attractor dynamics provide a stabilizing reference for the evolving structural state.",
            "The attractor is learned, so it can adapt to the corpus while still discouraging runaway motion.",
            "When the attractor strength is too low, structural energy may drift upward for many updates.",
            "When the attractor strength is too high, the core can become too rigid to carry useful context.",
            "A practical setting should make the pull visible in metrics but gentle in the forward pass.",
        ],
    ),
    (
        "Structural Conservation",
        [
            "The conservation loss measures the change in structural energy between consecutive core states.",
            "Its purpose is not to make energy constant, but to discourage sudden jumps that destabilize training.",
            "If conservation remains small while prediction loss decreases, the structural path is behaving well.",
            "If conservation spikes repeatedly, the run should be inspected for an excessive learning rate or drift scale.",
            "The best baseline reports conservation next to language loss so failures are visible early.",
        ],
    ),
    (
        "Memory Compression",
        [
            "Ring buffer memory stores summaries instead of full activation histories.",
            "This keeps memory bounded even when the training run lasts for many thousands of steps.",
            "Attention compression lets the model read relevant summaries without replaying the entire buffer.",
            "The memory path should remain auxiliary, because the backbone must still learn ordinary sequence patterns.",
            "A good memory experiment compares generation quality with memory enabled and disabled.",
        ],
    ),
    (
        "Training Curriculum",
        [
            "A stable curriculum starts with short sequences, modest batch sizes, and frequent metric printing.",
            "After the baseline is stable, the context length can increase to test longer dependencies.",
            "The batch size should grow only after GPU memory usage is understood.",
            "Checkpoints should be saved often enough that a long run can be resumed after interruption.",
            "Generation samples should use fixed prompts so changes across checkpoints are easy to compare.",
        ],
    ),
    (
        "Evaluation Protocol",
        [
            "The evaluation protocol should track prediction loss, structural energy, conservation loss, and generated text.",
            "Prediction loss shows whether the model learns token patterns in the corpus.",
            "Structural energy shows whether the persistent state is growing, shrinking, or stabilizing.",
            "Generated text reveals repetition, topic drift, and broken transitions that scalar metrics can miss.",
            "A strong report includes both quantitative curves and a small set of fixed prompt samples.",
        ],
    ),
    (
        "Failure Analysis",
        [
            "Repetition usually means the model is overconfident or the corpus contains too many repeated templates.",
            "Broken phrases can come from character-level tokenization, small context windows, or overly aggressive sampling.",
            "Numeric artifacts often appear when synthetic paragraph numbers are treated as normal tokens.",
            "A cleaner corpus should contain coherent paragraphs rather than isolated shuffled statements.",
            "Each failure should lead to one controlled change, not a large bundle of unrelated edits.",
        ],
    ),
    (
        "Ablation Strategy",
        [
            "An ablation study removes one structural component at a time and compares the result against the baseline.",
            "Removing the persistent core tests whether long-term state improves continuity.",
            "Removing the conservation term tests whether energy control contributes to stable training.",
            "Disabling memory tests whether compressed summaries improve generated text or only add noise.",
            "The best ablation table includes speed, memory usage, loss, and sample quality.",
        ],
    ),
]

TECHNICAL_TRANSITIONS = [
    "The practical consequence is straightforward.",
    "This matters during long training runs.",
    "The implementation should make this behavior measurable.",
    "The same principle appears in the generation samples.",
    "The next experiment should isolate this factor.",
    "A careful baseline keeps this mechanism visible.",
]

TECHNICAL_CONCLUSIONS = [
    "This section treats the mechanism as an engineering constraint rather than a metaphor.",
    "The goal is a model that can be trained, measured, and compared.",
    "The baseline should remain small enough for fast experiments and clear enough for later extensions.",
    "Stable behavior is more important than adding another speculative module.",
    "The result should be judged by loss curves, energy traces, and prompt-conditioned samples.",
]


def build_paragraph(rng: random.Random, index: int, min_sentences: int, max_sentences: int) -> str:
    opener = rng.choice(PARAGRAPH_OPENERS)
    topic = rng.choice(TOPICS)
    lines = [f"{opener} {index}: {topic}."]
    for _ in range(rng.randint(min_sentences, max_sentences)):
        if rng.random() < 0.25:
            lines.append(f"{rng.choice(CONNECTORS)}, {rng.choice(CLAUSES)}.")
        else:
            template = rng.choice(SENTENCE_TEMPLATES)
            lines.append(template.format(topic=rng.choice(TOPICS)))
    return " ".join(lines)


def build_technical_paragraph(rng: random.Random, index: int, note_numbers: bool) -> str:
    title, sentences = rng.choice(TECHNICAL_SECTIONS)
    selected = rng.sample(sentences, k=rng.randint(3, len(sentences)))
    related_title, related_sentences = rng.choice(TECHNICAL_SECTIONS)
    while related_title == title:
        related_title, related_sentences = rng.choice(TECHNICAL_SECTIONS)

    heading = f"{title} - note {index}." if note_numbers else f"{title}."
    paragraph = [heading]
    paragraph.extend(selected)
    paragraph.append(rng.choice(TECHNICAL_TRANSITIONS))
    paragraph.append(rng.choice(related_sentences))
    if rng.random() < 0.7:
        paragraph.append(rng.choice(TECHNICAL_CONCLUSIONS))
    return " ".join(paragraph)


def build_corpus(
    target_kb: int,
    seed: int,
    min_sentences: int,
    max_sentences: int,
    style: str,
    note_numbers: bool = True,
) -> str:
    rng = random.Random(seed)
    target_bytes = target_kb * 1024
    paragraphs: list[str] = []
    index = 1
    while len("\n\n".join(paragraphs).encode("utf-8")) < target_bytes:
        if style == "technical":
            paragraphs.append(build_technical_paragraph(rng, index, note_numbers))
        else:
            paragraphs.append(build_paragraph(rng, index, min_sentences, max_sentences))
        index += 1
    return "\n\n".join(paragraphs) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic SVF training corpus.")
    parser.add_argument("--out", type=Path, default=Path("data/svf_corpus.txt"))
    parser.add_argument("--target-kb", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-sentences", type=int, default=5)
    parser.add_argument("--max-sentences", type=int, default=10)
    parser.add_argument("--style", choices=["synthetic", "technical"], default="synthetic")
    parser.add_argument(
        "--no-note-numbers",
        action="store_true",
        help="Omit numeric note ids in technical-style headings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = build_corpus(
        args.target_kb,
        args.seed,
        args.min_sentences,
        args.max_sentences,
        args.style,
        note_numbers=not args.no_note_numbers,
    )
    args.out.write_text(text, encoding="utf-8")
    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
