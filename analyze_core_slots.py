from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class SlotAnalysisResult:
    delay_tokens: int
    variant: str
    seed: int
    trace_count: int
    slot_count: int
    slot_usage_entropy: float
    normalized_slot_usage_entropy: float
    dominant_slot_fraction: float
    mean_offdiag_cosine: float
    max_offdiag_cosine: float
    min_offdiag_cosine: float
    mean_slot_norm_std: float
    mean_core_norm: float


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def parse_trace_path(path: Path) -> tuple[int, str, int]:
    stem = path.stem
    # delay4096_persistent_core_seed43
    delay_part, *variant_parts, seed_part = stem.split("_")
    delay_tokens = int(delay_part.replace("delay", ""))
    seed = int(seed_part.replace("seed", ""))
    variant = "_".join(variant_parts)
    return delay_tokens, variant, seed


def analyze_trace(path: Path) -> SlotAnalysisResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    delay_tokens, variant, seed = parse_trace_path(path)
    entries = payload["trace_entries"]
    if not entries:
        raise ValueError(f"No trace entries in {path}")

    slot_count = len(entries[0]["slot_norms"])
    usage_counts = [0 for _ in range(slot_count)]
    offdiag_cosines: list[float] = []
    slot_norm_stds: list[float] = []
    core_norms: list[float] = []

    for entry in entries:
        slot_norms = entry["slot_norms"]
        dominant_slot = max(range(slot_count), key=lambda index: slot_norms[index])
        usage_counts[dominant_slot] += 1
        slot_mean = sum(slot_norms) / max(len(slot_norms), 1)
        variance = sum((value - slot_mean) ** 2 for value in slot_norms) / max(len(slot_norms), 1)
        slot_norm_stds.append(math.sqrt(variance))
        core_norms.append(float(entry["core_mean_norm"]))

        core_state = entry["core_state"]
        for left in range(slot_count):
            for right in range(left + 1, slot_count):
                offdiag_cosines.append(cosine(core_state[left], core_state[right]))

    total = sum(usage_counts)
    probabilities = [count / total for count in usage_counts if count > 0]
    entropy = -sum(prob * math.log(prob, 2) for prob in probabilities)
    max_entropy = math.log(slot_count, 2) if slot_count > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return SlotAnalysisResult(
        delay_tokens=delay_tokens,
        variant=variant,
        seed=seed,
        trace_count=len(entries),
        slot_count=slot_count,
        slot_usage_entropy=entropy,
        normalized_slot_usage_entropy=normalized_entropy,
        dominant_slot_fraction=max(usage_counts) / max(total, 1),
        mean_offdiag_cosine=sum(offdiag_cosines) / max(len(offdiag_cosines), 1),
        max_offdiag_cosine=max(offdiag_cosines) if offdiag_cosines else 0.0,
        min_offdiag_cosine=min(offdiag_cosines) if offdiag_cosines else 0.0,
        mean_slot_norm_std=sum(slot_norm_stds) / max(len(slot_norm_stds), 1),
        mean_core_norm=sum(core_norms) / max(len(core_norms), 1),
    )


def write_outputs(output_dir: Path, results: list[SlotAnalysisResult]) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"slot_analysis_{timestamp}.json"
    csv_path = output_dir / f"slot_analysis_{timestamp}.csv"
    md_path = output_dir / f"slot_analysis_{timestamp}.md"

    json_path.write_text(json.dumps([asdict(item) for item in results], indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    lines = [
        "# Slot Analysis",
        "",
        "| delay | variant | seed | norm_entropy | dominant_frac | mean_offdiag_cos | mean_slot_norm_std | mean_core_norm |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result.delay_tokens} | {result.variant} | {result.seed} | "
            f"{result.normalized_slot_usage_entropy:.4f} | {result.dominant_slot_fraction:.4f} | "
            f"{result.mean_offdiag_cosine:.4f} | {result.mean_slot_norm_std:.6f} | {result.mean_core_norm:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze slot usage and similarity from core traces.")
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_dir = args.trace_dir
    trace_paths = sorted(trace_dir.glob("*.json"))
    if not trace_paths:
        raise FileNotFoundError(f"No trace json files found in {trace_dir}")
    results = [analyze_trace(path) for path in trace_paths]
    output_dir = args.output_dir if args.output_dir is not None else trace_dir.parent
    json_path, csv_path, md_path = write_outputs(output_dir, results)
    print(f"wrote slot analysis to {json_path}")
    print(f"wrote slot analysis to {csv_path}")
    print(f"wrote slot analysis to {md_path}")


if __name__ == "__main__":
    main()
