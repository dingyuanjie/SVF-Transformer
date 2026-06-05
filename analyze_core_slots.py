from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


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
    mean_slot_routing_entropy: float
    mean_slot_read_entropy: float
    mean_offdiag_cosine: float
    max_offdiag_cosine: float
    min_offdiag_cosine: float
    mean_slot_norm_std: float
    mean_core_norm: float


@dataclass
class SlotAnalysisDetails:
    dominant_write_counts: list[int]
    mean_write_routing_weights: list[float]
    dominant_read_counts: list[int]
    mean_read_routing_weights: list[float]
    query_name_write_counts: dict[str, list[int]]
    query_name_read_counts: dict[str, list[int]]
    query_field_write_counts: dict[str, list[int]]
    query_field_read_counts: dict[str, list[int]]


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


def normalize_weights(weights: list[float]) -> list[float]:
    positive_weights = [max(float(weight), 1e-12) for weight in weights]
    total_weight = sum(positive_weights)
    return [weight / total_weight for weight in positive_weights]


def mean_vector(values: list[list[float]], slot_count: int) -> list[float]:
    if not values:
        return [0.0 for _ in range(slot_count)]
    return [
        sum(vector[slot_index] for vector in values) / len(values)
        for slot_index in range(slot_count)
    ]


def increment_mapping_slot(
    counts: dict[str, list[int]],
    key: str | None,
    slot_index: int,
    slot_count: int,
) -> None:
    if not key:
        return
    if key not in counts:
        counts[key] = [0 for _ in range(slot_count)]
    counts[key][slot_index] += 1


def counts_to_fraction_list(counts: list[int]) -> list[float]:
    total = sum(counts)
    if total == 0:
        return [0.0 for _ in counts]
    return [count / total for count in counts]


def summarize_mapping(counts: dict[str, list[int]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for key, slot_counts in sorted(counts.items()):
        dominant_slot = max(range(len(slot_counts)), key=lambda index: slot_counts[index])
        summary[key] = {
            "counts": slot_counts,
            "fractions": counts_to_fraction_list(slot_counts),
            "dominant_slot": dominant_slot,
            "dominant_fraction": counts_to_fraction_list(slot_counts)[dominant_slot],
        }
    return summary


def analyze_trace(path: Path) -> tuple[SlotAnalysisResult, SlotAnalysisDetails]:
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
    routing_entropies: list[float] = []
    read_entropies: list[float] = []
    dominant_read_counts = [0 for _ in range(slot_count)]
    write_routing_vectors: list[list[float]] = []
    read_routing_vectors: list[list[float]] = []
    query_name_write_counts: dict[str, list[int]] = {}
    query_name_read_counts: dict[str, list[int]] = {}
    query_field_write_counts: dict[str, list[int]] = {}
    query_field_read_counts: dict[str, list[int]] = {}

    for entry in entries:
        slot_norms = entry["slot_norms"]
        routing_weights = entry.get("slot_routing_weights")
        if routing_weights is not None:
            normalized_weights = normalize_weights(routing_weights)
            dominant_slot = max(range(slot_count), key=lambda index: normalized_weights[index])
            routing_entropies.append(-sum(weight * math.log(weight, 2) for weight in normalized_weights))
            write_routing_vectors.append(normalized_weights)
        else:
            dominant_slot = max(range(slot_count), key=lambda index: slot_norms[index])
            routing_entropies.append(0.0)
        usage_counts[dominant_slot] += 1
        increment_mapping_slot(query_name_write_counts, entry.get("query_name"), dominant_slot, slot_count)
        increment_mapping_slot(query_field_write_counts, entry.get("query_field"), dominant_slot, slot_count)

        read_weights = entry.get("slot_read_weights")
        if read_weights is not None:
            normalized_read_weights = normalize_weights(read_weights)
            dominant_read_slot = max(range(slot_count), key=lambda index: normalized_read_weights[index])
            dominant_read_counts[dominant_read_slot] += 1
            read_entropies.append(-sum(weight * math.log(weight, 2) for weight in normalized_read_weights))
            read_routing_vectors.append(normalized_read_weights)
            increment_mapping_slot(query_name_read_counts, entry.get("query_name"), dominant_read_slot, slot_count)
            increment_mapping_slot(query_field_read_counts, entry.get("query_field"), dominant_read_slot, slot_count)
        else:
            read_entropies.append(0.0)

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

    return (
        SlotAnalysisResult(
            delay_tokens=delay_tokens,
            variant=variant,
            seed=seed,
            trace_count=len(entries),
            slot_count=slot_count,
            slot_usage_entropy=entropy,
            normalized_slot_usage_entropy=normalized_entropy,
            dominant_slot_fraction=max(usage_counts) / max(total, 1),
            mean_slot_routing_entropy=sum(routing_entropies) / max(len(routing_entropies), 1),
            mean_slot_read_entropy=sum(read_entropies) / max(len(read_entropies), 1),
            mean_offdiag_cosine=sum(offdiag_cosines) / max(len(offdiag_cosines), 1),
            max_offdiag_cosine=max(offdiag_cosines) if offdiag_cosines else 0.0,
            min_offdiag_cosine=min(offdiag_cosines) if offdiag_cosines else 0.0,
            mean_slot_norm_std=sum(slot_norm_stds) / max(len(slot_norm_stds), 1),
            mean_core_norm=sum(core_norms) / max(len(core_norms), 1),
        ),
        SlotAnalysisDetails(
            dominant_write_counts=usage_counts,
            mean_write_routing_weights=mean_vector(write_routing_vectors, slot_count),
            dominant_read_counts=dominant_read_counts,
            mean_read_routing_weights=mean_vector(read_routing_vectors, slot_count),
            query_name_write_counts=query_name_write_counts,
            query_name_read_counts=query_name_read_counts,
            query_field_write_counts=query_field_write_counts,
            query_field_read_counts=query_field_read_counts,
        ),
    )


def build_summary(results: list[SlotAnalysisResult], details_list: list[SlotAnalysisDetails]) -> list[dict[str, Any]]:
    grouped_results: dict[tuple[int, str], list[SlotAnalysisResult]] = defaultdict(list)
    grouped_details: dict[tuple[int, str], list[SlotAnalysisDetails]] = defaultdict(list)
    for result, details in zip(results, details_list):
        key = (result.delay_tokens, result.variant)
        grouped_results[key].append(result)
        grouped_details[key].append(details)

    summaries: list[dict[str, Any]] = []
    for (delay_tokens, variant), variant_results in sorted(grouped_results.items()):
        variant_details = grouped_details[(delay_tokens, variant)]
        slot_count = variant_results[0].slot_count
        write_counts = [0 for _ in range(slot_count)]
        read_counts = [0 for _ in range(slot_count)]
        write_means = [0.0 for _ in range(slot_count)]
        read_means = [0.0 for _ in range(slot_count)]
        query_name_write: dict[str, list[int]] = {}
        query_name_read: dict[str, list[int]] = {}
        query_field_write: dict[str, list[int]] = {}
        query_field_read: dict[str, list[int]] = {}

        for details in variant_details:
            for slot_index in range(slot_count):
                write_counts[slot_index] += details.dominant_write_counts[slot_index]
                read_counts[slot_index] += details.dominant_read_counts[slot_index]
                write_means[slot_index] += details.mean_write_routing_weights[slot_index]
                read_means[slot_index] += details.mean_read_routing_weights[slot_index]
            for key, counts in details.query_name_write_counts.items():
                if key not in query_name_write:
                    query_name_write[key] = [0 for _ in range(slot_count)]
                for slot_index in range(slot_count):
                    query_name_write[key][slot_index] += counts[slot_index]
            for key, counts in details.query_name_read_counts.items():
                if key not in query_name_read:
                    query_name_read[key] = [0 for _ in range(slot_count)]
                for slot_index in range(slot_count):
                    query_name_read[key][slot_index] += counts[slot_index]
            for key, counts in details.query_field_write_counts.items():
                if key not in query_field_write:
                    query_field_write[key] = [0 for _ in range(slot_count)]
                for slot_index in range(slot_count):
                    query_field_write[key][slot_index] += counts[slot_index]
            for key, counts in details.query_field_read_counts.items():
                if key not in query_field_read:
                    query_field_read[key] = [0 for _ in range(slot_count)]
                for slot_index in range(slot_count):
                    query_field_read[key][slot_index] += counts[slot_index]

        summaries.append(
            {
                "delay_tokens": delay_tokens,
                "variant": variant,
                "runs": len(variant_results),
                "slot_count": slot_count,
                "mean_normalized_slot_usage_entropy": sum(
                    item.normalized_slot_usage_entropy for item in variant_results
                )
                / len(variant_results),
                "mean_dominant_slot_fraction": sum(item.dominant_slot_fraction for item in variant_results)
                / len(variant_results),
                "mean_slot_routing_entropy": sum(item.mean_slot_routing_entropy for item in variant_results)
                / len(variant_results),
                "mean_slot_read_entropy": sum(item.mean_slot_read_entropy for item in variant_results)
                / len(variant_results),
                "mean_offdiag_cosine": sum(item.mean_offdiag_cosine for item in variant_results)
                / len(variant_results),
                "mean_slot_norm_std": sum(item.mean_slot_norm_std for item in variant_results)
                / len(variant_results),
                "dominant_write_slot_fractions": counts_to_fraction_list(write_counts),
                "mean_write_slot_weights": [value / len(variant_details) for value in write_means],
                "dominant_read_slot_fractions": counts_to_fraction_list(read_counts),
                "mean_read_slot_weights": [value / len(variant_details) for value in read_means],
                "query_name_write_summary": summarize_mapping(query_name_write),
                "query_name_read_summary": summarize_mapping(query_name_read),
                "query_field_write_summary": summarize_mapping(query_field_write),
                "query_field_read_summary": summarize_mapping(query_field_read),
            }
        )
    return summaries


def write_outputs(
    output_dir: Path,
    results: list[SlotAnalysisResult],
    summary: list[dict[str, Any]],
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"slot_analysis_{timestamp}.json"
    csv_path = output_dir / f"slot_analysis_{timestamp}.csv"
    md_path = output_dir / f"slot_analysis_{timestamp}.md"
    summary_json_path = output_dir / f"slot_analysis_summary_{timestamp}.json"

    json_path.write_text(json.dumps([asdict(item) for item in results], indent=2), encoding="utf-8")
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    lines = [
        "# Slot Analysis",
        "",
        "| delay | variant | seed | norm_entropy | dominant_frac | write_entropy | read_entropy | mean_offdiag_cos | mean_slot_norm_std | mean_core_norm |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result.delay_tokens} | {result.variant} | {result.seed} | "
            f"{result.normalized_slot_usage_entropy:.4f} | {result.dominant_slot_fraction:.4f} | "
            f"{result.mean_slot_routing_entropy:.4f} | {result.mean_slot_read_entropy:.4f} | "
            f"{result.mean_offdiag_cosine:.4f} | "
            f"{result.mean_slot_norm_std:.6f} | {result.mean_core_norm:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Routing Summary",
            "",
            "| delay | variant | write_top_slot_frac | write_mean_weights | read_top_slot_frac | read_mean_weights |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in summary:
        lines.append(
            f"| {item['delay_tokens']} | {item['variant']} | "
            f"`{', '.join(f'{value:.2f}' for value in item['dominant_write_slot_fractions'])}` | "
            f"`{', '.join(f'{value:.2f}' for value in item['mean_write_slot_weights'])}` | "
            f"`{', '.join(f'{value:.2f}' for value in item['dominant_read_slot_fractions'])}` | "
            f"`{', '.join(f'{value:.2f}' for value in item['mean_read_slot_weights'])}` |"
        )
        if item["query_name_write_summary"]:
            lines.extend(["", f"### {item['variant']} query_name -> write slot", ""])
            for key, value in item["query_name_write_summary"].items():
                lines.append(
                    f"- {key}: write={value['dominant_slot']} "
                    f"fractions={', '.join(f'{fraction:.2f}' for fraction in value['fractions'])}"
                )
        if item["query_name_read_summary"]:
            lines.extend(["", f"### {item['variant']} query_name -> read slot", ""])
            for key, value in item["query_name_read_summary"].items():
                lines.append(
                    f"- {key}: read={value['dominant_slot']} "
                    f"fractions={', '.join(f'{fraction:.2f}' for fraction in value['fractions'])}"
                )
        if item["query_field_write_summary"]:
            lines.extend(["", f"### {item['variant']} query_field -> write slot", ""])
            for key, value in item["query_field_write_summary"].items():
                lines.append(
                    f"- {key}: write={value['dominant_slot']} "
                    f"fractions={', '.join(f'{fraction:.2f}' for fraction in value['fractions'])}"
                )
        if item["query_field_read_summary"]:
            lines.extend(["", f"### {item['variant']} query_field -> read slot", ""])
            for key, value in item["query_field_read_summary"].items():
                lines.append(
                    f"- {key}: read={value['dominant_slot']} "
                    f"fractions={', '.join(f'{fraction:.2f}' for fraction in value['fractions'])}"
                )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, md_path, summary_json_path


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
    analyzed = [analyze_trace(path) for path in trace_paths]
    results = [result for result, _ in analyzed]
    details = [detail for _, detail in analyzed]
    summary = build_summary(results, details)
    output_dir = args.output_dir if args.output_dir is not None else trace_dir.parent
    json_path, csv_path, md_path, summary_json_path = write_outputs(output_dir, results, summary)
    print(f"wrote slot analysis to {json_path}")
    print(f"wrote slot analysis to {csv_path}")
    print(f"wrote slot analysis to {md_path}")
    print(f"wrote slot analysis summary to {summary_json_path}")


if __name__ == "__main__":
    main()
