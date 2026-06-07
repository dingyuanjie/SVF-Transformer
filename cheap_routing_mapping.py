from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class MappingSummary:
    dominant_slot: int
    dominant_fraction: float


@dataclass
class RoutingGroupSummary:
    variant: str
    balance_tag: str
    slot_count: int
    trace_files: int
    trace_entries: int
    dominant_write_slot_fractions: list[float]
    dominant_read_slot_fractions: list[float]
    mean_write_slot_weights: list[float]
    mean_read_slot_weights: list[float]
    effective_write_slots: int
    effective_read_slots: int
    write_effective_slot_indices: list[int]
    read_effective_slot_indices: list[int]
    query_name_write_map: dict[str, MappingSummary]
    query_name_read_map: dict[str, MappingSummary]
    query_field_write_map: dict[str, MappingSummary]
    query_field_read_map: dict[str, MappingSummary]
    collapsed_to_two_slots: bool


@dataclass
class QuestionVerdict:
    variant: str
    balance_tag: str
    answer: bool
    slot_2_is_fully_used: bool
    larger_slot_counts_collapse_to_two: bool
    slot_counts_checked: list[int]
    details: list[str]


def normalize_weights(weights: list[float]) -> list[float]:
    clipped = [max(float(weight), 1e-12) for weight in weights]
    total = sum(clipped)
    return [weight / total for weight in clipped]


def dominant_slot(weights: list[float]) -> int:
    return max(range(len(weights)), key=lambda index: weights[index])


def increment_key_slot(
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


def counts_to_fractions(counts: list[int]) -> list[float]:
    total = sum(counts)
    if total == 0:
        return [0.0 for _ in counts]
    return [count / total for count in counts]


def summarize_mapping(counts: dict[str, list[int]]) -> dict[str, MappingSummary]:
    summary: dict[str, MappingSummary] = {}
    for key, slot_counts in sorted(counts.items()):
        top_slot = dominant_slot(slot_counts)
        fractions = counts_to_fractions(slot_counts)
        summary[key] = MappingSummary(
            dominant_slot=top_slot,
            dominant_fraction=fractions[top_slot],
        )
    return summary


def effective_slot_indices(
    counts: list[int],
    mean_weights: list[float],
    *,
    min_count_fraction: float,
    min_mean_weight: float,
) -> list[int]:
    total = sum(counts)
    active_slots = {
        index
        for index, value in enumerate(mean_weights)
        if value >= min_mean_weight
    }
    if total > 0:
        active_slots.update(
            index
            for index, count in enumerate(counts)
            if (count / total) >= min_count_fraction
        )
    if not active_slots and counts:
        active_slots.add(dominant_slot(counts))
    return sorted(active_slots)


def parse_group_from_trace_path(path: Path) -> tuple[str, str, int]:
    core_traces_dir = path.parent
    slots_dir = core_traces_dir.parent
    balance_dir = slots_dir.parent
    variant_dir = balance_dir.parent
    slot_count = int(slots_dir.name.replace("slots_", ""))
    return variant_dir.name, balance_dir.name, slot_count


def analyze_group(
    trace_paths: list[Path],
    *,
    variant: str,
    balance_tag: str,
    slot_count: int,
    min_count_fraction: float,
    min_mean_weight: float,
) -> RoutingGroupSummary:
    dominant_write_counts = [0 for _ in range(slot_count)]
    dominant_read_counts = [0 for _ in range(slot_count)]
    write_weight_sums = [0.0 for _ in range(slot_count)]
    read_weight_sums = [0.0 for _ in range(slot_count)]
    write_weight_vectors = 0
    read_weight_vectors = 0
    trace_entries = 0

    query_name_write_counts: dict[str, list[int]] = {}
    query_name_read_counts: dict[str, list[int]] = {}
    query_field_write_counts: dict[str, list[int]] = {}
    query_field_read_counts: dict[str, list[int]] = {}

    for path in trace_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("trace_entries", [])
        for entry in entries:
            trace_entries += 1
            write_weights = entry.get("slot_routing_weights")
            if write_weights is None:
                write_weights = entry["slot_norms"]
            normalized_write = normalize_weights(write_weights)
            write_slot = dominant_slot(normalized_write)
            dominant_write_counts[write_slot] += 1
            write_weight_vectors += 1
            for index, value in enumerate(normalized_write):
                write_weight_sums[index] += value
            increment_key_slot(query_name_write_counts, entry.get("query_name"), write_slot, slot_count)
            increment_key_slot(query_field_write_counts, entry.get("query_field"), write_slot, slot_count)

            read_weights = entry.get("slot_read_weights")
            if read_weights is None:
                normalized_read = normalized_write
            else:
                normalized_read = normalize_weights(read_weights)
            read_slot = dominant_slot(normalized_read)
            dominant_read_counts[read_slot] += 1
            read_weight_vectors += 1
            for index, value in enumerate(normalized_read):
                read_weight_sums[index] += value
            increment_key_slot(query_name_read_counts, entry.get("query_name"), read_slot, slot_count)
            increment_key_slot(query_field_read_counts, entry.get("query_field"), read_slot, slot_count)

    mean_write_weights = [
        value / max(write_weight_vectors, 1)
        for value in write_weight_sums
    ]
    mean_read_weights = [
        value / max(read_weight_vectors, 1)
        for value in read_weight_sums
    ]
    write_slots = effective_slot_indices(
        dominant_write_counts,
        mean_write_weights,
        min_count_fraction=min_count_fraction,
        min_mean_weight=min_mean_weight,
    )
    read_slots = effective_slot_indices(
        dominant_read_counts,
        mean_read_weights,
        min_count_fraction=min_count_fraction,
        min_mean_weight=min_mean_weight,
    )
    return RoutingGroupSummary(
        variant=variant,
        balance_tag=balance_tag,
        slot_count=slot_count,
        trace_files=len(trace_paths),
        trace_entries=trace_entries,
        dominant_write_slot_fractions=counts_to_fractions(dominant_write_counts),
        dominant_read_slot_fractions=counts_to_fractions(dominant_read_counts),
        mean_write_slot_weights=mean_write_weights,
        mean_read_slot_weights=mean_read_weights,
        effective_write_slots=len(write_slots),
        effective_read_slots=len(read_slots),
        write_effective_slot_indices=write_slots,
        read_effective_slot_indices=read_slots,
        query_name_write_map=summarize_mapping(query_name_write_counts),
        query_name_read_map=summarize_mapping(query_name_read_counts),
        query_field_write_map=summarize_mapping(query_field_write_counts),
        query_field_read_map=summarize_mapping(query_field_read_counts),
        collapsed_to_two_slots=max(len(write_slots), len(read_slots)) <= 2,
    )


def analyze_experiment_root(
    input_root: Path,
    *,
    min_count_fraction: float,
    min_mean_weight: float,
) -> tuple[list[RoutingGroupSummary], list[QuestionVerdict]]:
    trace_paths = sorted(input_root.glob("**/core_traces/*.json"))
    if not trace_paths:
        raise FileNotFoundError(f"No trace files found under {input_root}")

    grouped_paths: dict[tuple[str, str, int], list[Path]] = defaultdict(list)
    for path in trace_paths:
        grouped_paths[parse_group_from_trace_path(path)].append(path)

    summaries = [
        analyze_group(
            paths,
            variant=variant,
            balance_tag=balance_tag,
            slot_count=slot_count,
            min_count_fraction=min_count_fraction,
            min_mean_weight=min_mean_weight,
        )
        for (variant, balance_tag, slot_count), paths in sorted(grouped_paths.items())
    ]
    verdicts = build_question_verdicts(summaries)
    return summaries, verdicts


def build_question_verdicts(summaries: list[RoutingGroupSummary]) -> list[QuestionVerdict]:
    grouped: dict[tuple[str, str], list[RoutingGroupSummary]] = defaultdict(list)
    for item in summaries:
        grouped[(item.variant, item.balance_tag)].append(item)

    verdicts: list[QuestionVerdict] = []
    for (variant, balance_tag), items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: item.slot_count)
        slot_2 = next((item for item in items if item.slot_count == 2), None)
        slot_2_is_fully_used = bool(
            slot_2
            and slot_2.effective_write_slots == 2
            and slot_2.effective_read_slots == 2
        )
        larger_items = [item for item in items if item.slot_count > 2]
        larger_slot_counts_collapse_to_two = all(item.collapsed_to_two_slots for item in larger_items)
        details = []
        for item in items:
            details.append(
                f"slots={item.slot_count}: write={item.write_effective_slot_indices}, "
                f"read={item.read_effective_slot_indices}"
            )
        verdicts.append(
            QuestionVerdict(
                variant=variant,
                balance_tag=balance_tag,
                answer=slot_2_is_fully_used and larger_slot_counts_collapse_to_two,
                slot_2_is_fully_used=slot_2_is_fully_used,
                larger_slot_counts_collapse_to_two=larger_slot_counts_collapse_to_two,
                slot_counts_checked=[item.slot_count for item in items],
                details=details,
            )
        )
    return verdicts


def mapping_items_text(mapping: dict[str, MappingSummary], top_k: int) -> list[str]:
    items = list(mapping.items())[:top_k]
    return [
        f"- {key} -> slot {value.dominant_slot} ({value.dominant_fraction:.2f})"
        for key, value in items
    ]


def write_markdown_report(
    path: Path,
    *,
    summaries: list[RoutingGroupSummary],
    verdicts: list[QuestionVerdict],
    input_root: Path,
    top_k: int,
) -> None:
    lines = [
        "# Cheap Routing Mapping Report",
        "",
        f"- input_root: `{input_root}`",
        "",
        "## Main Question",
        "",
        "- Does routing fully use 2 slots, then keep using only 2 slots when capacity grows?",
        "",
        "## Verdict",
        "",
    ]
    for verdict in verdicts:
        answer = "YES" if verdict.answer else "NO"
        lines.append(
            f"- {verdict.variant} / {verdict.balance_tag}: {answer} "
            f"(slot_2_full={verdict.slot_2_is_fully_used}, "
            f"larger_collapse={verdict.larger_slot_counts_collapse_to_two})"
        )
        for detail in verdict.details:
            lines.append(f"  - {detail}")

    lines.extend(
        [
            "",
            "## Routing Groups",
            "",
        ]
    )
    for item in summaries:
        lines.extend(
            [
                f"### {item.variant} / {item.balance_tag} / slots_{item.slot_count}",
                "",
                f"- trace_files: `{item.trace_files}`",
                f"- trace_entries: `{item.trace_entries}`",
                f"- effective_write_slots: `{item.write_effective_slot_indices}`",
                f"- effective_read_slots: `{item.read_effective_slot_indices}`",
                f"- dominant_write_slot_fractions: `{', '.join(f'{value:.2f}' for value in item.dominant_write_slot_fractions)}`",
                f"- dominant_read_slot_fractions: `{', '.join(f'{value:.2f}' for value in item.dominant_read_slot_fractions)}`",
                f"- mean_write_slot_weights: `{', '.join(f'{value:.2f}' for value in item.mean_write_slot_weights)}`",
                f"- mean_read_slot_weights: `{', '.join(f'{value:.2f}' for value in item.mean_read_slot_weights)}`",
                "",
                "#### query_name -> write slot",
                "",
            ]
        )
        lines.extend(mapping_items_text(item.query_name_write_map, top_k=top_k) or ["- (empty)"])
        lines.extend(["", "#### query_field -> write slot", ""])
        lines.extend(mapping_items_text(item.query_field_write_map, top_k=top_k) or ["- (empty)"])
        lines.extend(["", "#### query_name -> read slot", ""])
        lines.extend(mapping_items_text(item.query_name_read_map, top_k=top_k) or ["- (empty)"])
        lines.extend(["", "#### query_field -> read slot", ""])
        lines.extend(mapping_items_text(item.query_field_read_map, top_k=top_k) or ["- (empty)"])
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_json_report(
    path: Path,
    *,
    summaries: list[RoutingGroupSummary],
    verdicts: list[QuestionVerdict],
    input_root: Path,
) -> None:
    payload = {
        "input_root": str(input_root),
        "groups": [
            {
                **asdict(item),
                "query_name_write_map": {key: asdict(value) for key, value in item.query_name_write_map.items()},
                "query_name_read_map": {key: asdict(value) for key, value in item.query_name_read_map.items()},
                "query_field_write_map": {key: asdict(value) for key, value in item.query_field_write_map.items()},
                "query_field_read_map": {key: asdict(value) for key, value in item.query_field_read_map.items()},
            }
            for item in summaries
        ],
        "verdicts": [asdict(item) for item in verdicts],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cheap routing mapper for Phase G traces. It answers whether routing collapses to two effective slots."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--min-count-fraction", type=float, default=0.10)
    parser.add_argument("--min-mean-weight", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_prefix = args.output_prefix or (args.input_root / "cheap_routing_mapping")
    summaries, verdicts = analyze_experiment_root(
        args.input_root,
        min_count_fraction=args.min_count_fraction,
        min_mean_weight=args.min_mean_weight,
    )
    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    write_json_report(json_path, summaries=summaries, verdicts=verdicts, input_root=args.input_root)
    write_markdown_report(
        md_path,
        summaries=summaries,
        verdicts=verdicts,
        input_root=args.input_root,
        top_k=args.top_k,
    )
    print(f"wrote cheap routing json to {json_path}")
    print(f"wrote cheap routing markdown to {md_path}")
    for verdict in verdicts:
        answer = "YES" if verdict.answer else "NO"
        print(f"{verdict.variant}/{verdict.balance_tag}: {answer}")


if __name__ == "__main__":
    main()
