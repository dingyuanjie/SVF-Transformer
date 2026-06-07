from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def normalize_weights(weights: list[float]) -> list[float]:
    clipped = [max(float(weight), 1e-12) for weight in weights]
    total = sum(clipped)
    return [weight / total for weight in clipped]


def dominant_slot(weights: list[float]) -> int:
    return max(range(len(weights)), key=lambda index: weights[index])


def answer_value_text(entry: dict[str, Any]) -> str:
    value = entry.get("answer_value") or []
    return "".join(str(token) for token in value)


def field_order_signature(entry: dict[str, Any]) -> str:
    query_name = entry.get("query_name")
    orders = entry.get("field_orders_by_entity") or {}
    if query_name and query_name in orders:
        return ",".join(str(item) for item in orders[query_name])
    return ""


def structure_key(entry: dict[str, Any], *, include_query_name: bool) -> tuple[str, ...]:
    components = [
        f"query_field={entry.get('query_field')}",
        f"query_entity_index={entry.get('query_entity_index')}",
        f"query_field_index={entry.get('query_field_index')}",
        f"query_fact_token_start={entry.get('query_fact_token_start')}",
        f"query_fact_token_end={entry.get('query_fact_token_end')}",
        f"query_fact_to_context_distance={query_fact_to_context_distance(entry)}",
        f"query_fact_position_bucket={entry.get('query_fact_position_bucket')}",
        f"remember_position_mode={entry.get('remember_position_mode')}",
        f"field_order_signature={field_order_signature(entry)}",
    ]
    if include_query_name:
        components.insert(0, f"query_name={entry.get('query_name')}")
    return tuple(components)


def query_fact_to_context_distance(entry: dict[str, Any]) -> int | None:
    context_index = entry.get("context_token_index")
    fact_start = entry.get("query_fact_token_start")
    if context_index is None or fact_start is None:
        return None
    return int(context_index) - int(fact_start)


@dataclass
class PairExample:
    structure_key: str
    value_a: str
    slot_a: int
    value_b: str
    slot_b: int


@dataclass
class ProbeScore:
    name: str
    total_pairs: int
    slot_match_rate: float
    slot_flip_rate: float
    examples: list[PairExample]


def score_same_structure_diff_value(
    entries: list[dict[str, Any]],
    *,
    slot_key: str,
    include_query_name: bool,
    max_examples: int,
) -> ProbeScore:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for entry in entries:
        groups.setdefault(structure_key(entry, include_query_name=include_query_name), []).append(entry)

    total_pairs = 0
    match_pairs = 0
    examples: list[PairExample] = []
    for key, group in groups.items():
        for left, right in itertools.combinations(group, 2):
            if answer_value_text(left) == answer_value_text(right):
                continue
            total_pairs += 1
            if left[slot_key] == right[slot_key]:
                match_pairs += 1
            if len(examples) < max_examples:
                examples.append(
                    PairExample(
                        structure_key=" | ".join(key),
                        value_a=answer_value_text(left),
                        slot_a=left[slot_key],
                        value_b=answer_value_text(right),
                        slot_b=right[slot_key],
                    )
                )
    match_rate = (match_pairs / total_pairs) if total_pairs else 0.0
    return ProbeScore(
        name="same_structure_diff_value",
        total_pairs=total_pairs,
        slot_match_rate=match_rate,
        slot_flip_rate=1.0 - match_rate if total_pairs else 0.0,
        examples=examples,
    )


def score_same_value_diff_structure(
    entries: list[dict[str, Any]],
    *,
    slot_key: str,
    include_query_name: bool,
    max_examples: int,
) -> ProbeScore:
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        groups.setdefault(answer_value_text(entry), []).append(entry)

    total_pairs = 0
    match_pairs = 0
    examples: list[PairExample] = []
    for value, group in groups.items():
        for left, right in itertools.combinations(group, 2):
            if structure_key(left, include_query_name=include_query_name) == structure_key(
                right, include_query_name=include_query_name
            ):
                continue
            total_pairs += 1
            if left[slot_key] == right[slot_key]:
                match_pairs += 1
            if len(examples) < max_examples:
                examples.append(
                    PairExample(
                        structure_key="value=" + value,
                        value_a=" | ".join(structure_key(left, include_query_name=include_query_name)),
                        slot_a=left[slot_key],
                        value_b=" | ".join(structure_key(right, include_query_name=include_query_name)),
                        slot_b=right[slot_key],
                    )
                )
    match_rate = (match_pairs / total_pairs) if total_pairs else 0.0
    return ProbeScore(
        name="same_value_diff_structure",
        total_pairs=total_pairs,
        slot_match_rate=match_rate,
        slot_flip_rate=1.0 - match_rate if total_pairs else 0.0,
        examples=examples,
    )


def load_trace_entries(trace_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace_paths = sorted(trace_dir.glob("*.json"))
    if not trace_paths:
        raise FileNotFoundError(f"No trace json files found in {trace_dir}")

    entries: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    for path in trace_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = {
            key: payload.get(key)
            for key in [
                "task_type",
                "delay_tokens",
                "variant",
                "fields",
                "value_length",
                "top1_routing",
                "field_order_mode",
                "remember_position_mode",
            ]
        }
        for entry in payload.get("trace_entries", []):
            write_weights = normalize_weights(entry.get("slot_routing_weights") or entry["slot_norms"])
            read_weights = normalize_weights(entry.get("slot_read_weights") or write_weights)
            entries.append(
                {
                    **entry,
                    "answer_value_text": answer_value_text(entry),
                    "write_slot": dominant_slot(write_weights),
                    "read_slot": dominant_slot(read_weights),
                }
            )
    return entries, metadata


def analyze_trace_dir(trace_dir: Path, *, max_examples: int) -> dict[str, Any]:
    entries, metadata = load_trace_entries(trace_dir)
    probe_views = {
        "strict": {"include_query_name": True},
        "positional": {"include_query_name": False},
    }
    write_results: dict[str, dict[str, Any]] = {}
    read_results: dict[str, dict[str, Any]] = {}
    for view_name, options in probe_views.items():
        write_same_structure = score_same_structure_diff_value(
            entries,
            slot_key="write_slot",
            include_query_name=options["include_query_name"],
            max_examples=max_examples,
        )
        write_same_value = score_same_value_diff_structure(
            entries,
            slot_key="write_slot",
            include_query_name=options["include_query_name"],
            max_examples=max_examples,
        )
        read_same_structure = score_same_structure_diff_value(
            entries,
            slot_key="read_slot",
            include_query_name=options["include_query_name"],
            max_examples=max_examples,
        )
        read_same_value = score_same_value_diff_structure(
            entries,
            slot_key="read_slot",
            include_query_name=options["include_query_name"],
            max_examples=max_examples,
        )
        write_results[view_name] = {
            "same_structure_diff_value": asdict(write_same_structure),
            "same_value_diff_structure": asdict(write_same_value),
            "structure_minus_value_match_rate": (
                write_same_structure.slot_match_rate - write_same_value.slot_match_rate
            ),
        }
        read_results[view_name] = {
            "same_structure_diff_value": asdict(read_same_structure),
            "same_value_diff_structure": asdict(read_same_value),
            "structure_minus_value_match_rate": (
                read_same_structure.slot_match_rate - read_same_value.slot_match_rate
            ),
        }
    return {
        "trace_dir": str(trace_dir),
        "trace_entries": len(entries),
        "metadata": metadata,
        "write_probe": write_results,
        "read_probe": read_results,
    }


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Value-Swap Probe",
        "",
        f"- trace_dir: `{summary['trace_dir']}`",
        f"- trace_entries: `{summary['trace_entries']}`",
        "",
    ]
    for channel_key, title in [("write_probe", "Write Probe"), ("read_probe", "Read Probe")]:
        lines.extend([f"## {title}", ""])
        for view_name, result in summary[channel_key].items():
            lines.extend(
                [
                    f"### {view_name}",
                    "",
                    f"- structure_minus_value_match_rate: `{result['structure_minus_value_match_rate']:.4f}`",
                ]
            )
            for score_name in ["same_structure_diff_value", "same_value_diff_structure"]:
                score = result[score_name]
                lines.extend(
                    [
                        f"- {score_name}: pairs=`{score['total_pairs']}` "
                        f"slot_match_rate=`{score['slot_match_rate']:.4f}` "
                        f"slot_flip_rate=`{score['slot_flip_rate']:.4f}`",
                    ]
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze natural value-swap counterfactuals from routing traces."
    )
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.trace_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = analyze_trace_dir(args.trace_dir, max_examples=args.max_examples)
    json_path = output_dir / "value_swap_probe.json"
    md_path = output_dir / "value_swap_probe.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_report(md_path, summary)
    print(f"wrote value swap probe to {json_path}")
    print(f"wrote value swap probe to {md_path}")


if __name__ == "__main__":
    main()
