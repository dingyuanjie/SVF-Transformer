from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class FactorValueSummary:
    value: str
    count: int
    dominant_slot: int
    dominant_fraction: float
    slot_fractions: list[float]


@dataclass
class FactorScore:
    name: str
    num_values: int
    predictability: float
    baseline_accuracy: float
    gain_over_baseline: float
    mutual_information: float
    normalized_mutual_information: float
    values: list[FactorValueSummary]


def normalize_weights(weights: list[float]) -> list[float]:
    clipped = [max(float(weight), 1e-12) for weight in weights]
    total = sum(clipped)
    return [weight / total for weight in clipped]


def dominant_slot(weights: list[float]) -> int:
    return max(range(len(weights)), key=lambda index: weights[index])


def counts_to_fractions(counts: list[int]) -> list[float]:
    total = sum(counts)
    if total == 0:
        return [0.0 for _ in counts]
    return [count / total for count in counts]


def effective_slot_indices(
    counts: list[int],
    mean_weights: list[float],
    *,
    min_count_fraction: float,
    min_mean_weight: float,
) -> list[int]:
    total = sum(counts)
    active = {
        index
        for index, value in enumerate(mean_weights)
        if value >= min_mean_weight
    }
    if total > 0:
        active.update(
            index
            for index, count in enumerate(counts)
            if (count / total) >= min_count_fraction
        )
    if not active and counts:
        active.add(dominant_slot(counts))
    return sorted(active)


def extract_factor_values(entry: dict[str, Any]) -> dict[str, str]:
    entity_names = entry.get("entity_names") or []
    query_name = entry.get("query_name")
    query_field = entry.get("query_field")
    answer_value = entry.get("answer_value") or []
    answer_text = "".join(str(token) for token in answer_value)
    query_entity_index = entry.get("query_entity_index")
    if query_entity_index is None and query_name in entity_names:
        query_entity_index = entity_names.index(query_name)
    query_field_index = entry.get("query_field_index")
    query_fact_token_start = entry.get("query_fact_token_start")
    query_fact_token_end = entry.get("query_fact_token_end")
    query_entity_token_start = entry.get("query_entity_token_start")
    query_entity_token_end = entry.get("query_entity_token_end")
    remember_token_index = entry.get("remember_token_index")
    question_token_index = entry.get("question_token_index")
    context_token_index = entry.get("context_token_index")
    prefix_noise_length = entry.get("prefix_noise_length")
    suffix_noise_length = entry.get("suffix_noise_length")
    answer_first_token = entry.get("answer_first_token")
    if answer_first_token is None and answer_value:
        answer_first_token = answer_value[0]

    factors: dict[str, str] = {}
    if query_name:
        factors["query_name"] = str(query_name)
        factors["query_name_initial"] = str(query_name)[0]
    if query_field:
        factors["query_field"] = str(query_field)
    if query_entity_index is not None:
        factors["query_entity_index"] = str(query_entity_index)
        factors["query_entity_parity"] = "even" if int(query_entity_index) % 2 == 0 else "odd"
    if entry.get("query_entity_bucket") is not None:
        factors["query_entity_bucket"] = str(entry["query_entity_bucket"])
    if query_field_index is not None:
        factors["query_field_index"] = str(query_field_index)
    if entry.get("field_order_mode") is not None:
        factors["field_order_mode"] = str(entry["field_order_mode"])
    if entry.get("remember_position_mode") is not None:
        factors["remember_position_mode"] = str(entry["remember_position_mode"])
    if answer_first_token is not None:
        factors["answer_first_token"] = str(answer_first_token)
    if answer_text:
        factors["answer_value"] = answer_text
        if all(token.isdigit() for token in answer_text):
            digit_sum = sum(int(token) for token in answer_text)
            factors["answer_digit_sum_parity"] = "even" if digit_sum % 2 == 0 else "odd"
    if entity_names:
        factors["entity_count"] = str(len(entity_names))
    if prefix_noise_length is not None:
        factors["prefix_noise_length"] = str(prefix_noise_length)
        factors["prefix_noise_bucket"] = position_bucket(int(prefix_noise_length), max(len(entity_names), 1) + 1)
    if suffix_noise_length is not None:
        factors["suffix_noise_length"] = str(suffix_noise_length)
        factors["suffix_noise_bucket"] = position_bucket(int(suffix_noise_length), max(len(entity_names), 1) + 1)
    if query_fact_token_start is not None:
        factors["query_fact_token_start"] = str(query_fact_token_start)
    if query_fact_token_end is not None:
        factors["query_fact_token_end"] = str(query_fact_token_end)
    if query_entity_token_start is not None:
        factors["query_entity_token_start"] = str(query_entity_token_start)
    if query_entity_token_end is not None:
        factors["query_entity_token_end"] = str(query_entity_token_end)
    if entry.get("query_fact_position_bucket") is not None:
        factors["query_fact_position_bucket"] = str(entry["query_fact_position_bucket"])
    query_fact_position_ratio = entry.get("query_fact_position_ratio")
    if query_fact_position_ratio is not None:
        factors["query_fact_position_ratio_bucket"] = ratio_bucket(float(query_fact_position_ratio))
    if remember_token_index is not None and question_token_index is not None:
        factors["remember_to_question_distance"] = str(int(question_token_index) - int(remember_token_index))
    if context_token_index is not None and query_fact_token_start is not None:
        factors["query_fact_to_context_distance"] = str(int(context_token_index) - int(query_fact_token_start))
    return factors


def position_bucket(value: int, scale_hint: int) -> str:
    if value <= 0:
        return "zero"
    if value <= max(1, scale_hint // 2):
        return "short"
    return "long"


def ratio_bucket(value: float) -> str:
    if value < 1.0 / 3.0:
        return "front"
    if value < 2.0 / 3.0:
        return "middle"
    return "back"


def compute_factor_score(
    factor_name: str,
    value_slot_counts: dict[str, list[int]],
    slot_totals: list[int],
    *,
    top_k_values: int,
) -> FactorScore:
    total = sum(slot_totals)
    baseline_accuracy = (max(slot_totals) / total) if total else 0.0
    weighted_predictability = 0.0
    joint_information = 0.0
    slot_probs = counts_to_fractions(slot_totals)
    value_probs = {
        value: sum(counts) / total
        for value, counts in value_slot_counts.items()
        if total > 0
    }
    value_summaries: list[FactorValueSummary] = []
    entropy_values = 0.0
    entropy_slots = 0.0

    for slot_prob in slot_probs:
        if slot_prob > 0:
            entropy_slots -= slot_prob * math.log(slot_prob, 2)

    for value, counts in sorted(value_slot_counts.items(), key=lambda item: (-sum(item[1]), item[0])):
        count = sum(counts)
        if count == 0 or total == 0:
            continue
        fractions = counts_to_fractions(counts)
        top_slot = dominant_slot(counts)
        top_fraction = fractions[top_slot]
        weighted_predictability += count * top_fraction
        value_prob = count / total
        entropy_values -= value_prob * math.log(value_prob, 2)
        for slot_index, joint_count in enumerate(counts):
            if joint_count == 0:
                continue
            joint_prob = joint_count / total
            joint_information += joint_prob * math.log(
                joint_prob / max(value_prob * slot_probs[slot_index], 1e-12),
                2,
            )
        value_summaries.append(
            FactorValueSummary(
                value=value,
                count=count,
                dominant_slot=top_slot,
                dominant_fraction=top_fraction,
                slot_fractions=fractions,
            )
        )

    nmi = 0.0
    if entropy_values > 0 and entropy_slots > 0:
        nmi = joint_information / math.sqrt(entropy_values * entropy_slots)
    predictability = weighted_predictability / total if total else 0.0
    return FactorScore(
        name=factor_name,
        num_values=len(value_slot_counts),
        predictability=predictability,
        baseline_accuracy=baseline_accuracy,
        gain_over_baseline=predictability - baseline_accuracy,
        mutual_information=joint_information,
        normalized_mutual_information=nmi,
        values=value_summaries[:top_k_values],
    )


def analyze_trace_dir(
    trace_dir: Path,
    *,
    min_count_fraction: float,
    min_mean_weight: float,
    top_k_factors: int,
    top_k_values: int,
) -> dict[str, Any]:
    trace_paths = sorted(trace_dir.glob("*.json"))
    if not trace_paths:
        raise FileNotFoundError(f"No trace json files found in {trace_dir}")

    write_entries: list[tuple[int, dict[str, str]]] = []
    read_entries: list[tuple[int, dict[str, str]]] = []
    write_slot_totals: list[int] | None = None
    read_slot_totals: list[int] | None = None
    write_weight_sums: list[float] | None = None
    read_weight_sums: list[float] | None = None
    trace_count = 0
    payload_meta: dict[str, Any] = {}

    for path in trace_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload_meta = {
            key: payload.get(key)
            for key in ["task_type", "delay_tokens", "variant", "seed", "fields", "value_length", "top1_routing"]
        }
        entries = payload.get("trace_entries", [])
        for entry in entries:
            trace_count += 1
            slot_count = len(entry.get("slot_routing_weights") or entry["slot_norms"])
            if write_slot_totals is None:
                write_slot_totals = [0 for _ in range(slot_count)]
                read_slot_totals = [0 for _ in range(slot_count)]
                write_weight_sums = [0.0 for _ in range(slot_count)]
                read_weight_sums = [0.0 for _ in range(slot_count)]
            assert write_slot_totals is not None and read_slot_totals is not None
            assert write_weight_sums is not None and read_weight_sums is not None

            write_weights = normalize_weights(entry.get("slot_routing_weights") or entry["slot_norms"])
            read_weights = normalize_weights(entry.get("slot_read_weights") or write_weights)
            write_slot = dominant_slot(write_weights)
            read_slot = dominant_slot(read_weights)
            write_slot_totals[write_slot] += 1
            read_slot_totals[read_slot] += 1
            for index, value in enumerate(write_weights):
                write_weight_sums[index] += value
            for index, value in enumerate(read_weights):
                read_weight_sums[index] += value
            factor_values = extract_factor_values(entry)
            write_entries.append((write_slot, factor_values))
            read_entries.append((read_slot, factor_values))

    assert write_slot_totals is not None and read_slot_totals is not None
    assert write_weight_sums is not None and read_weight_sums is not None

    mean_write_weights = [value / max(trace_count, 1) for value in write_weight_sums]
    mean_read_weights = [value / max(trace_count, 1) for value in read_weight_sums]
    active_write_slots = effective_slot_indices(
        write_slot_totals,
        mean_write_weights,
        min_count_fraction=min_count_fraction,
        min_mean_weight=min_mean_weight,
    )
    active_read_slots = effective_slot_indices(
        read_slot_totals,
        mean_read_weights,
        min_count_fraction=min_count_fraction,
        min_mean_weight=min_mean_weight,
    )

    def score_entries(entries: list[tuple[int, dict[str, str]]], slot_totals: list[int]) -> list[FactorScore]:
        factor_slot_counts: dict[str, dict[str, list[int]]] = {}
        for slot_index, factor_values in entries:
            for factor_name, factor_value in factor_values.items():
                if factor_name not in factor_slot_counts:
                    factor_slot_counts[factor_name] = {}
                if factor_value not in factor_slot_counts[factor_name]:
                    factor_slot_counts[factor_name][factor_value] = [0 for _ in slot_totals]
                factor_slot_counts[factor_name][factor_value][slot_index] += 1
        scores = [
            compute_factor_score(
                factor_name,
                value_counts,
                slot_totals,
                top_k_values=top_k_values,
            )
            for factor_name, value_counts in factor_slot_counts.items()
        ]
        return sorted(
            scores,
            key=lambda item: (
                -item.gain_over_baseline,
                -item.normalized_mutual_information,
                item.name,
            ),
        )[:top_k_factors]

    write_scores = score_entries(write_entries, write_slot_totals)
    read_scores = score_entries(read_entries, read_slot_totals)
    return {
        "trace_dir": str(trace_dir),
        "trace_files": len(trace_paths),
        "trace_entries": trace_count,
        "slot_count": len(write_slot_totals),
        "active_write_slots": active_write_slots,
        "active_read_slots": active_read_slots,
        "dominant_write_slot_fractions": counts_to_fractions(write_slot_totals),
        "dominant_read_slot_fractions": counts_to_fractions(read_slot_totals),
        "mean_write_slot_weights": mean_write_weights,
        "mean_read_slot_weights": mean_read_weights,
        "write_top_factors": [asdict(score) for score in write_scores],
        "read_top_factors": [asdict(score) for score in read_scores],
        "metadata": payload_meta,
    }


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Slot Routing Factor Analysis",
        "",
        f"- trace_dir: `{summary['trace_dir']}`",
        f"- trace_files: `{summary['trace_files']}`",
        f"- trace_entries: `{summary['trace_entries']}`",
        f"- slot_count: `{summary['slot_count']}`",
        f"- active_write_slots: `{summary['active_write_slots']}`",
        f"- active_read_slots: `{summary['active_read_slots']}`",
        f"- dominant_write_slot_fractions: `{', '.join(f'{value:.2f}' for value in summary['dominant_write_slot_fractions'])}`",
        f"- dominant_read_slot_fractions: `{', '.join(f'{value:.2f}' for value in summary['dominant_read_slot_fractions'])}`",
        "",
        "## Write Factors",
        "",
    ]
    for factor in summary["write_top_factors"]:
        lines.extend(
            [
                f"### {factor['name']}",
                "",
                f"- predictability: `{factor['predictability']:.4f}`",
                f"- baseline_accuracy: `{factor['baseline_accuracy']:.4f}`",
                f"- gain_over_baseline: `{factor['gain_over_baseline']:.4f}`",
                f"- normalized_mutual_information: `{factor['normalized_mutual_information']:.4f}`",
                "",
            ]
        )
        for value in factor["values"]:
            lines.append(
                f"- {value['value']}: slot {value['dominant_slot']} "
                f"fraction={value['dominant_fraction']:.2f} count={value['count']}"
            )
        lines.append("")

    lines.extend(["## Read Factors", ""])
    for factor in summary["read_top_factors"]:
        lines.extend(
            [
                f"### {factor['name']}",
                "",
                f"- predictability: `{factor['predictability']:.4f}`",
                f"- baseline_accuracy: `{factor['baseline_accuracy']:.4f}`",
                f"- gain_over_baseline: `{factor['gain_over_baseline']:.4f}`",
                f"- normalized_mutual_information: `{factor['normalized_mutual_information']:.4f}`",
                "",
            ]
        )
        for value in factor["values"]:
            lines.append(
                f"- {value['value']}: slot {value['dominant_slot']} "
                f"fraction={value['dominant_fraction']:.2f} count={value['count']}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze which factors explain slot routing choices.")
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-count-fraction", type=float, default=0.10)
    parser.add_argument("--min-mean-weight", type=float, default=0.10)
    parser.add_argument("--top-k-factors", type=int, default=5)
    parser.add_argument("--top-k-values", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.trace_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = analyze_trace_dir(
        args.trace_dir,
        min_count_fraction=args.min_count_fraction,
        min_mean_weight=args.min_mean_weight,
        top_k_factors=args.top_k_factors,
        top_k_values=args.top_k_values,
    )
    json_path = output_dir / "slot_factor_analysis.json"
    md_path = output_dir / "slot_factor_analysis.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_report(md_path, summary)
    print(f"wrote slot factor analysis to {json_path}")
    print(f"wrote slot factor analysis to {md_path}")


if __name__ == "__main__":
    main()
