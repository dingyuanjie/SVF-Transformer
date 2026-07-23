from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from models import SVFTransformer, SVFTransformerConfig
from train_delayed_recall import DIGIT_TOKENS, DelayedRecallDataset, SyntheticTokenizer, TaskSpec


@dataclass
class ReplayExample:
    dataset_index: int
    query_name: str
    query_field: str
    original_value: str
    counterfactual_value: str
    original_write_slot: int
    counterfactual_write_slot: int
    original_read_slot: int
    counterfactual_read_slot: int


def normalize_weights(weights: list[float]) -> list[float]:
    clipped = [max(float(weight), 1e-12) for weight in weights]
    total = sum(clipped)
    return [weight / total for weight in clipped]


def dominant_slot(weights: list[float]) -> int:
    return max(range(len(weights)), key=lambda index: weights[index])


def l1_distance(left: list[float], right: list[float]) -> float:
    return sum(abs(float(a) - float(b)) for a, b in zip(left, right))


def load_synthetic_tokenizer(state: dict[str, Any]) -> SyntheticTokenizer:
    if state.get("type") != "synthetic":
        raise ValueError("Counterfactual replay currently supports synthetic delayed-recall checkpoints only.")
    return SyntheticTokenizer(
        noise_vocab_size=int(state["noise_vocab_size"]),
        fields=list(state["fields"]),
    )


def load_checkpoint(path: Path, device: str) -> tuple[SVFTransformer, SyntheticTokenizer, TaskSpec, dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = SVFTransformerConfig(**payload["config"])
    model = SVFTransformer(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    tokenizer = load_synthetic_tokenizer(payload["tokenizer"])
    task_spec = TaskSpec(**payload["task_spec"])
    saved_args = dict(payload.get("args") or {})
    return model, tokenizer, task_spec, saved_args, payload


def build_validation_dataset(
    *,
    saved_args: dict[str, Any],
    tokenizer: SyntheticTokenizer,
    task_spec: TaskSpec,
    delay_tokens: int,
) -> DelayedRecallDataset:
    seed = int(saved_args["seed"]) + 10_000
    return DelayedRecallDataset(
        num_samples=int(saved_args["val_samples"]),
        delay_tokens=delay_tokens,
        noise_vocab_size=int(saved_args["noise_vocab_size"]),
        seed=seed,
        tokenizer=tokenizer,
        task_spec=task_spec,
        field_order_mode=str(saved_args.get("field_order_mode", "fixed")),
        remember_position_mode=str(saved_args.get("remember_position_mode", "front")),
    )


def decode_full_tokens(dataset: DelayedRecallDataset, tokenizer: SyntheticTokenizer, index: int) -> list[str]:
    x, y, _ = dataset[index]
    ids = [int(x[0].item()), *[int(token_id) for token_id in y.tolist()]]
    return tokenizer.decode(ids)


def answer_value_tokens(metadata: dict[str, Any]) -> list[str]:
    return [str(token) for token in (metadata.get("answer_value") or [])]


def candidate_counterfactual_values(
    original_value: list[str],
    *,
    max_counterfactuals: int,
    rng: random.Random,
) -> list[list[str]]:
    if len(original_value) == 1:
        return [[digit] for digit in DIGIT_TOKENS if digit != original_value[0]][:max_counterfactuals]

    candidates: set[tuple[str, ...]] = set()
    original_tuple = tuple(original_value)
    attempts = 0
    while len(candidates) < max_counterfactuals and attempts < max_counterfactuals * 20:
        attempts += 1
        proposal = tuple(rng.choice(DIGIT_TOKENS) for _ in original_value)
        if proposal != original_tuple:
            candidates.add(proposal)
    return [list(candidate) for candidate in sorted(candidates)]


def apply_counterfactual_value(
    full_tokens: list[str],
    metadata: dict[str, Any],
    *,
    new_value_tokens: list[str],
) -> list[str]:
    updated = list(full_tokens)
    fact_value_start = int(metadata["query_fact_token_start"]) + 1
    fact_value_end = int(metadata["query_fact_token_end"])
    answer_start = int(metadata["answer_token_start"])
    answer_end = int(metadata["answer_token_end"])
    expected_length = fact_value_end - fact_value_start + 1
    if expected_length != len(new_value_tokens):
        raise ValueError("Counterfactual value length must match the original value span.")
    updated[fact_value_start : fact_value_end + 1] = new_value_tokens
    updated[answer_start : answer_end + 1] = new_value_tokens
    return updated


@torch.no_grad()
def run_sample(model: SVFTransformer, tokenizer: SyntheticTokenizer, tokens: list[str], device: str) -> dict[str, Any]:
    ids = tokenizer.encode(tokens)
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    model.reset_state()
    out = model(x, targets=y, core_state=None, use_memory=model.config.use_memory, write_memory=False)
    if out.slot_routing_weights is not None:
        write_weights = out.slot_routing_weights[0].detach().cpu().tolist()
    else:
        write_weights = normalize_weights(out.core_state[0].norm(dim=-1).detach().cpu().tolist())
    if out.slot_read_weights is not None:
        read_weights = out.slot_read_weights[0].detach().cpu().tolist()
    else:
        read_weights = write_weights
    return {
        "write_slot": dominant_slot(write_weights),
        "read_slot": dominant_slot(read_weights),
        "write_weights": write_weights,
        "read_weights": read_weights,
    }


def analyze_checkpoint(
    checkpoint_path: Path,
    *,
    device: str,
    max_samples: int,
    max_counterfactuals_per_sample: int,
    rng_seed: int,
    max_examples: int,
) -> dict[str, Any]:
    model, tokenizer, task_spec, saved_args, payload = load_checkpoint(checkpoint_path, device)
    dataset = build_validation_dataset(
        saved_args=saved_args,
        tokenizer=tokenizer,
        task_spec=task_spec,
        delay_tokens=int(payload["delay_tokens"]),
    )
    rng = random.Random(rng_seed)

    total_counterfactuals = 0
    write_slot_same = 0
    read_slot_same = 0
    write_l1_sum = 0.0
    read_l1_sum = 0.0
    per_field: dict[str, dict[str, float]] = {}
    examples: list[ReplayExample] = []

    for dataset_index in range(min(max_samples, len(dataset))):
        metadata = dataset.get_metadata(dataset_index)
        original_tokens = decode_full_tokens(dataset, tokenizer, dataset_index)
        original_run = run_sample(model, tokenizer, original_tokens, device)
        original_value = answer_value_tokens(metadata)
        for counterfactual_value in candidate_counterfactual_values(
            original_value,
            max_counterfactuals=max_counterfactuals_per_sample,
            rng=rng,
        ):
            counterfactual_tokens = apply_counterfactual_value(
                original_tokens,
                metadata,
                new_value_tokens=counterfactual_value,
            )
            counterfactual_run = run_sample(model, tokenizer, counterfactual_tokens, device)
            total_counterfactuals += 1
            write_same_flag = int(original_run["write_slot"] == counterfactual_run["write_slot"])
            read_same_flag = int(original_run["read_slot"] == counterfactual_run["read_slot"])
            write_slot_same += write_same_flag
            read_slot_same += read_same_flag
            write_l1_sum += l1_distance(original_run["write_weights"], counterfactual_run["write_weights"])
            read_l1_sum += l1_distance(original_run["read_weights"], counterfactual_run["read_weights"])

            field_key = str(metadata["query_field"])
            bucket = per_field.setdefault(
                field_key,
                {
                    "counterfactuals": 0.0,
                    "write_slot_same": 0.0,
                    "read_slot_same": 0.0,
                },
            )
            bucket["counterfactuals"] += 1
            bucket["write_slot_same"] += write_same_flag
            bucket["read_slot_same"] += read_same_flag

            if len(examples) < max_examples and (
                not write_same_flag or not read_same_flag
            ):
                examples.append(
                    ReplayExample(
                        dataset_index=dataset_index,
                        query_name=str(metadata["query_name"]),
                        query_field=field_key,
                        original_value="".join(original_value),
                        counterfactual_value="".join(counterfactual_value),
                        original_write_slot=int(original_run["write_slot"]),
                        counterfactual_write_slot=int(counterfactual_run["write_slot"]),
                        original_read_slot=int(original_run["read_slot"]),
                        counterfactual_read_slot=int(counterfactual_run["read_slot"]),
                    )
                )

    for bucket in per_field.values():
        total = max(bucket["counterfactuals"], 1.0)
        bucket["write_slot_same_rate"] = bucket["write_slot_same"] / total
        bucket["read_slot_same_rate"] = bucket["read_slot_same"] / total

    return {
        "checkpoint_path": str(checkpoint_path),
        "variant": payload.get("variant"),
        "delay_tokens": payload.get("delay_tokens"),
        "step": payload.get("step"),
        "task_spec": asdict(task_spec),
        "metadata": {
            "field_order_mode": saved_args.get("field_order_mode"),
            "remember_position_mode": saved_args.get("remember_position_mode"),
            "top1_routing": bool(saved_args.get("top1_routing", False)),
            "core_slots": saved_args.get("core_slots"),
        },
        "samples_evaluated": min(max_samples, len(dataset)),
        "total_counterfactuals": total_counterfactuals,
        "write_slot_same_rate": write_slot_same / max(total_counterfactuals, 1),
        "read_slot_same_rate": read_slot_same / max(total_counterfactuals, 1),
        "write_mean_l1_delta": write_l1_sum / max(total_counterfactuals, 1),
        "read_mean_l1_delta": read_l1_sum / max(total_counterfactuals, 1),
        "per_field": per_field,
        "examples": [asdict(example) for example in examples],
    }


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Counterfactual Value-Swap Replay",
        "",
        f"- checkpoint_path: `{summary['checkpoint_path']}`",
        f"- variant: `{summary['variant']}`",
        f"- delay_tokens: `{summary['delay_tokens']}`",
        f"- samples_evaluated: `{summary['samples_evaluated']}`",
        f"- total_counterfactuals: `{summary['total_counterfactuals']}`",
        f"- write_slot_same_rate: `{summary['write_slot_same_rate']:.4f}`",
        f"- read_slot_same_rate: `{summary['read_slot_same_rate']:.4f}`",
        f"- write_mean_l1_delta: `{summary['write_mean_l1_delta']:.4f}`",
        f"- read_mean_l1_delta: `{summary['read_mean_l1_delta']:.4f}`",
        "",
        "## Per Field",
        "",
    ]
    for field_name, bucket in sorted(summary["per_field"].items()):
        lines.append(
            f"- {field_name}: counterfactuals=`{int(bucket['counterfactuals'])}` "
            f"write_slot_same_rate=`{bucket['write_slot_same_rate']:.4f}` "
            f"read_slot_same_rate=`{bucket['read_slot_same_rate']:.4f}`"
        )
    lines.extend(["", "## Example Flips", ""])
    for example in summary["examples"]:
        lines.append(
            f"- idx={example['dataset_index']} {example['query_name']}.{example['query_field']}: "
            f"{example['original_value']} -> {example['counterfactual_value']} | "
            f"write {example['original_write_slot']} -> {example['counterfactual_write_slot']} | "
            f"read {example['original_read_slot']} -> {example['counterfactual_read_slot']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay delayed-recall checkpoints under true value-only counterfactuals.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-counterfactuals-per-sample", type=int, default=9)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-examples", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = analyze_checkpoint(
        args.checkpoint_path,
        device=args.device,
        max_samples=args.max_samples,
        max_counterfactuals_per_sample=args.max_counterfactuals_per_sample,
        rng_seed=args.seed,
        max_examples=args.max_examples,
    )
    json_path = output_dir / "counterfactual_value_swap_replay.json"
    md_path = output_dir / "counterfactual_value_swap_replay.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_report(md_path, summary)
    print(f"wrote counterfactual replay to {json_path}")
    print(f"wrote counterfactual replay to {md_path}")


if __name__ == "__main__":
    main()
