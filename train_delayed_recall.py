from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from models import SVFTransformer, SVFTransformerConfig, build_config_for_variant
from train_experiment import auto_match_variant_config, build_base_config, count_parameters


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]
CONTROL_TOKENS = [
    "remember",
    "context",
    "question",
    "answer",
    "sep",
    "lookup",
]
DIGIT_TOKENS = [str(value) for value in range(10)]
NAME_TOKENS = [
    "alice",
    "bob",
    "carol",
    "dave",
    "erin",
    "frank",
    "grace",
    "heidi",
    "ivan",
    "judy",
    "mallory",
    "niaj",
]
FIELD_TOKENS = ["code", "age", "city", "color", "pet", "fruit"]


@dataclass
class TaskSpec:
    task_type: str
    entities_per_sample: int
    fields: list[str]
    value_length: int


class SyntheticTokenizer:
    def __init__(self, noise_vocab_size: int, fields: list[str]) -> None:
        tokens = SPECIAL_TOKENS + CONTROL_TOKENS + DIGIT_TOKENS + NAME_TOKENS + sorted(set(fields))
        tokens += [f"noise_{index}" for index in range(noise_vocab_size)]
        self.stoi = {token: index for index, token in enumerate(tokens)}
        self.itos = {index: token for token, index in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi[token] for token in tokens]


def serialize_fact_key(name: str, field: str) -> str:
    return f"{name}.{field}"


def split_filler_tokens(
    filler: list[str],
    *,
    remember_position_mode: str,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    if remember_position_mode == "front":
        return [], filler
    if remember_position_mode == "middle":
        midpoint = len(filler) // 2
        return filler[:midpoint], filler[midpoint:]
    if remember_position_mode == "back":
        return filler, []
    if remember_position_mode == "random":
        split_index = rng.randint(0, len(filler))
        return filler[:split_index], filler[split_index:]
    raise ValueError(f"Unsupported remember_position_mode: {remember_position_mode}")


class DelayedRecallDataset(Dataset):
    def __init__(
        self,
        *,
        num_samples: int,
        delay_tokens: int,
        noise_vocab_size: int,
        seed: int,
        tokenizer: SyntheticTokenizer,
        task_spec: TaskSpec,
        field_order_mode: str = "fixed",
        remember_position_mode: str = "front",
    ) -> None:
        self.answer_length = task_spec.value_length
        self.samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.sample_metadata: list[dict[str, Any]] = []
        rng = random.Random(seed)
        noise_tokens = [f"noise_{index}" for index in range(noise_vocab_size)]

        if task_spec.entities_per_sample > len(NAME_TOKENS):
            raise ValueError("entities_per_sample exceeds available synthetic names.")

        for _ in range(num_samples):
            names = rng.sample(NAME_TOKENS, k=task_spec.entities_per_sample)
            facts: dict[tuple[str, str], list[str]] = {}
            facts_by_entity: dict[str, dict[str, list[str]]] = {}

            for name in names:
                facts_by_entity[name] = {}
                for field in task_spec.fields:
                    value = [rng.choice(DIGIT_TOKENS) for _ in range(task_spec.value_length)]
                    facts[(name, field)] = value
                    facts_by_entity[name][field] = value

            fact_tokens: list[str] = []
            entity_spans: dict[str, dict[str, int]] = {}
            fact_spans: dict[str, dict[str, int]] = {}
            field_orders_by_entity: dict[str, list[str]] = {}
            for entity_index, name in enumerate(names):
                entity_start = len(fact_tokens)
                fact_tokens.append(name)
                if field_order_mode == "shuffled":
                    field_order = list(task_spec.fields)
                    rng.shuffle(field_order)
                else:
                    field_order = list(task_spec.fields)
                field_orders_by_entity[name] = field_order
                for field in field_order:
                    value = facts[(name, field)]
                    field_start = len(fact_tokens)
                    fact_tokens.extend([field, *value])
                    field_end = len(fact_tokens) - 1
                    fact_spans[serialize_fact_key(name, field)] = {
                        "start": field_start,
                        "end": field_end,
                        "entity_index": entity_index,
                        "field_index": field_order.index(field),
                    }
                if entity_index < len(names) - 1:
                    fact_tokens.append("sep")
                entity_spans[name] = {
                    "start": entity_start,
                    "end": len(fact_tokens) - 1,
                    "entity_index": entity_index,
                }

            query_name = rng.choice(names)
            query_field = rng.choice(task_spec.fields)
            query_entity_index = names.index(query_name)
            query_field_index = field_orders_by_entity[query_name].index(query_field)
            answer_value = facts[(query_name, query_field)]
            filler = [rng.choice(noise_tokens) for _ in range(delay_tokens)]
            prefix_filler, suffix_filler = split_filler_tokens(
                filler,
                remember_position_mode=remember_position_mode,
                rng=rng,
            )
            tokens = [
                "<bos>",
                *prefix_filler,
                "remember",
                *fact_tokens,
                "context",
                *suffix_filler,
                "question",
                query_name,
                query_field,
                "answer",
                *answer_value,
                "<eos>",
            ]
            remember_token_index = 1 + len(prefix_filler)
            fact_tokens_start = remember_token_index + 1
            fact_tokens_end = fact_tokens_start + len(fact_tokens) - 1
            context_token_index = fact_tokens_end + 1
            suffix_filler_start = context_token_index + 1
            suffix_filler_end = suffix_filler_start + len(suffix_filler) - 1
            question_token_index = suffix_filler_end + 1 if suffix_filler else context_token_index + 1
            query_name_token_index = question_token_index + 1
            query_field_token_index = question_token_index + 2
            answer_start = len(tokens) - task_spec.value_length - 1
            answer_positions = list(range(answer_start - 1, answer_start - 1 + task_spec.value_length))
            answer_token_start = answer_start
            answer_token_end = answer_start + task_spec.value_length - 1
            query_fact_span = fact_spans[serialize_fact_key(query_name, query_field)]
            query_entity_span = entity_spans[query_name]
            query_fact_position_ratio = (
                (query_fact_span["start"] - fact_tokens_start) / max(len(fact_tokens) - 1, 1)
                if fact_tokens
                else 0.0
            )
            if query_fact_position_ratio < 1.0 / 3.0:
                query_fact_bucket = "front"
            elif query_fact_position_ratio < 2.0 / 3.0:
                query_fact_bucket = "middle"
            else:
                query_fact_bucket = "back"
            ids = tokenizer.encode(tokens)
            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)
            answer_mask = torch.zeros(len(y), dtype=torch.bool)
            for position in answer_positions:
                answer_mask[position] = True
            self.samples.append((x, y, answer_mask))
            self.sample_metadata.append(
                {
                    "entity_names": names,
                    "entity_count": len(names),
                    "query_name": query_name,
                    "query_name_initial": query_name[0],
                    "query_entity_index": query_entity_index,
                    "query_entity_bucket": (
                        "front_half"
                        if query_entity_index < max(1, math.ceil(len(names) / 2))
                        else "back_half"
                    ),
                    "query_field": query_field,
                    "query_field_index": query_field_index,
                    "answer_value": answer_value,
                    "answer_first_token": answer_value[0] if answer_value else None,
                    "field_order_mode": field_order_mode,
                    "remember_position_mode": remember_position_mode,
                    "remember_token_index": remember_token_index,
                    "fact_tokens_start": fact_tokens_start,
                    "fact_tokens_end": fact_tokens_end,
                    "context_token_index": context_token_index,
                    "prefix_noise_length": len(prefix_filler),
                    "suffix_noise_length": len(suffix_filler),
                    "question_token_index": question_token_index,
                    "query_name_token_index": query_name_token_index,
                    "query_field_token_index": query_field_token_index,
                    "answer_token_start": answer_token_start,
                    "answer_token_end": answer_token_end,
                    "query_entity_token_start": fact_tokens_start + query_entity_span["start"],
                    "query_entity_token_end": fact_tokens_start + query_entity_span["end"],
                    "query_fact_token_start": fact_tokens_start + query_fact_span["start"],
                    "query_fact_token_end": fact_tokens_start + query_fact_span["end"],
                    "query_fact_position_ratio": query_fact_position_ratio,
                    "query_fact_position_bucket": query_fact_bucket,
                    "entity_spans": {
                        key: {
                            **value,
                            "token_start": fact_tokens_start + value["start"],
                            "token_end": fact_tokens_start + value["end"],
                        }
                        for key, value in entity_spans.items()
                    },
                    "fact_spans": {
                        key: {
                            **value,
                            "token_start": fact_tokens_start + value["start"],
                            "token_end": fact_tokens_start + value["end"],
                        }
                        for key, value in fact_spans.items()
                    },
                    "field_orders_by_entity": field_orders_by_entity,
                    "facts_by_entity": facts_by_entity,
                }
            )

        self.sequence_length = len(self.samples[0][0])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[index]

    def get_metadata(self, index: int) -> dict[str, Any]:
        return self.sample_metadata[index]


@dataclass
class RecallMetrics:
    loss: float
    ce_loss: float
    answer_token_accuracy: float
    answer_exact_accuracy: float
    structural_energy: float
    drift_loss: float
    core_norm_mean: float
    core_slot_norm_std: float
    attractor_distance: float
    slot_diversity_loss: float
    slot_balance_loss: float


@dataclass
class RecallResult:
    task_type: str
    delay_tokens: int
    variant: str
    seed: int
    steps: int
    parameter_count: int
    final_train_loss: float
    final_val_loss: float
    final_val_ce_loss: float
    final_answer_token_accuracy: float
    final_answer_exact_accuracy: float
    final_val_structural_energy: float
    final_val_drift_loss: float
    final_core_norm_mean: float
    final_core_slot_norm_std: float
    final_attractor_distance: float
    final_slot_diversity_loss: float
    final_slot_balance_loss: float
    core_trace_path: Optional[str]


@dataclass
class RecallAggregate:
    task_type: str
    delay_tokens: int
    variant: str
    runs: int
    mean_val_ce_loss: float
    std_val_ce_loss: float
    mean_answer_token_accuracy: float
    std_answer_token_accuracy: float
    mean_answer_exact_accuracy: float
    std_answer_exact_accuracy: float
    mean_val_structural_energy: float
    mean_val_drift_loss: float
    mean_core_norm_mean: float
    mean_core_slot_norm_std: float
    mean_attractor_distance: float
    mean_slot_diversity_loss: float
    mean_slot_balance_loss: float
    mean_parameter_count: float


def build_task_spec(args: argparse.Namespace) -> TaskSpec:
    if args.task_type == "single_entity":
        return TaskSpec(
            task_type=args.task_type,
            entities_per_sample=1,
            fields=["code"],
            value_length=args.value_length,
        )
    return TaskSpec(
        task_type=args.task_type,
        entities_per_sample=args.entities_per_sample,
        fields=list(args.fields),
        value_length=args.value_length,
    )


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def resolve_effective_batch_size(args: argparse.Namespace, seq_len: int) -> int:
    if not args.adaptive_batch_size:
        return args.batch_size
    reference = max(args.adaptive_batch_reference_seq_len, 1)
    scale = (reference / max(seq_len, 1)) ** 2
    effective = max(1, int(math.floor(args.batch_size * scale)))
    return min(args.batch_size, effective)


def resolve_config(
    args: argparse.Namespace,
    *,
    variant: str,
    vocab_size: int,
    seq_len: int,
) -> SVFTransformerConfig:
    local_args = argparse.Namespace(**vars(args))
    local_args.seq_len = seq_len
    base_config = build_base_config(local_args, vocab_size)
    config = build_config_for_variant(base_config, variant)
    if variant == "baseline" and args.match_baseline_to is not None:
        reference_config = build_config_for_variant(base_config, args.match_baseline_to)
        target_params = count_parameters(SVFTransformer(reference_config))
        config = auto_match_variant_config(local_args, variant, vocab_size, target_params)
    config = SVFTransformerConfig(
        **{
            **config.__dict__,
            "use_slot_balance_loss": bool(args.slot_balance_loss and config.use_slot_routing),
            "slot_balance_weight": args.slot_balance_weight,
            "use_top1_routing": bool(args.top1_routing and config.use_slot_routing),
        }
    )
    return config


def measure_core_stats(model: SVFTransformer, core_state: torch.Tensor) -> tuple[float, float, float]:
    core_norm_mean = float(core_state.norm(dim=-1).mean().item())
    slot_norm_std = float(core_state.norm(dim=-1).std(dim=1).mean().item())
    if model.config.use_persistent_core and hasattr(model.core, "attractor"):
        attractor = model.core.attractor.unsqueeze(0).to(core_state.device)
        attractor_distance = float((core_state - attractor).pow(2).mean().sqrt().item())
    else:
        attractor_distance = 0.0
    return core_norm_mean, slot_norm_std, attractor_distance


@torch.no_grad()
def evaluate_variant(
    model: SVFTransformer,
    loader: DataLoader,
    *,
    device: str,
    collect_traces: bool = False,
    max_trace_batches: int = 0,
    max_trace_examples: int = 0,
) -> tuple[RecallMetrics, list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_energy = 0.0
    total_drift = 0.0
    total_core_norm = 0.0
    total_core_slot_std = 0.0
    total_attractor_distance = 0.0
    total_slot_diversity = 0.0
    total_slot_balance = 0.0
    batches = 0
    token_correct = 0
    token_total = 0
    exact_correct = 0
    example_total = 0
    traces: list[dict[str, Any]] = []
    dataset_offset = 0

    for batch_idx, (x, y, answer_mask) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        answer_mask = answer_mask.to(device)
        model.reset_state()
        out = model(x, targets=y, core_state=None, use_memory=model.config.use_memory, write_memory=False)
        assert out.loss is not None and out.ce_loss is not None
        total_loss += float(out.loss.item())
        total_ce += float(out.ce_loss.item())
        total_energy += float(out.structural_energy.item())
        total_drift += float(out.drift_loss.item())
        core_norm_mean, slot_norm_std, attractor_distance = measure_core_stats(model, out.core_state)
        total_core_norm += core_norm_mean
        total_core_slot_std += slot_norm_std
        total_attractor_distance += attractor_distance
        total_slot_diversity += float(out.slot_diversity_loss.item())
        total_slot_balance += float(out.slot_balance_loss.item())
        batches += 1

        predictions = out.logits.argmax(dim=-1)
        token_correct += int(((predictions == y) & answer_mask).sum().item())
        token_total += int(answer_mask.sum().item())
        exact_masked = (predictions == y) | (~answer_mask)
        exact_correct += int(exact_masked.all(dim=1).sum().item())
        example_total += int(x.size(0))

        if collect_traces and batch_idx < max_trace_batches:
            slot_norms = out.core_state.norm(dim=-1).detach().cpu()
            for sample_idx in range(min(max_trace_examples, x.size(0))):
                metadata: dict[str, Any] = {}
                if hasattr(loader.dataset, "get_metadata"):
                    metadata = loader.dataset.get_metadata(dataset_offset + sample_idx)
                traces.append(
                    {
                        "batch_index": batch_idx,
                        "sample_index": sample_idx,
                        "dataset_index": dataset_offset + sample_idx,
                        "slot_norms": slot_norms[sample_idx].tolist(),
                        "core_mean_norm": float(slot_norms[sample_idx].mean().item()),
                        "core_state": out.core_state[sample_idx].detach().cpu().tolist(),
                        "slot_routing_weights": (
                            out.slot_routing_weights[sample_idx].detach().cpu().tolist()
                            if out.slot_routing_weights is not None
                            else None
                        ),
                        "slot_read_weights": (
                            out.slot_read_weights[sample_idx].detach().cpu().tolist()
                            if out.slot_read_weights is not None
                            else None
                        ),
                        "structural_energy": float(out.structural_energy.item()),
                        "drift_loss": float(out.drift_loss.item()),
                        "slot_diversity_loss": float(out.slot_diversity_loss.item()),
                        "slot_balance_loss": float(out.slot_balance_loss.item()),
                        "attractor_distance": attractor_distance,
                        "entity_names": metadata.get("entity_names"),
                        "entity_count": metadata.get("entity_count"),
                        "query_name": metadata.get("query_name"),
                        "query_name_initial": metadata.get("query_name_initial"),
                        "query_entity_index": metadata.get("query_entity_index"),
                        "query_entity_bucket": metadata.get("query_entity_bucket"),
                        "query_field": metadata.get("query_field"),
                        "query_field_index": metadata.get("query_field_index"),
                        "answer_value": metadata.get("answer_value"),
                        "answer_first_token": metadata.get("answer_first_token"),
                        "field_order_mode": metadata.get("field_order_mode"),
                        "remember_position_mode": metadata.get("remember_position_mode"),
                        "remember_token_index": metadata.get("remember_token_index"),
                        "fact_tokens_start": metadata.get("fact_tokens_start"),
                        "fact_tokens_end": metadata.get("fact_tokens_end"),
                        "context_token_index": metadata.get("context_token_index"),
                        "prefix_noise_length": metadata.get("prefix_noise_length"),
                        "suffix_noise_length": metadata.get("suffix_noise_length"),
                        "question_token_index": metadata.get("question_token_index"),
                        "query_name_token_index": metadata.get("query_name_token_index"),
                        "query_field_token_index": metadata.get("query_field_token_index"),
                        "answer_token_start": metadata.get("answer_token_start"),
                        "answer_token_end": metadata.get("answer_token_end"),
                        "query_entity_token_start": metadata.get("query_entity_token_start"),
                        "query_entity_token_end": metadata.get("query_entity_token_end"),
                        "query_fact_token_start": metadata.get("query_fact_token_start"),
                        "query_fact_token_end": metadata.get("query_fact_token_end"),
                        "query_fact_position_ratio": metadata.get("query_fact_position_ratio"),
                        "query_fact_position_bucket": metadata.get("query_fact_position_bucket"),
                        "entity_spans": metadata.get("entity_spans"),
                        "fact_spans": metadata.get("fact_spans"),
                        "field_orders_by_entity": metadata.get("field_orders_by_entity"),
                        "facts_by_entity": metadata.get("facts_by_entity"),
                    }
                )
        dataset_offset += x.size(0)

    model.train()
    return (
        RecallMetrics(
            loss=total_loss / max(batches, 1),
            ce_loss=total_ce / max(batches, 1),
            answer_token_accuracy=token_correct / max(token_total, 1),
            answer_exact_accuracy=exact_correct / max(example_total, 1),
            structural_energy=total_energy / max(batches, 1),
            drift_loss=total_drift / max(batches, 1),
            core_norm_mean=total_core_norm / max(batches, 1),
            core_slot_norm_std=total_core_slot_std / max(batches, 1),
            attractor_distance=total_attractor_distance / max(batches, 1),
            slot_diversity_loss=total_slot_diversity / max(batches, 1),
            slot_balance_loss=total_slot_balance / max(batches, 1),
        ),
        traces,
    )


def maybe_write_core_trace(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    task_spec: TaskSpec,
    delay_tokens: int,
    variant: str,
    seed: int,
    trace_entries: list[dict[str, Any]],
) -> Optional[Path]:
    if not args.save_core_traces:
        return None
    trace_dir = output_dir / "core_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"delay{delay_tokens}_{variant}_seed{seed}.json"
    payload = {
        "task_type": task_spec.task_type,
        "delay_tokens": delay_tokens,
        "variant": variant,
        "seed": seed,
        "entities_per_sample": task_spec.entities_per_sample,
        "fields": task_spec.fields,
        "value_length": task_spec.value_length,
        "top1_routing": bool(args.top1_routing),
        "field_order_mode": args.field_order_mode,
        "remember_position_mode": args.remember_position_mode,
        "trace_entries": trace_entries,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def train_variant(
    *,
    args: argparse.Namespace,
    variant: str,
    delay_tokens: int,
    tokenizer: SyntheticTokenizer,
    task_spec: TaskSpec,
) -> RecallResult:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()

    train_dataset = DelayedRecallDataset(
        num_samples=args.train_samples,
        delay_tokens=delay_tokens,
        noise_vocab_size=args.noise_vocab_size,
        seed=args.seed,
        tokenizer=tokenizer,
        task_spec=task_spec,
        field_order_mode=args.field_order_mode,
        remember_position_mode=args.remember_position_mode,
    )
    val_dataset = DelayedRecallDataset(
        num_samples=args.val_samples,
        delay_tokens=delay_tokens,
        noise_vocab_size=args.noise_vocab_size,
        seed=args.seed + 10_000,
        tokenizer=tokenizer,
        task_spec=task_spec,
        field_order_mode=args.field_order_mode,
        remember_position_mode=args.remember_position_mode,
    )
    effective_batch_size = resolve_effective_batch_size(args, train_dataset.sequence_length)
    train_loader = build_loader(train_dataset, effective_batch_size, shuffle=True, seed=args.seed)
    val_loader = build_loader(val_dataset, effective_batch_size, shuffle=False, seed=args.seed)

    config = resolve_config(args, variant=variant, vocab_size=tokenizer.vocab_size, seq_len=train_dataset.sequence_length)
    model = SVFTransformer(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"\n=== Delayed recall variant={variant} task={task_spec.task_type} delay={delay_tokens} "
        f"seq_len={train_dataset.sequence_length} batch_size={effective_batch_size} "
        f"params={count_parameters(model)} ==="
    )
    step = 0
    final_train_loss = 0.0
    model.train()

    while step < args.steps:
        for x, y, _ in train_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            model.reset_state()
            out = model(x, targets=y, core_state=None, use_memory=config.use_memory, write_memory=config.use_memory)
            assert out.loss is not None

            optimizer.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            final_train_loss = float(out.loss.item())
            if step % args.log_interval == 0:
                print(f"step={step:05d} train_loss={final_train_loss:.4f}")

            step += 1
            if args.eval_interval > 0 and step % args.eval_interval == 0:
                metrics, _ = evaluate_variant(model, val_loader, device=args.device)
                print(
                    f"eval step={step:05d} val_ce={metrics.ce_loss:.4f} "
                    f"answer_token_acc={metrics.answer_token_accuracy:.4f} "
                    f"answer_exact_acc={metrics.answer_exact_accuracy:.4f} "
                    f"core_norm={metrics.core_norm_mean:.4f}"
                )
            if step >= args.steps:
                break

    final_metrics, trace_entries = evaluate_variant(
        model,
        val_loader,
        device=args.device,
        collect_traces=args.save_core_traces,
        max_trace_batches=args.trace_batches,
        max_trace_examples=args.trace_examples,
    )
    trace_path = maybe_write_core_trace(
        Path(args.output_dir),
        args=args,
        task_spec=task_spec,
        delay_tokens=delay_tokens,
        variant=variant,
        seed=args.seed,
        trace_entries=trace_entries,
    )
    return RecallResult(
        task_type=task_spec.task_type,
        delay_tokens=delay_tokens,
        variant=variant,
        seed=args.seed,
        steps=step,
        parameter_count=count_parameters(model),
        final_train_loss=final_train_loss,
        final_val_loss=final_metrics.loss,
        final_val_ce_loss=final_metrics.ce_loss,
        final_answer_token_accuracy=final_metrics.answer_token_accuracy,
        final_answer_exact_accuracy=final_metrics.answer_exact_accuracy,
        final_val_structural_energy=final_metrics.structural_energy,
        final_val_drift_loss=final_metrics.drift_loss,
        final_core_norm_mean=final_metrics.core_norm_mean,
        final_core_slot_norm_std=final_metrics.core_slot_norm_std,
        final_attractor_distance=final_metrics.attractor_distance,
        final_slot_diversity_loss=final_metrics.slot_diversity_loss,
        final_slot_balance_loss=final_metrics.slot_balance_loss,
        core_trace_path=str(trace_path) if trace_path is not None else None,
    )


def write_summary(output_dir: Path, results: list[RecallResult]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"recall_summary_{timestamp}.json"
    csv_path = output_dir / f"recall_summary_{timestamp}.csv"
    json_path.write_text(json.dumps([asdict(item) for item in results], indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return json_path, csv_path


def aggregate_results(results: list[RecallResult]) -> list[RecallAggregate]:
    grouped: dict[tuple[str, int, str], list[RecallResult]] = {}
    for result in results:
        grouped.setdefault((result.task_type, result.delay_tokens, result.variant), []).append(result)

    aggregates: list[RecallAggregate] = []
    for (task_type, delay_tokens, variant), items in grouped.items():
        def mean_of(field: str) -> float:
            return statistics.fmean(float(getattr(item, field)) for item in items)

        def std_of(field: str) -> float:
            values = [float(getattr(item, field)) for item in items]
            return statistics.pstdev(values) if len(values) > 1 else 0.0

        aggregates.append(
            RecallAggregate(
                task_type=task_type,
                delay_tokens=delay_tokens,
                variant=variant,
                runs=len(items),
                mean_val_ce_loss=mean_of("final_val_ce_loss"),
                std_val_ce_loss=std_of("final_val_ce_loss"),
                mean_answer_token_accuracy=mean_of("final_answer_token_accuracy"),
                std_answer_token_accuracy=std_of("final_answer_token_accuracy"),
                mean_answer_exact_accuracy=mean_of("final_answer_exact_accuracy"),
                std_answer_exact_accuracy=std_of("final_answer_exact_accuracy"),
                mean_val_structural_energy=mean_of("final_val_structural_energy"),
                mean_val_drift_loss=mean_of("final_val_drift_loss"),
                mean_core_norm_mean=mean_of("final_core_norm_mean"),
                mean_core_slot_norm_std=mean_of("final_core_slot_norm_std"),
                mean_attractor_distance=mean_of("final_attractor_distance"),
                mean_slot_diversity_loss=mean_of("final_slot_diversity_loss"),
                mean_slot_balance_loss=mean_of("final_slot_balance_loss"),
                mean_parameter_count=mean_of("parameter_count"),
            )
        )
    return sorted(
        aggregates,
        key=lambda item: (item.delay_tokens, -item.mean_answer_exact_accuracy, item.mean_val_ce_loss),
    )


def write_aggregate_summary(output_dir: Path, results: list[RecallAggregate]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"recall_aggregate_{timestamp}.json"
    csv_path = output_dir / f"recall_aggregate_{timestamp}.csv"
    json_path.write_text(json.dumps([asdict(item) for item in results], indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return json_path, csv_path


def write_manifest(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    tokenizer: SyntheticTokenizer,
    task_spec: TaskSpec,
    results: list[RecallResult],
    summary_json: Path,
    summary_csv: Path,
    aggregate_json: Path,
    aggregate_csv: Path,
) -> tuple[Path, Path]:
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": "delayed_recall",
        "task_type": task_spec.task_type,
        "entities_per_sample": task_spec.entities_per_sample,
        "fields": task_spec.fields,
        "value_length": task_spec.value_length,
        "variants": args.variants,
        "delays": args.delays,
        "seeds": args.seeds,
        "args": vars(args),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "artifacts": {
            "summary_json": str(summary_json),
            "summary_csv": str(summary_csv),
            "aggregate_json": str(aggregate_json),
            "aggregate_csv": str(aggregate_csv),
        },
        "results": [asdict(item) for item in results],
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"recall_manifest_{timestamp}.json"
    md_path = output_dir / f"recall_manifest_{timestamp}.md"
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [
        "# Delayed Recall Manifest",
        "",
        f"- task_type: `{task_spec.task_type}`",
        f"- entities_per_sample: `{task_spec.entities_per_sample}`",
        f"- fields: `{', '.join(task_spec.fields)}`",
        f"- value_length: `{task_spec.value_length}`",
        f"- variants: `{', '.join(args.variants)}`",
        f"- delays: `{', '.join(str(item) for item in args.delays)}`",
        f"- seeds: `{', '.join(str(item) for item in args.seeds)}`",
        f"- save_core_traces: `{args.save_core_traces}`",
        f"- vocab_size: `{tokenizer.vocab_size}`",
        f"- summary_json: `{summary_json}`",
        f"- aggregate_json: `{aggregate_json}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SVF variants on delayed recall tasks.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "persistent_core"],
        choices=[
            "baseline",
            "memory",
            "persistent_core",
            "core_dynamics",
            "specialized_core",
            "specialized_core_dynamics",
            "memory_core",
            "svf",
            "specialized_svf",
        ],
    )
    parser.add_argument("--task-type", type=str, default="single_entity", choices=["single_entity", "multi_entity"])
    parser.add_argument("--delays", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--adaptive-batch-size", action="store_true", default=True)
    parser.add_argument("--no-adaptive-batch-size", dest="adaptive_batch_size", action="store_false")
    parser.add_argument("--adaptive-batch-reference-seq-len", type=int, default=1024)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--value-length", dest="value_length", type=int, default=5)
    parser.add_argument("--code-length", dest="value_length", type=int)
    parser.add_argument("--entities-per-sample", type=int, default=3)
    parser.add_argument("--fields", nargs="+", default=["age", "city", "color"])
    parser.add_argument("--noise-vocab-size", type=int, default=64)
    parser.add_argument("--field-order-mode", choices=["fixed", "shuffled"], default="fixed")
    parser.add_argument("--remember-position-mode", choices=["front", "middle", "back", "random"], default="front")
    parser.add_argument("--save-core-traces", action="store_true")
    parser.add_argument("--trace-batches", type=int, default=2)
    parser.add_argument("--trace-examples", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--core-slots", type=int, default=4)
    parser.add_argument("--memory-size", type=int, default=256)
    parser.add_argument("--attractor-strength", type=float, default=0.05)
    parser.add_argument("--drift-scale", type=float, default=0.1)
    parser.add_argument("--conservation-weight", type=float, default=0.01)
    parser.add_argument("--drift-weight", type=float, default=0.001)
    parser.add_argument("--slot-balance-loss", action="store_true")
    parser.add_argument("--slot-balance-weight", type=float, default=0.01)
    parser.add_argument("--top1-routing", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--match-baseline-to",
        type=str,
        default="persistent_core",
        choices=["memory", "persistent_core", "memory_core", "svf"],
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="outputs/experiments/phaseD_delayed_recall")
    args = parser.parse_args()
    unknown_fields = sorted(set(args.fields) - set(FIELD_TOKENS))
    if args.task_type == "multi_entity" and unknown_fields:
        parser.error(f"Unsupported fields: {', '.join(unknown_fields)}")
    return args


def main() -> None:
    args = parse_args()
    task_spec = build_task_spec(args)
    tokenizer = SyntheticTokenizer(noise_vocab_size=args.noise_vocab_size, fields=task_spec.fields)
    results: list[RecallResult] = []

    for delay_tokens in args.delays:
        print(f"\n##### Delay {delay_tokens} #####")
        for seed in args.seeds:
            args.seed = seed
            print(f"\n### Seed {seed} ###")
            for variant in args.variants:
                results.append(
                    train_variant(
                        args=args,
                        variant=variant,
                        delay_tokens=delay_tokens,
                        tokenizer=tokenizer,
                        task_spec=task_spec,
                    )
                )

    output_dir = Path(args.output_dir)
    summary_json, summary_csv = write_summary(output_dir, results)
    aggregate = aggregate_results(results)
    aggregate_json, aggregate_csv = write_aggregate_summary(output_dir, aggregate)
    manifest_json, manifest_md = write_manifest(
        output_dir,
        args=args,
        tokenizer=tokenizer,
        task_spec=task_spec,
        results=results,
        summary_json=summary_json,
        summary_csv=summary_csv,
        aggregate_json=aggregate_json,
        aggregate_csv=aggregate_csv,
    )

    print("\n=== Delayed Recall Aggregate ===")
    for item in aggregate:
        print(
            f"delay={item.delay_tokens:<5d} "
            f"variant={item.variant:>16} "
            f"exact_acc={item.mean_answer_exact_accuracy:.4f} "
            f"token_acc={item.mean_answer_token_accuracy:.4f} "
            f"core_norm={item.mean_core_norm_mean:.4f} "
            f"val_ce={item.mean_val_ce_loss:.4f}"
        )
    print(f"wrote summary to {summary_json}")
    print(f"wrote aggregate to {aggregate_json}")
    print(f"wrote manifest to {manifest_json} and {manifest_md}")


if __name__ == "__main__":
    main()
