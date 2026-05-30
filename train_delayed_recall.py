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

import torch
from torch.utils.data import DataLoader, Dataset

from models import SVFTransformer, SVFTransformerConfig, build_config_for_variant
from train_experiment import auto_match_variant_config, build_base_config, count_parameters


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]
CONTROL_TOKENS = ["remember", "code", "context", "question", "answer"]
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


class SyntheticTokenizer:
    def __init__(self, noise_vocab_size: int) -> None:
        tokens = SPECIAL_TOKENS + CONTROL_TOKENS + DIGIT_TOKENS + NAME_TOKENS
        tokens += [f"noise_{index}" for index in range(noise_vocab_size)]
        self.stoi = {token: index for index, token in enumerate(tokens)}
        self.itos = {index: token for token, index in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi[token] for token in tokens]


class DelayedRecallDataset(Dataset):
    def __init__(
        self,
        *,
        num_samples: int,
        delay_tokens: int,
        code_length: int,
        noise_vocab_size: int,
        seed: int,
        tokenizer: SyntheticTokenizer,
    ) -> None:
        self.answer_length = code_length
        self.samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        rng = random.Random(seed)
        noise_tokens = [f"noise_{index}" for index in range(noise_vocab_size)]

        for _ in range(num_samples):
            name = rng.choice(NAME_TOKENS)
            code = [rng.choice(DIGIT_TOKENS) for _ in range(code_length)]
            filler = [rng.choice(noise_tokens) for _ in range(delay_tokens)]
            tokens = [
                "<bos>",
                "remember",
                name,
                "code",
                *code,
                "context",
                *filler,
                "question",
                name,
                "code",
                "answer",
                *code,
                "<eos>",
            ]
            answer_start = len(tokens) - code_length - 1
            answer_positions = list(range(answer_start - 1, answer_start - 1 + code_length))
            ids = tokenizer.encode(tokens)
            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)
            answer_mask = torch.zeros(len(y), dtype=torch.bool)
            for position in answer_positions:
                answer_mask[position] = True
            self.samples.append((x, y, answer_mask))

        self.sequence_length = len(self.samples[0][0])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[index]


@dataclass
class RecallMetrics:
    loss: float
    ce_loss: float
    answer_token_accuracy: float
    answer_exact_accuracy: float


@dataclass
class RecallResult:
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


@dataclass
class RecallAggregate:
    delay_tokens: int
    variant: str
    runs: int
    mean_val_ce_loss: float
    std_val_ce_loss: float
    mean_answer_token_accuracy: float
    std_answer_token_accuracy: float
    mean_answer_exact_accuracy: float
    std_answer_exact_accuracy: float
    mean_parameter_count: float


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


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
        return auto_match_variant_config(local_args, variant, vocab_size, target_params)
    return config


@torch.no_grad()
def evaluate_variant(
    model: SVFTransformer,
    loader: DataLoader,
    *,
    device: str,
) -> RecallMetrics:
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    batches = 0
    token_correct = 0
    token_total = 0
    exact_correct = 0
    example_total = 0

    for x, y, answer_mask in loader:
        x = x.to(device)
        y = y.to(device)
        answer_mask = answer_mask.to(device)
        model.reset_state()
        out = model(x, targets=y, core_state=None, use_memory=model.config.use_memory, write_memory=False)
        assert out.loss is not None and out.ce_loss is not None
        total_loss += float(out.loss.item())
        total_ce += float(out.ce_loss.item())
        batches += 1

        predictions = out.logits.argmax(dim=-1)
        token_correct += int(((predictions == y) & answer_mask).sum().item())
        token_total += int(answer_mask.sum().item())
        exact_masked = (predictions == y) | (~answer_mask)
        exact_correct += int(exact_masked.all(dim=1).sum().item())
        example_total += int(x.size(0))

    model.train()
    return RecallMetrics(
        loss=total_loss / max(batches, 1),
        ce_loss=total_ce / max(batches, 1),
        answer_token_accuracy=token_correct / max(token_total, 1),
        answer_exact_accuracy=exact_correct / max(example_total, 1),
    )


def train_variant(
    *,
    args: argparse.Namespace,
    variant: str,
    delay_tokens: int,
    tokenizer: SyntheticTokenizer,
) -> RecallResult:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_dataset = DelayedRecallDataset(
        num_samples=args.train_samples,
        delay_tokens=delay_tokens,
        code_length=args.code_length,
        noise_vocab_size=args.noise_vocab_size,
        seed=args.seed,
        tokenizer=tokenizer,
    )
    val_dataset = DelayedRecallDataset(
        num_samples=args.val_samples,
        delay_tokens=delay_tokens,
        code_length=args.code_length,
        noise_vocab_size=args.noise_vocab_size,
        seed=args.seed + 10_000,
        tokenizer=tokenizer,
    )
    train_loader = build_loader(train_dataset, args.batch_size, shuffle=True, seed=args.seed)
    val_loader = build_loader(val_dataset, args.batch_size, shuffle=False, seed=args.seed)

    config = resolve_config(args, variant=variant, vocab_size=tokenizer.vocab_size, seq_len=train_dataset.sequence_length)
    model = SVFTransformer(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"\n=== Delayed recall variant={variant} delay={delay_tokens} "
        f"seq_len={train_dataset.sequence_length} params={count_parameters(model)} ==="
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
                metrics = evaluate_variant(model, val_loader, device=args.device)
                print(
                    f"eval step={step:05d} val_ce={metrics.ce_loss:.4f} "
                    f"answer_token_acc={metrics.answer_token_accuracy:.4f} "
                    f"answer_exact_acc={metrics.answer_exact_accuracy:.4f}"
                )
            if step >= args.steps:
                break

    final_metrics = evaluate_variant(model, val_loader, device=args.device)
    return RecallResult(
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
    grouped: dict[tuple[int, str], list[RecallResult]] = {}
    for result in results:
        grouped.setdefault((result.delay_tokens, result.variant), []).append(result)

    aggregates: list[RecallAggregate] = []
    for (delay_tokens, variant), items in grouped.items():
        def mean_of(field: str) -> float:
            return statistics.fmean(float(getattr(item, field)) for item in items)

        def std_of(field: str) -> float:
            values = [float(getattr(item, field)) for item in items]
            return statistics.pstdev(values) if len(values) > 1 else 0.0

        aggregates.append(
            RecallAggregate(
                delay_tokens=delay_tokens,
                variant=variant,
                runs=len(items),
                mean_val_ce_loss=mean_of("final_val_ce_loss"),
                std_val_ce_loss=std_of("final_val_ce_loss"),
                mean_answer_token_accuracy=mean_of("final_answer_token_accuracy"),
                std_answer_token_accuracy=std_of("final_answer_token_accuracy"),
                mean_answer_exact_accuracy=mean_of("final_answer_exact_accuracy"),
                std_answer_exact_accuracy=std_of("final_answer_exact_accuracy"),
                mean_parameter_count=mean_of("parameter_count"),
            )
        )
    return sorted(aggregates, key=lambda item: (item.delay_tokens, -item.mean_answer_exact_accuracy, item.mean_val_ce_loss))


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
    results: list[RecallResult],
    summary_json: Path,
    summary_csv: Path,
    aggregate_json: Path,
    aggregate_csv: Path,
) -> tuple[Path, Path]:
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": "delayed_recall",
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
        f"- variants: `{', '.join(args.variants)}`",
        f"- delays: `{', '.join(str(item) for item in args.delays)}`",
        f"- seeds: `{', '.join(str(item) for item in args.seeds)}`",
        f"- vocab_size: `{tokenizer.vocab_size}`",
        f"- summary_json: `{summary_json}`",
        f"- aggregate_json: `{aggregate_json}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SVF variants on a delayed recall task.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "persistent_core"],
        choices=["baseline", "memory", "persistent_core", "core_dynamics", "memory_core", "svf"],
    )
    parser.add_argument("--delays", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--code-length", type=int, default=5)
    parser.add_argument("--noise-vocab-size", type=int, default=64)
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
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--match-baseline-to", type=str, default="persistent_core", choices=["memory", "persistent_core", "memory_core", "svf"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="outputs/experiments/phaseD_delayed_recall")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = SyntheticTokenizer(noise_vocab_size=args.noise_vocab_size)
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
        results=results,
        summary_json=summary_json,
        summary_csv=summary_csv,
        aggregate_json=aggregate_json,
        aggregate_csv=aggregate_csv,
    )

    print("\n=== Delayed Recall Aggregate ===")
    for item in aggregate:
        print(
            f"delay={item.delay_tokens:<4d} "
            f"variant={item.variant:>16} "
            f"exact_acc={item.mean_answer_exact_accuracy:.4f} "
            f"token_acc={item.mean_answer_token_accuracy:.4f} "
            f"val_ce={item.mean_val_ce_loss:.4f}"
        )
    print(f"wrote summary to {summary_json}")
    print(f"wrote aggregate to {aggregate_json}")
    print(f"wrote manifest to {manifest_json} and {manifest_md}")


if __name__ == "__main__":
    main()
