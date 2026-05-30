from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader, Dataset

from models import SVFTransformer, SVFTransformerConfig, build_config_for_variant
from train import DEFAULT_TEXT


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
VARIANT_SUITES = {
    "phase1": ["baseline", "svf"],
    "phase2": ["baseline", "memory", "persistent_core", "memory_core", "svf"],
}


class SequenceDataset(Dataset):
    def __init__(self, ids: list[int], seq_len: int) -> None:
        if len(ids) < seq_len + 2:
            repeats = (seq_len + 2) // max(len(ids), 1) + 1
            ids = ids * repeats
        self.seq_len = seq_len
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return max(1, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


class CharTokenizer:
    def __init__(self, text: str) -> None:
        chars = ["<unk>"] + sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        unk_id = self.stoi["<unk>"]
        return [self.stoi.get(ch, unk_id) for ch in text]

    def decode(self, ids: list[int]) -> str:
        pieces: list[str] = []
        for token_id in ids:
            token = self.itos[int(token_id)]
            pieces.append("" if token == "<unk>" else token)
        return "".join(pieces)

    def save_state(self) -> dict[str, Any]:
        return {"type": "char", "stoi": self.stoi, "itos": self.itos}


class BPETokenizerWrapper:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def save_state(self) -> dict[str, Any]:
        return {"type": "bpe", "tokenizer_json": self.tokenizer.to_str()}


@dataclass
class EvalMetrics:
    loss: float
    ce_loss: float
    conservation_loss: float
    drift_loss: float
    structural_energy: float


@dataclass
class ExperimentResult:
    variant: str
    seed: int
    steps: int
    tokenizer: str
    train_tokens: int
    val_tokens: int
    parameter_count: int
    final_train_loss: float
    final_train_ce_loss: float
    final_val_loss: float
    final_val_ce_loss: float
    final_val_perplexity: float
    final_val_conservation_loss: float
    final_val_drift_loss: float
    final_val_structural_energy: float
    checkpoint_path: Optional[str]
    best_checkpoint_path: Optional[str]


@dataclass
class AggregateResult:
    variant: str
    runs: int
    mean_val_loss: float
    std_val_loss: float
    mean_val_ce_loss: float
    std_val_ce_loss: float
    mean_val_perplexity: float
    std_val_perplexity: float
    mean_train_loss: float
    mean_train_ce_loss: float
    mean_val_conservation_loss: float
    mean_val_drift_loss: float
    mean_val_structural_energy: float
    mean_parameter_count: float
    best_seed: int
    best_val_ce_loss: float


def load_text(path: Optional[str]) -> str:
    if path is None:
        return DEFAULT_TEXT
    return Path(path).read_text(encoding="utf-8")


def split_text(text: str, val_fraction: float) -> tuple[str, str]:
    if len(text) < 8:
        raise ValueError("Corpus is too small for train/val split.")
    if not 0 < val_fraction < 0.5:
        raise ValueError("--val-fraction must be greater than 0 and less than 0.5")
    split = max(4, int(len(text) * (1.0 - val_fraction)))
    split = min(split, len(text) - 4)
    return text[:split], text[split:]


def train_bpe_tokenizer(text: str, vocab_size: int, min_frequency: int) -> BPETokenizerWrapper:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator([text], trainer=trainer)
    return BPETokenizerWrapper(tokenizer)


def build_tokenizer(
    text: str,
    tokenizer_name: str,
    vocab_size: int,
    min_frequency: int,
) -> CharTokenizer | BPETokenizerWrapper:
    if tokenizer_name == "char":
        return CharTokenizer(text)
    if tokenizer_name == "bpe":
        return train_bpe_tokenizer(text, vocab_size=vocab_size, min_frequency=min_frequency)
    raise ValueError("Unsupported tokenizer. Expected one of: char, bpe.")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_loader(
    dataset: SequenceDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=generator,
    )


def maybe_detach_core_state(
    core_state: torch.Tensor,
    use_persistent_core: bool,
) -> Optional[torch.Tensor]:
    if not use_persistent_core:
        return None
    return core_state.detach()


@torch.no_grad()
def evaluate_variant(
    model: SVFTransformer,
    loader: DataLoader,
    device: str,
    max_batches: int,
) -> EvalMetrics:
    model.eval()
    model.reset_state()
    losses: list[float] = []
    ce_losses: list[float] = []
    conservation_losses: list[float] = []
    drift_losses: list[float] = []
    structural_energies: list[float] = []
    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        core_state = None
        model.reset_state()

        out = model(
            x,
            targets=y,
            core_state=core_state,
            use_memory=model.config.use_memory,
            write_memory=False,
        )

        if out.loss is not None:
            losses.append(float(out.loss.item()))
        if out.ce_loss is not None:
            ce_losses.append(float(out.ce_loss.item()))
        conservation_losses.append(float(out.conservation_loss.item()))
        drift_losses.append(float(out.drift_loss.item()))
        structural_energies.append(float(out.structural_energy.item()))

    model.train()
    return EvalMetrics(
        loss=sum(losses) / max(len(losses), 1),
        ce_loss=sum(ce_losses) / max(len(ce_losses), 1),
        conservation_loss=sum(conservation_losses) / max(len(conservation_losses), 1),
        drift_loss=sum(drift_losses) / max(len(drift_losses), 1),
        structural_energy=sum(structural_energies) / max(len(structural_energies), 1),
    )


def build_base_config(
    args: argparse.Namespace,
    vocab_size: int,
    *,
    d_model: Optional[int] = None,
    d_ff: Optional[int] = None,
) -> SVFTransformerConfig:
    return SVFTransformerConfig(
        vocab_size=vocab_size,
        d_model=d_model if d_model is not None else args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        d_ff=d_ff if d_ff is not None else args.d_ff,
        dropout=args.dropout,
        max_seq_len=args.seq_len,
        core_slots=args.core_slots,
        memory_size=args.memory_size,
        attractor_strength=args.attractor_strength,
        drift_scale=args.drift_scale,
        conservation_weight=args.conservation_weight,
        drift_weight=args.drift_weight,
    )


def auto_match_variant_config(
    args: argparse.Namespace,
    variant: str,
    vocab_size: int,
    target_params: int,
) -> SVFTransformerConfig:
    ff_ratio = max(args.d_ff / max(args.d_model, 1), 1.0)
    best_config: Optional[SVFTransformerConfig] = None
    best_gap: Optional[int] = None
    max_d_model = max(args.d_model * 2, args.heads * 8)
    candidate_values: list[int] = []
    while max_d_model <= 4096:
        candidate_values = list(range(args.heads, max_d_model + 1, args.heads))
        exceeded_target = False
        for candidate_d_model in candidate_values:
            candidate_d_ff = max(args.heads, int(round(candidate_d_model * ff_ratio)))
            candidate_base = build_base_config(
                args,
                vocab_size,
                d_model=candidate_d_model,
                d_ff=candidate_d_ff,
            )
            candidate_config = build_config_for_variant(candidate_base, variant)
            candidate_params = count_parameters(SVFTransformer(candidate_config))
            gap = abs(candidate_params - target_params)
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_config = candidate_config
            if candidate_params >= target_params:
                exceeded_target = True
                break
        if exceeded_target:
            break
        max_d_model *= 2

    if best_config is not None:
        return best_config

    for candidate_d_model in range(args.heads, 4097, args.heads):
        candidate_d_ff = max(args.heads, int(round(candidate_d_model * ff_ratio)))
        candidate_base = build_base_config(
            args,
            vocab_size,
            d_model=candidate_d_model,
            d_ff=candidate_d_ff,
        )
        candidate_config = build_config_for_variant(candidate_base, variant)
        candidate_params = count_parameters(SVFTransformer(candidate_config))
        gap = abs(candidate_params - target_params)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_config = candidate_config

    assert best_config is not None
    return best_config


def resolve_config_for_variant(
    args: argparse.Namespace,
    variant: str,
    vocab_size: int,
) -> SVFTransformerConfig:
    base_config = build_base_config(args, vocab_size)
    config = build_config_for_variant(base_config, variant)
    if variant == "baseline" and args.match_baseline_to is not None:
        reference_base = build_base_config(args, vocab_size)
        reference_config = build_config_for_variant(reference_base, args.match_baseline_to)
        target_params = count_parameters(SVFTransformer(reference_config))
        matched_config = auto_match_variant_config(args, variant, vocab_size, target_params)
        matched_params = count_parameters(SVFTransformer(matched_config))
        print(
            "matched baseline params "
            f"target_variant={args.match_baseline_to} "
            f"target_params={target_params} "
            f"matched_params={matched_params} "
            f"d_model={matched_config.d_model} "
            f"d_ff={matched_config.d_ff}"
        )
        return matched_config
    return config


def save_checkpoint(
    path: Path,
    model: SVFTransformer,
    config: SVFTransformerConfig,
    tokenizer_state: dict[str, Any],
    variant: str,
    step: int,
    metrics: EvalMetrics,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.__dict__,
            "variant": variant,
            "step": step,
            "metrics": asdict(metrics),
            "tokenizer": tokenizer_state,
        },
        path,
    )


def train_variant(
    *,
    variant: str,
    args: argparse.Namespace,
    train_dataset: SequenceDataset,
    val_dataset: SequenceDataset,
    tokenizer_state: dict[str, Any],
    vocab_size: int,
) -> ExperimentResult:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_loader = build_loader(train_dataset, args.batch_size, shuffle=True, seed=args.seed)
    val_loader = build_loader(val_dataset, args.batch_size, shuffle=False, seed=args.seed)

    config = resolve_config_for_variant(args, variant, vocab_size)
    model = SVFTransformer(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"\n=== Training variant={variant} "
        f"memory={config.use_memory} "
        f"core={config.use_persistent_core} "
        f"dynamics={config.use_structural_dynamics} "
        f"struct_loss={config.use_structural_loss} ==="
    )
    print(f"parameter_count={count_parameters(model)}")

    step = 0
    best_val_loss = float("inf")
    best_checkpoint_path: Optional[Path] = None
    final_train_loss = 0.0
    final_train_ce_loss = 0.0
    core_state = None
    model.train()
    model.reset_state()

    while step < args.steps:
        for x, y in train_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            if core_state is not None and core_state.size(0) != x.size(0):
                core_state = None
            if not args.carry_state_across_batches:
                core_state = None
                model.reset_state()

            out = model(
                x,
                targets=y,
                core_state=core_state,
                use_memory=config.use_memory,
                write_memory=config.use_memory,
            )
            core_state = maybe_detach_core_state(out.core_state, config.use_persistent_core)
            assert out.loss is not None

            optimizer.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            final_train_loss = float(out.loss.item())
            final_train_ce_loss = float(out.ce_loss.item()) if out.ce_loss is not None else final_train_loss

            if step % args.log_interval == 0:
                print(
                    f"step={step:05d} "
                    f"train_loss={final_train_loss:.4f} "
                    f"train_ce={final_train_ce_loss:.4f} "
                    f"energy={out.structural_energy.item():.6f} "
                    f"conservation={out.conservation_loss.item():.6f} "
                    f"drift={out.drift_loss.item():.6f}"
                )

            step += 1
            if args.eval_interval > 0 and step % args.eval_interval == 0:
                metrics = evaluate_variant(model, val_loader, args.device, args.eval_batches)
                print(
                    f"eval step={step:05d} "
                    f"val_loss={metrics.loss:.4f} "
                    f"val_ce={metrics.ce_loss:.4f} "
                    f"ppl={math.exp(min(metrics.ce_loss, 20.0)):.4f}"
                )
                if args.save_checkpoints and metrics.loss < best_val_loss:
                    best_val_loss = metrics.loss
                    best_checkpoint_path = (
                        Path(args.output_dir) / "checkpoints" / f"{variant}_seed{args.seed}_best.pt"
                    )
                    save_checkpoint(
                        best_checkpoint_path,
                        model,
                        config,
                        tokenizer_state,
                        variant,
                        step,
                        metrics,
                    )
                    print(f"saved best checkpoint to {best_checkpoint_path}")
                model.reset_state()
                core_state = None

            if step >= args.steps:
                break

    final_metrics = evaluate_variant(model, val_loader, args.device, args.eval_batches)
    checkpoint_path: Optional[Path] = None
    if args.save_checkpoints:
        checkpoint_path = Path(args.output_dir) / "checkpoints" / f"{variant}_seed{args.seed}_final.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            config,
            tokenizer_state,
            variant,
            step,
            final_metrics,
        )
        print(f"saved final checkpoint to {checkpoint_path}")

    return ExperimentResult(
        variant=variant,
        seed=args.seed,
        steps=step,
        tokenizer=args.tokenizer,
        train_tokens=len(train_dataset.data),
        val_tokens=len(val_dataset.data),
        parameter_count=count_parameters(model),
        final_train_loss=final_train_loss,
        final_train_ce_loss=final_train_ce_loss,
        final_val_loss=final_metrics.loss,
        final_val_ce_loss=final_metrics.ce_loss,
        final_val_perplexity=math.exp(min(final_metrics.ce_loss, 20.0)),
        final_val_conservation_loss=final_metrics.conservation_loss,
        final_val_drift_loss=final_metrics.drift_loss,
        final_val_structural_energy=final_metrics.structural_energy,
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        best_checkpoint_path=str(best_checkpoint_path) if best_checkpoint_path is not None else None,
    )


def resolve_variants(args: argparse.Namespace) -> list[str]:
    if args.variant:
        return [args.variant]
    if args.suite:
        return VARIANT_SUITES[args.suite]
    raise ValueError("Please provide either --variant or --suite.")


def write_summary(output_dir: Path, results: list[ExperimentResult]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"experiment_summary_{timestamp}.json"
    csv_path = output_dir / f"experiment_summary_{timestamp}.csv"

    json_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    return json_path, csv_path


def aggregate_results(results: list[ExperimentResult]) -> list[AggregateResult]:
    by_variant: dict[str, list[ExperimentResult]] = {}
    for result in results:
        by_variant.setdefault(result.variant, []).append(result)

    aggregates: list[AggregateResult] = []
    for variant, variant_results in by_variant.items():
        best_result = min(variant_results, key=lambda item: item.final_val_ce_loss)

        def mean_of(field: str) -> float:
            values = [float(getattr(item, field)) for item in variant_results]
            return statistics.fmean(values)

        def std_of(field: str) -> float:
            values = [float(getattr(item, field)) for item in variant_results]
            if len(values) < 2:
                return 0.0
            return statistics.pstdev(values)

        aggregates.append(
            AggregateResult(
                variant=variant,
                runs=len(variant_results),
                mean_val_loss=mean_of("final_val_loss"),
                std_val_loss=std_of("final_val_loss"),
                mean_val_ce_loss=mean_of("final_val_ce_loss"),
                std_val_ce_loss=std_of("final_val_ce_loss"),
                mean_val_perplexity=mean_of("final_val_perplexity"),
                std_val_perplexity=std_of("final_val_perplexity"),
                mean_train_loss=mean_of("final_train_loss"),
                mean_train_ce_loss=mean_of("final_train_ce_loss"),
                mean_val_conservation_loss=mean_of("final_val_conservation_loss"),
                mean_val_drift_loss=mean_of("final_val_drift_loss"),
                mean_val_structural_energy=mean_of("final_val_structural_energy"),
                mean_parameter_count=mean_of("parameter_count"),
                best_seed=best_result.seed,
                best_val_ce_loss=best_result.final_val_ce_loss,
            )
        )

    return sorted(aggregates, key=lambda item: item.mean_val_ce_loss)


def write_aggregate_summary(output_dir: Path, results: list[AggregateResult]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"aggregate_summary_{timestamp}.json"
    csv_path = output_dir / f"aggregate_summary_{timestamp}.csv"

    json_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    return json_path, csv_path


def build_run_manifest(
    *,
    args: argparse.Namespace,
    variants: list[str],
    seeds: list[int],
    train_text: str,
    val_text: str,
    train_dataset: SequenceDataset,
    val_dataset: SequenceDataset,
    tokenizer_vocab_size: int,
    results: list[ExperimentResult],
    summary_json: Path,
    summary_csv: Path,
    aggregate_json: Optional[Path],
    aggregate_csv: Optional[Path],
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_path": args.data,
        "output_dir": args.output_dir,
        "variants": variants,
        "seeds": seeds,
        "args": vars(args),
        "protocol": {
            "tokenizer_fit_scope": "train_split_only",
            "validation_batches_independent": True,
            "eval_write_memory": False,
            "carry_state_across_batches": args.carry_state_across_batches,
            "baseline_parameter_matching": args.match_baseline_to,
        },
        "dataset": {
            "tokenizer": args.tokenizer,
            "tokenizer_vocab_size": tokenizer_vocab_size,
            "train_text_chars": len(train_text),
            "val_text_chars": len(val_text),
            "train_tokens": len(train_dataset.data),
            "val_tokens": len(val_dataset.data),
            "val_fraction": args.val_fraction,
        },
        "artifacts": {
            "summary_json": str(summary_json),
            "summary_csv": str(summary_csv),
            "aggregate_summary_json": str(aggregate_json) if aggregate_json is not None else None,
            "aggregate_summary_csv": str(aggregate_csv) if aggregate_csv is not None else None,
        },
        "results": [asdict(result) for result in results],
    }


def write_run_manifest(output_dir: Path, manifest: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"run_manifest_{timestamp}.json"
    md_path = output_dir / f"run_manifest_{timestamp}.md"
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# Experiment Manifest",
        "",
        f"- timestamp: `{manifest['timestamp']}`",
        f"- data_path: `{manifest['data_path']}`",
        f"- output_dir: `{manifest['output_dir']}`",
        f"- variants: `{', '.join(manifest['variants'])}`",
        f"- seeds: `{', '.join(str(seed) for seed in manifest['seeds'])}`",
        "",
        "## Protocol",
        "",
        f"- tokenizer_fit_scope: `{manifest['protocol']['tokenizer_fit_scope']}`",
        f"- validation_batches_independent: `{manifest['protocol']['validation_batches_independent']}`",
        f"- eval_write_memory: `{manifest['protocol']['eval_write_memory']}`",
        f"- carry_state_across_batches: `{manifest['protocol']['carry_state_across_batches']}`",
        f"- baseline_parameter_matching: `{manifest['protocol']['baseline_parameter_matching']}`",
        "",
        "## Dataset",
        "",
        f"- tokenizer: `{manifest['dataset']['tokenizer']}`",
        f"- tokenizer_vocab_size: `{manifest['dataset']['tokenizer_vocab_size']}`",
        f"- train_text_chars: `{manifest['dataset']['train_text_chars']}`",
        f"- val_text_chars: `{manifest['dataset']['val_text_chars']}`",
        f"- train_tokens: `{manifest['dataset']['train_tokens']}`",
        f"- val_tokens: `{manifest['dataset']['val_tokens']}`",
        "",
        "## Artifacts",
        "",
        f"- summary_json: `{manifest['artifacts']['summary_json']}`",
        f"- summary_csv: `{manifest['artifacts']['summary_csv']}`",
        f"- aggregate_summary_json: `{manifest['artifacts']['aggregate_summary_json']}`",
        f"- aggregate_summary_csv: `{manifest['artifacts']['aggregate_summary_csv']}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified training script for baseline vs SVF and ablation studies.")
    parser.add_argument("--data", type=str, default=None, help="Optional UTF-8 text file.")
    parser.add_argument("--variant", type=str, default=None, choices=["baseline", "memory", "persistent_core", "memory_core", "svf"])
    parser.add_argument("--suite", type=str, default=None, choices=sorted(VARIANT_SUITES))
    parser.add_argument("--tokenizer", type=str, default="bpe", choices=["char", "bpe"])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
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
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Optional list of seeds for repeated runs.")
    parser.add_argument("--carry-state-across-batches", action="store_true")
    parser.add_argument(
        "--match-baseline-to",
        type=str,
        default=None,
        choices=["memory", "persistent_core", "memory_core", "svf"],
        help="Auto-scale the baseline width to approximately match the parameter count of the target variant.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="outputs/experiments")
    parser.add_argument("--save-checkpoints", action="store_true")
    args = parser.parse_args()

    if bool(args.variant) == bool(args.suite):
        parser.error("Provide exactly one of --variant or --suite.")

    return args


def main() -> None:
    args = parse_args()
    text = load_text(args.data)
    train_text, val_text = split_text(text, args.val_fraction)
    tokenizer = build_tokenizer(train_text, args.tokenizer, args.vocab_size, args.min_frequency)
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    train_dataset = SequenceDataset(train_ids, seq_len=args.seq_len)
    val_dataset = SequenceDataset(val_ids, seq_len=args.seq_len)
    variants = resolve_variants(args)

    print(f"tokenizer={args.tokenizer} vocab_size={tokenizer.vocab_size}")
    print(f"train_text_chars={len(train_text)} val_text_chars={len(val_text)}")
    print(f"train_tokens={len(train_dataset.data)} val_tokens={len(val_dataset.data)}")
    print(f"variants={', '.join(variants)}")
    seeds = args.seeds if args.seeds is not None else [args.seed]
    print(f"seeds={', '.join(str(seed) for seed in seeds)}")

    results: list[ExperimentResult] = []
    tokenizer_state = tokenizer.save_state()
    for seed in seeds:
        args.seed = seed
        print(f"\n##### Seed {seed} #####")
        for variant in variants:
            results.append(
                train_variant(
                    variant=variant,
                    args=args,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    tokenizer_state=tokenizer_state,
                    vocab_size=tokenizer.vocab_size,
                )
            )

    output_dir = Path(args.output_dir)
    json_path, csv_path = write_summary(output_dir, results)
    aggregate_json_path = None
    aggregate_csv_path = None
    if len(seeds) > 1:
        aggregate_results_data = aggregate_results(results)
        aggregate_json_path, aggregate_csv_path = write_aggregate_summary(output_dir, aggregate_results_data)
    print("\n=== Summary ===")
    for result in sorted(results, key=lambda item: item.final_val_ce_loss):
        print(
            f"{result.variant:>16} "
            f"seed={result.seed:<4d} "
            f"val_ce={result.final_val_ce_loss:.4f} "
            f"ppl={result.final_val_perplexity:.4f} "
            f"params={result.parameter_count}"
        )
    if len(seeds) > 1 and aggregate_json_path is not None and aggregate_csv_path is not None:
        print("\n=== Aggregate Summary ===")
        for result in aggregate_results(results):
            print(
                f"{result.variant:>16} "
                f"runs={result.runs} "
                f"mean_val_ce={result.mean_val_ce_loss:.4f} "
                f"std={result.std_val_ce_loss:.4f} "
                f"best_seed={result.best_seed}"
            )
    print(f"summary_json={json_path}")
    print(f"summary_csv={csv_path}")
    if aggregate_json_path is not None and aggregate_csv_path is not None:
        print(f"aggregate_summary_json={aggregate_json_path}")
        print(f"aggregate_summary_csv={aggregate_csv_path}")


if __name__ == "__main__":
    main()
