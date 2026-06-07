from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CONDITION_PRESETS: dict[str, dict[str, str]] = {
    "baseline": {"field_order_mode": "fixed", "remember_position_mode": "front"},
    "shuffle_fields": {"field_order_mode": "shuffled", "remember_position_mode": "front"},
    "remember_middle": {"field_order_mode": "fixed", "remember_position_mode": "middle"},
    "remember_back": {"field_order_mode": "fixed", "remember_position_mode": "back"},
    "remember_random": {"field_order_mode": "fixed", "remember_position_mode": "random"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run position perturbation studies for delayed recall routing."
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=sorted(CONDITION_PRESETS.keys()),
        default=["baseline", "shuffle_fields", "remember_middle", "remember_random"],
    )
    parser.add_argument("--variants", nargs="+", default=["specialized_core"])
    parser.add_argument("--slot-counts", nargs="+", type=int, default=[4, 16])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--delays", nargs="+", type=int, default=[32])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--val-samples", type=int, default=128)
    parser.add_argument("--entities-per-sample", type=int, default=4)
    parser.add_argument("--fields", nargs="+", default=["color"])
    parser.add_argument("--value-length", type=int, default=1)
    parser.add_argument("--noise-vocab-size", type=int, default=64)
    parser.add_argument("--trace-batches", type=int, default=8)
    parser.add_argument("--trace-examples", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top1-routing", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/experiments/phaseG_position_perturbation"))
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    routing_tag = "top1" if args.top1_routing else "soft"

    for condition in args.conditions:
        preset = CONDITION_PRESETS[condition]
        for variant in args.variants:
            for slot_count in args.slot_counts:
                output_dir = output_root / condition / routing_tag / variant / f"slots_{slot_count}"
                train_command = [
                    sys.executable,
                    "train_delayed_recall.py",
                    "--variants",
                    variant,
                    "--task-type",
                    "multi_entity",
                    "--delays",
                    *[str(delay) for delay in args.delays],
                    "--seeds",
                    *[str(seed) for seed in args.seeds],
                    "--steps",
                    str(args.steps),
                    "--batch-size",
                    str(args.batch_size),
                    "--train-samples",
                    str(args.train_samples),
                    "--val-samples",
                    str(args.val_samples),
                    "--entities-per-sample",
                    str(args.entities_per_sample),
                    "--fields",
                    *args.fields,
                    "--value-length",
                    str(args.value_length),
                    "--noise-vocab-size",
                    str(args.noise_vocab_size),
                    "--save-core-traces",
                    "--trace-batches",
                    str(args.trace_batches),
                    "--trace-examples",
                    str(args.trace_examples),
                    "--d-model",
                    str(args.d_model),
                    "--d-ff",
                    str(args.d_ff),
                    "--layers",
                    str(args.layers),
                    "--heads",
                    str(args.heads),
                    "--core-slots",
                    str(slot_count),
                    "--field-order-mode",
                    preset["field_order_mode"],
                    "--remember-position-mode",
                    preset["remember_position_mode"],
                    "--device",
                    args.device,
                    "--output-dir",
                    str(output_dir),
                ]
                if args.top1_routing:
                    train_command.append("--top1-routing")

                slot_analysis_command = [
                    sys.executable,
                    "analyze_core_slots.py",
                    "--trace-dir",
                    str(output_dir / "core_traces"),
                    "--output-dir",
                    str(output_dir),
                ]
                factor_analysis_command = [
                    sys.executable,
                    "analyze_slot_factors.py",
                    "--trace-dir",
                    str(output_dir / "core_traces"),
                    "--output-dir",
                    str(output_dir),
                ]

                print(
                    f"\n=== Position perturbation condition={condition} routing={routing_tag} "
                    f"variant={variant} slots={slot_count} ==="
                )
                run_command(train_command)
                run_command(slot_analysis_command)
                run_command(factor_analysis_command)


if __name__ == "__main__":
    main()
