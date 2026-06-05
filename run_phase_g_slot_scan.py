from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase G slot-count routing scans.")
    parser.add_argument("--variants", nargs="+", default=["specialized_core"])
    parser.add_argument("--slot-counts", nargs="+", type=int, default=[2, 3, 4, 8])
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
    parser.add_argument("--slot-balance-loss", action="store_true")
    parser.add_argument("--slot-balance-weight", type=float, default=0.05)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/experiments/phaseG_slot_scan"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    balance_tag = "balance" if args.slot_balance_loss else "nobalance"

    for variant in args.variants:
        for slot_count in args.slot_counts:
            output_dir = output_root / variant / balance_tag / f"slots_{slot_count}"
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
                "--device",
                args.device,
                "--output-dir",
                str(output_dir),
            ]
            if args.slot_balance_loss:
                train_command.extend(
                    [
                        "--slot-balance-loss",
                        "--slot-balance-weight",
                        str(args.slot_balance_weight),
                    ]
                )

            analyze_command = [
                sys.executable,
                "analyze_core_slots.py",
                "--trace-dir",
                str(output_dir / "core_traces"),
                "--output-dir",
                str(output_dir),
            ]

            print(f"\n=== Phase G variant={variant} balance={balance_tag} slots={slot_count} ===")
            print(" ".join(train_command))
            subprocess.run(train_command, check=True)
            print(" ".join(analyze_command))
            subprocess.run(analyze_command, check=True)


if __name__ == "__main__":
    main()
