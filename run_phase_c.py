from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DATASETS = {
    "technical": "data/svf_corpus_technical_clean_10mb.txt",
    "instruction": "data/svf_instruction_10mb.txt",
    "prose": "data/svf_prose_20mb.txt",
}

SCAN_PRESETS = {
    "capacity": [1, 2, 4, 8, 16, 32],
    "attractor": [0.0, 0.01, 0.05, 0.1, 0.2],
    "drift": [0.0, 0.05, 0.1, 0.2],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase C Persistent Core scans.")
    parser.add_argument("--scan", choices=sorted(SCAN_PRESETS), required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="prose")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--tokenizer", type=str, default="bpe", choices=["char", "bpe"])
    parser.add_argument("--split-unit", type=str, default="paragraph", choices=["char", "line", "paragraph"])
    parser.add_argument("--split-shuffle", action="store_true")
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/experiments/phaseC_scans"))
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = SCAN_PRESETS[args.scan]
    data_path = DATASETS[args.dataset]
    output_root = args.output_root / args.dataset / args.scan
    output_root.mkdir(parents=True, exist_ok=True)

    for value in values:
        if args.scan == "capacity":
            variant = "persistent_core"
            tag = f"slots_{value}"
            extra = ["--core-slots", str(int(value))]
        elif args.scan == "attractor":
            variant = "core_dynamics"
            tag = f"attractor_{value}"
            extra = ["--attractor-strength", str(value)]
        else:
            variant = "core_dynamics"
            tag = f"drift_{value}"
            extra = ["--drift-scale", str(value)]

        output_dir = output_root / tag
        command = [
            sys.executable,
            "train_experiment.py",
            "--variant",
            variant,
            "--data",
            data_path,
            "--tokenizer",
            args.tokenizer,
            "--steps",
            str(args.steps),
            "--batch-size",
            str(args.batch_size),
            "--seq-len",
            str(args.seq_len),
            "--d-model",
            str(args.d_model),
            "--layers",
            str(args.layers),
            "--heads",
            str(args.heads),
            "--device",
            args.device,
            "--eval-interval",
            str(args.eval_interval),
            "--split-unit",
            args.split_unit,
            "--split-seed",
            str(args.split_seed),
            "--output-dir",
            str(output_dir),
            "--seeds",
            *[str(seed) for seed in args.seeds],
            *extra,
        ]
        if args.split_shuffle:
            command.append("--split-shuffle")
        if args.save_checkpoints:
            command.append("--save-checkpoints")

        print(f"\n=== Phase C scan={args.scan} dataset={args.dataset} setting={tag} ===")
        print(" ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
