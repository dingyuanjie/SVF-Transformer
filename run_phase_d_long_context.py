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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run long-context scaling experiments.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="prose")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "persistent_core", "svf"],
        choices=["baseline", "memory", "persistent_core", "core_dynamics", "memory_core", "svf"],
    )
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--tokenizer", type=str, default="bpe", choices=["char", "bpe"])
    parser.add_argument("--split-unit", type=str, default="paragraph", choices=["char", "line", "paragraph"])
    parser.add_argument("--split-shuffle", action="store_true")
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--match-baseline-to", type=str, default="svf", choices=["memory", "persistent_core", "memory_core", "svf"])
    parser.add_argument("--output-root", type=Path, default=Path("outputs/experiments/phaseD_long_context"))
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = DATASETS[args.dataset]
    output_root = args.output_root / args.dataset
    output_root.mkdir(parents=True, exist_ok=True)

    for seq_len in args.seq_lens:
        output_dir = output_root / f"seq_{seq_len}"
        command = [
            sys.executable,
            "train_experiment.py",
            "--variants",
            *args.variants,
            "--data",
            data_path,
            "--tokenizer",
            args.tokenizer,
            "--steps",
            str(args.steps),
            "--batch-size",
            str(args.batch_size),
            "--seq-len",
            str(seq_len),
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
        ]
        if "baseline" in args.variants and any(variant != "baseline" for variant in args.variants):
            command.extend(["--match-baseline-to", args.match_baseline_to])
        if args.split_shuffle:
            command.append("--split-shuffle")
        if args.save_checkpoints:
            command.append("--save-checkpoints")

        print(f"\n=== Phase D long-context dataset={args.dataset} seq_len={seq_len} ===")
        print(" ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
