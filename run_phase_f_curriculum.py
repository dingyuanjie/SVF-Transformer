from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CURRICULUM_LEVELS = {
    "level1": {
        "task_type": "multi_entity",
        "delays": [512],
        "entities_per_sample": 2,
        "fields": ["age", "city"],
        "value_length": 2,
    },
    "level2": {
        "task_type": "multi_entity",
        "delays": [1024],
        "entities_per_sample": 2,
        "fields": ["age", "city"],
        "value_length": 2,
    },
    "level3": {
        "task_type": "multi_entity",
        "delays": [1024],
        "entities_per_sample": 3,
        "fields": ["age", "city", "color"],
        "value_length": 2,
    },
    "level4": {
        "task_type": "multi_entity",
        "delays": [2048],
        "entities_per_sample": 4,
        "fields": ["age", "city", "color"],
        "value_length": 2,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run curriculum delayed-recall experiments.")
    parser.add_argument("--levels", nargs="+", default=list(CURRICULUM_LEVELS), choices=sorted(CURRICULUM_LEVELS))
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "persistent_core"],
        choices=["baseline", "memory", "persistent_core", "core_dynamics", "memory_core", "svf"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-core-traces", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/experiments/phaseF_curriculum"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    for level_name in args.levels:
        spec = CURRICULUM_LEVELS[level_name]
        output_dir = output_root / level_name
        command = [
            sys.executable,
            "train_delayed_recall.py",
            "--task-type",
            spec["task_type"],
            "--variants",
            *args.variants,
            "--delays",
            *[str(item) for item in spec["delays"]],
            "--entities-per-sample",
            str(spec["entities_per_sample"]),
            "--fields",
            *spec["fields"],
            "--value-length",
            str(spec["value_length"]),
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
            "--device",
            args.device,
            "--output-dir",
            str(output_dir),
        ]
        if args.save_core_traces:
            command.append("--save-core-traces")

        print(f"\n=== Phase F curriculum {level_name} ===")
        print(" ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
