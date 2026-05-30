# Experiments

This directory stores experiment-facing documentation so the result trail is preserved alongside the codebase.

## Layout

- `phase1/`: baseline vs SVF comparison records.
- `phase2/`: ablation records.
- `strict_reproduction/`: stricter protocol reruns and parameter-matched controls.

Large binary artifacts such as checkpoints and raw logs remain under `outputs/experiments/`.
Each completed run should have:

- command line and seed list
- protocol notes
- output directory
- summary files
- checkpoint directory

## Current Runs

- `phase1/formal_phase1.md`: first formal baseline vs SVF result.
- `phase2/formal_phase2.md`: formal ablation result.
- `strict_reproduction/strict_phase1_parammatch.md`: stricter rerun with train-only tokenizer fitting, independent eval batches, and baseline parameter matching.
