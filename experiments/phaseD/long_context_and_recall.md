# Phase D: Long-Range Validation

## Goal

- Test whether `Persistent Core` helps more as context length grows.
- Test whether `Persistent Core` improves delayed recall beyond ordinary language-model CE.

## Long-Context Scaling

- Script: `run_phase_d_long_context.py`
- Default variants: `baseline`, `persistent_core`, `svf`
- Default context lengths: `128 256 512 1024 2048`
- Uses the unified training pipeline, so it preserves:
  - tokenizer fit on train split only
  - shuffled paragraph split
  - multi-seed aggregation
  - manifest generation
  - optional baseline parameter matching

### Suggested command

```bash
python run_phase_d_long_context.py --dataset prose --seeds 42 43 44 --split-shuffle --save-checkpoints --output-root outputs/experiments/phaseD_long_context_formal
```

### Question answered

- Does the performance gap between `Persistent Core` and `baseline` widen as `seq_len` increases?

## Delayed Recall

- Script: `train_delayed_recall.py`
- Default variants: `baseline`, `persistent_core`
- Default delays: `128 256 512 1024`
- Task format:
  - remember a name and digit code
  - insert long filler context
  - ask for the code later

### Metrics

- `final_val_ce_loss`
- `final_answer_token_accuracy`
- `final_answer_exact_accuracy`

### Suggested command

```bash
python train_delayed_recall.py --variants baseline persistent_core --delays 128 256 512 1024 --seeds 42 43 44 --output-dir outputs/experiments/phaseD_delayed_recall_formal
```

### Questions answered

- Does `Persistent Core` preserve recall accuracy better at longer delays?
- Does its advantage grow on exact answer recovery instead of only teacher-forced CE?
