# Phase E: Multi-Entity Delayed Recall

## Goal

- Stress-test `Persistent Core` as a long-term state mechanism rather than a generic CE improver.
- Move beyond single-key recall into multi-entity, multi-field retrieval.
- Record compact `core state` traces during validation.

## Task Modes

- `single_entity`
  - Backward-compatible version of the original delayed recall task.
  - Stores one entity and one field: `code`.

- `multi_entity`
  - Stores multiple entities and multiple fields per sample.
  - Example layout:
    - `remember alice age 3 0 city 7 1 color 4 2 sep bob age ...`
    - long noise context
    - `question bob city answer 7 1`

## Script

- Script: `train_delayed_recall.py`
- New controls:
  - `--task-type single_entity|multi_entity`
  - `--entities-per-sample`
  - `--fields`
  - `--save-core-traces`
  - `--trace-batches`
  - `--trace-examples`

## Suggested Formal Command

```bash
python train_delayed_recall.py --task-type multi_entity --variants baseline persistent_core --delays 1024 2048 4096 8192 --entities-per-sample 3 --fields age city color --value-length 2 --seeds 42 43 44 --save-core-traces --output-dir outputs/experiments/phaseE_multientity_recall_formal
```

## Outputs

- `recall_summary_*.json/csv`
- `recall_aggregate_*.json/csv`
- `recall_manifest_*.json/md`
- `core_traces/*.json` when `--save-core-traces` is enabled

## Main Questions

- Does `Persistent Core` retain exact-answer accuracy at `delay=2048/4096/8192` better than `baseline`?
- Does the gap widen under multi-entity interference?
- Do core traces show stable slot structure, bounded drift, and convergence toward attractor regions?
