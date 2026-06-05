# Phase G: Slot Count Routing Scan

Goal: test whether the current routing mechanism naturally settles into a two-slot solution even when `core_slots` changes.

## Main Question

- Does routing fully use `2` slots, then continue to use only `2` slots when capacity increases to `3/4/8`?

## Recommended Scan

```bash
python run_phase_g_slot_scan.py --variants specialized_core --slot-counts 2 3 4 8 --seeds 42 43 --delays 32 --steps 300 --batch-size 16 --train-samples 512 --val-samples 128 --entities-per-sample 4 --fields color --value-length 1 --d-model 64 --d-ff 128 --layers 1 --heads 4 --device cuda --output-root outputs/experiments/phaseG_slot_scan
```

Balance-loss comparison:

```bash
python run_phase_g_slot_scan.py --variants specialized_core --slot-counts 2 3 4 8 --seeds 42 43 --delays 32 --steps 300 --batch-size 16 --train-samples 512 --val-samples 128 --entities-per-sample 4 --fields color --value-length 1 --d-model 64 --d-ff 128 --layers 1 --heads 4 --device cuda --slot-balance-loss --slot-balance-weight 0.05 --output-root outputs/experiments/phaseG_slot_scan
```

## Readouts

- `recall_aggregate_*.json`: task metrics
- `slot_analysis_*.json`: per-run slot diagnostics
- `slot_analysis_summary_*.json`: aggregate routing frequencies and `query_name/query_field -> slot`

## Key Metrics

- `dominant_write_slot_fractions`
- `mean_write_slot_weights`
- `mean_slot_routing_entropy`
- `query_name_write_summary`
- `query_field_write_summary`
