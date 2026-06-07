# Slot Routing Factor Analysis

- trace_dir: `outputs\experiments\phaseG_multifield_rank_study_formal\specialized_core\nobalance_soft\slots_8\core_traces`
- trace_files: `2`
- trace_entries: `64`
- slot_count: `8`
- active_write_slots: `[0, 6]`
- active_read_slots: `[0, 6]`
- dominant_write_slot_fractions: `0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.00`
- dominant_read_slot_fractions: `0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.00`

## Write Factors

### query_fact_to_context_distance

- predictability: `0.7500`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2500`
- normalized_mutual_information: `0.1791`

- 22: slot 6 fraction=0.88 count=8
- 34: slot 0 fraction=0.57 count=7
- 6: slot 0 fraction=0.71 count=7
- 2: slot 0 fraction=0.67 count=6
- 38: slot 6 fraction=0.60 count=5
- 12: slot 6 fraction=0.75 count=4
- 18: slot 6 fraction=1.00 count=4
- 24: slot 0 fraction=0.50 count=4

### query_fact_token_end

- predictability: `0.7500`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2500`
- normalized_mutual_information: `0.1791`

- 20: slot 6 fraction=0.88 count=8
- 36: slot 0 fraction=0.71 count=7
- 8: slot 0 fraction=0.57 count=7
- 40: slot 0 fraction=0.67 count=6
- 4: slot 6 fraction=0.60 count=5
- 14: slot 0 fraction=0.50 count=4
- 18: slot 0 fraction=0.50 count=4
- 24: slot 6 fraction=1.00 count=4

### query_fact_token_start

- predictability: `0.7500`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2500`
- normalized_mutual_information: `0.1791`

- 19: slot 6 fraction=0.88 count=8
- 35: slot 0 fraction=0.71 count=7
- 7: slot 0 fraction=0.57 count=7
- 39: slot 0 fraction=0.67 count=6
- 3: slot 6 fraction=0.60 count=5
- 13: slot 0 fraction=0.50 count=4
- 17: slot 0 fraction=0.50 count=4
- 23: slot 6 fraction=1.00 count=4

### answer_first_token

- predictability: `0.7031`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2031`
- normalized_mutual_information: `0.1121`

- 7: slot 0 fraction=0.60 count=10
- 0: slot 0 fraction=0.62 count=8
- 4: slot 0 fraction=0.86 count=7
- 8: slot 0 fraction=0.71 count=7
- 2: slot 6 fraction=0.67 count=6
- 3: slot 6 fraction=1.00 count=6
- 9: slot 6 fraction=0.67 count=6
- 1: slot 0 fraction=0.60 count=5

### answer_value

- predictability: `0.7031`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2031`
- normalized_mutual_information: `0.1121`

- 7: slot 0 fraction=0.60 count=10
- 0: slot 0 fraction=0.62 count=8
- 4: slot 0 fraction=0.86 count=7
- 8: slot 0 fraction=0.71 count=7
- 2: slot 6 fraction=0.67 count=6
- 3: slot 6 fraction=1.00 count=6
- 9: slot 6 fraction=0.67 count=6
- 1: slot 0 fraction=0.60 count=5

## Read Factors

### query_fact_to_context_distance

- predictability: `0.7500`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2500`
- normalized_mutual_information: `0.1791`

- 22: slot 6 fraction=0.88 count=8
- 34: slot 0 fraction=0.57 count=7
- 6: slot 0 fraction=0.71 count=7
- 2: slot 0 fraction=0.67 count=6
- 38: slot 6 fraction=0.60 count=5
- 12: slot 6 fraction=0.75 count=4
- 18: slot 6 fraction=1.00 count=4
- 24: slot 0 fraction=0.50 count=4

### query_fact_token_end

- predictability: `0.7500`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2500`
- normalized_mutual_information: `0.1791`

- 20: slot 6 fraction=0.88 count=8
- 36: slot 0 fraction=0.71 count=7
- 8: slot 0 fraction=0.57 count=7
- 40: slot 0 fraction=0.67 count=6
- 4: slot 6 fraction=0.60 count=5
- 14: slot 0 fraction=0.50 count=4
- 18: slot 0 fraction=0.50 count=4
- 24: slot 6 fraction=1.00 count=4

### query_fact_token_start

- predictability: `0.7500`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2500`
- normalized_mutual_information: `0.1791`

- 19: slot 6 fraction=0.88 count=8
- 35: slot 0 fraction=0.71 count=7
- 7: slot 0 fraction=0.57 count=7
- 39: slot 0 fraction=0.67 count=6
- 3: slot 6 fraction=0.60 count=5
- 13: slot 0 fraction=0.50 count=4
- 17: slot 0 fraction=0.50 count=4
- 23: slot 6 fraction=1.00 count=4

### answer_first_token

- predictability: `0.7031`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2031`
- normalized_mutual_information: `0.1121`

- 7: slot 0 fraction=0.60 count=10
- 0: slot 0 fraction=0.62 count=8
- 4: slot 0 fraction=0.86 count=7
- 8: slot 0 fraction=0.71 count=7
- 2: slot 6 fraction=0.67 count=6
- 3: slot 6 fraction=1.00 count=6
- 9: slot 6 fraction=0.67 count=6
- 1: slot 0 fraction=0.60 count=5

### answer_value

- predictability: `0.7031`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2031`
- normalized_mutual_information: `0.1121`

- 7: slot 0 fraction=0.60 count=10
- 0: slot 0 fraction=0.62 count=8
- 4: slot 0 fraction=0.86 count=7
- 8: slot 0 fraction=0.71 count=7
- 2: slot 6 fraction=0.67 count=6
- 3: slot 6 fraction=1.00 count=6
- 9: slot 6 fraction=0.67 count=6
- 1: slot 0 fraction=0.60 count=5
