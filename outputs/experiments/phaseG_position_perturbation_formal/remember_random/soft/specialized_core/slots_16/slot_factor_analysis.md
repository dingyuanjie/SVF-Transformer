# Slot Routing Factor Analysis

- trace_dir: `outputs\experiments\phaseG_position_perturbation_formal\remember_random\soft\specialized_core\slots_16\core_traces`
- trace_files: `2`
- trace_entries: `64`
- slot_count: `16`
- active_write_slots: `[4, 9]`
- active_read_slots: `[4, 9]`
- dominant_write_slot_fractions: `0.00, 0.00, 0.00, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00`
- dominant_read_slot_fractions: `0.00, 0.00, 0.00, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00`

## Write Factors

### query_entity_token_start

- predictability: `0.8125`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.3125`
- normalized_mutual_information: `0.2405`

- 21: slot 9 fraction=0.75 count=4
- 31: slot 4 fraction=0.75 count=4
- 41: slot 9 fraction=1.00 count=4
- 10: slot 4 fraction=0.67 count=3
- 13: slot 4 fraction=1.00 count=3
- 23: slot 4 fraction=0.67 count=3
- 26: slot 9 fraction=0.67 count=3
- 33: slot 4 fraction=0.67 count=3

### query_fact_token_end

- predictability: `0.8125`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.3125`
- normalized_mutual_information: `0.2405`

- 23: slot 9 fraction=0.75 count=4
- 33: slot 4 fraction=0.75 count=4
- 43: slot 9 fraction=1.00 count=4
- 12: slot 4 fraction=0.67 count=3
- 15: slot 4 fraction=1.00 count=3
- 25: slot 4 fraction=0.67 count=3
- 28: slot 9 fraction=0.67 count=3
- 35: slot 4 fraction=0.67 count=3

### query_fact_token_start

- predictability: `0.8125`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.3125`
- normalized_mutual_information: `0.2405`

- 22: slot 9 fraction=0.75 count=4
- 32: slot 4 fraction=0.75 count=4
- 42: slot 9 fraction=1.00 count=4
- 11: slot 4 fraction=0.67 count=3
- 14: slot 4 fraction=1.00 count=3
- 24: slot 4 fraction=0.67 count=3
- 27: slot 9 fraction=0.67 count=3
- 34: slot 4 fraction=0.67 count=3

### query_entity_token_end

- predictability: `0.7969`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2969`
- normalized_mutual_information: `0.2311`

- 29: slot 9 fraction=0.75 count=4
- 34: slot 4 fraction=0.75 count=4
- 13: slot 4 fraction=0.67 count=3
- 16: slot 4 fraction=1.00 count=3
- 23: slot 9 fraction=1.00 count=3
- 32: slot 9 fraction=0.67 count=3
- 36: slot 4 fraction=0.67 count=3
- 39: slot 4 fraction=0.67 count=3

### prefix_noise_length

- predictability: `0.7500`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2500`
- normalized_mutual_information: `0.1917`

- 23: slot 4 fraction=1.00 count=4
- 27: slot 9 fraction=0.75 count=4
- 7: slot 4 fraction=0.50 count=4
- 11: slot 4 fraction=0.67 count=3
- 15: slot 9 fraction=0.67 count=3
- 25: slot 4 fraction=1.00 count=3
- 28: slot 4 fraction=0.67 count=3
- 4: slot 4 fraction=0.67 count=3

## Read Factors

### query_entity_token_start

- predictability: `0.8125`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.3125`
- normalized_mutual_information: `0.2405`

- 21: slot 9 fraction=0.75 count=4
- 31: slot 4 fraction=0.75 count=4
- 41: slot 9 fraction=1.00 count=4
- 10: slot 4 fraction=0.67 count=3
- 13: slot 4 fraction=1.00 count=3
- 23: slot 4 fraction=0.67 count=3
- 26: slot 9 fraction=0.67 count=3
- 33: slot 4 fraction=0.67 count=3

### query_fact_token_end

- predictability: `0.8125`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.3125`
- normalized_mutual_information: `0.2405`

- 23: slot 9 fraction=0.75 count=4
- 33: slot 4 fraction=0.75 count=4
- 43: slot 9 fraction=1.00 count=4
- 12: slot 4 fraction=0.67 count=3
- 15: slot 4 fraction=1.00 count=3
- 25: slot 4 fraction=0.67 count=3
- 28: slot 9 fraction=0.67 count=3
- 35: slot 4 fraction=0.67 count=3

### query_fact_token_start

- predictability: `0.8125`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.3125`
- normalized_mutual_information: `0.2405`

- 22: slot 9 fraction=0.75 count=4
- 32: slot 4 fraction=0.75 count=4
- 42: slot 9 fraction=1.00 count=4
- 11: slot 4 fraction=0.67 count=3
- 14: slot 4 fraction=1.00 count=3
- 24: slot 4 fraction=0.67 count=3
- 27: slot 9 fraction=0.67 count=3
- 34: slot 4 fraction=0.67 count=3

### query_entity_token_end

- predictability: `0.7969`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2969`
- normalized_mutual_information: `0.2311`

- 29: slot 9 fraction=0.75 count=4
- 34: slot 4 fraction=0.75 count=4
- 13: slot 4 fraction=0.67 count=3
- 16: slot 4 fraction=1.00 count=3
- 23: slot 9 fraction=1.00 count=3
- 32: slot 9 fraction=0.67 count=3
- 36: slot 4 fraction=0.67 count=3
- 39: slot 4 fraction=0.67 count=3

### prefix_noise_length

- predictability: `0.7500`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2500`
- normalized_mutual_information: `0.1917`

- 23: slot 4 fraction=1.00 count=4
- 27: slot 9 fraction=0.75 count=4
- 7: slot 4 fraction=0.50 count=4
- 11: slot 4 fraction=0.67 count=3
- 15: slot 9 fraction=0.67 count=3
- 25: slot 4 fraction=1.00 count=3
- 28: slot 4 fraction=0.67 count=3
- 4: slot 4 fraction=0.67 count=3
