# Slot Routing Factor Analysis

- trace_dir: `outputs\experiments\smoke_position_perturbation\core_traces`
- trace_files: `1`
- trace_entries: `2`
- slot_count: `4`
- active_write_slots: `[0, 1, 2, 3]`
- active_read_slots: `[0, 1, 2, 3]`
- dominant_write_slot_fractions: `0.00, 0.00, 1.00, 0.00`
- dominant_read_slot_fractions: `0.00, 0.50, 0.00, 0.50`

## Write Factors

### answer_digit_sum_parity

- predictability: `1.0000`
- baseline_accuracy: `1.0000`
- gain_over_baseline: `0.0000`
- normalized_mutual_information: `0.0000`

- even: slot 2 fraction=1.00 count=2

### answer_first_token

- predictability: `1.0000`
- baseline_accuracy: `1.0000`
- gain_over_baseline: `0.0000`
- normalized_mutual_information: `0.0000`

- 2: slot 2 fraction=1.00 count=1
- 8: slot 2 fraction=1.00 count=1

### answer_value

- predictability: `1.0000`
- baseline_accuracy: `1.0000`
- gain_over_baseline: `0.0000`
- normalized_mutual_information: `0.0000`

- 2: slot 2 fraction=1.00 count=1
- 8: slot 2 fraction=1.00 count=1

### entity_count

- predictability: `1.0000`
- baseline_accuracy: `1.0000`
- gain_over_baseline: `0.0000`
- normalized_mutual_information: `0.0000`

- 4: slot 2 fraction=1.00 count=2

### field_order_mode

- predictability: `1.0000`
- baseline_accuracy: `1.0000`
- gain_over_baseline: `0.0000`
- normalized_mutual_information: `0.0000`

- shuffled: slot 2 fraction=1.00 count=2

## Read Factors

### answer_first_token

- predictability: `1.0000`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.5000`
- normalized_mutual_information: `1.0000`

- 2: slot 1 fraction=1.00 count=1
- 8: slot 3 fraction=1.00 count=1

### answer_value

- predictability: `1.0000`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.5000`
- normalized_mutual_information: `1.0000`

- 2: slot 1 fraction=1.00 count=1
- 8: slot 3 fraction=1.00 count=1

### prefix_noise_length

- predictability: `1.0000`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.5000`
- normalized_mutual_information: `1.0000`

- 5: slot 3 fraction=1.00 count=1
- 6: slot 1 fraction=1.00 count=1

### query_entity_index

- predictability: `1.0000`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.5000`
- normalized_mutual_information: `1.0000`

- 0: slot 1 fraction=1.00 count=1
- 1: slot 3 fraction=1.00 count=1

### query_entity_parity

- predictability: `1.0000`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.5000`
- normalized_mutual_information: `1.0000`

- even: slot 1 fraction=1.00 count=1
- odd: slot 3 fraction=1.00 count=1
