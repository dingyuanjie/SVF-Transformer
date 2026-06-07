# Slot Routing Factor Analysis

- trace_dir: `outputs\experiments\smoke_top1_routing\core_traces`
- trace_files: `1`
- trace_entries: `2`
- slot_count: `4`
- active_write_slots: `[0, 2]`
- active_read_slots: `[0, 1, 2, 3]`
- dominant_write_slot_fractions: `0.50, 0.00, 0.50, 0.00`
- dominant_read_slot_fractions: `0.50, 0.00, 0.50, 0.00`

## Write Factors

### answer_digit_sum_parity

- predictability: `1.0000`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.5000`
- normalized_mutual_information: `1.0000`

- even: slot 0 fraction=1.00 count=1
- odd: slot 2 fraction=1.00 count=1

### answer_first_token

- predictability: `1.0000`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.5000`
- normalized_mutual_information: `1.0000`

- 4: slot 0 fraction=1.00 count=1
- 5: slot 2 fraction=1.00 count=1

### answer_value

- predictability: `1.0000`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.5000`
- normalized_mutual_information: `1.0000`

- 4: slot 0 fraction=1.00 count=1
- 5: slot 2 fraction=1.00 count=1

### query_entity_bucket

- predictability: `1.0000`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.5000`
- normalized_mutual_information: `1.0000`

- back_half: slot 0 fraction=1.00 count=1
- front_half: slot 2 fraction=1.00 count=1

### query_entity_index

- predictability: `1.0000`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.5000`
- normalized_mutual_information: `1.0000`

- 0: slot 2 fraction=1.00 count=1
- 2: slot 0 fraction=1.00 count=1

## Read Factors

### answer_digit_sum_parity

- predictability: `1.0000`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.5000`
- normalized_mutual_information: `1.0000`

- even: slot 0 fraction=1.00 count=1
- odd: slot 2 fraction=1.00 count=1

### answer_first_token

- predictability: `1.0000`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.5000`
- normalized_mutual_information: `1.0000`

- 4: slot 0 fraction=1.00 count=1
- 5: slot 2 fraction=1.00 count=1

### answer_value

- predictability: `1.0000`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.5000`
- normalized_mutual_information: `1.0000`

- 4: slot 0 fraction=1.00 count=1
- 5: slot 2 fraction=1.00 count=1

### query_entity_bucket

- predictability: `1.0000`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.5000`
- normalized_mutual_information: `1.0000`

- back_half: slot 0 fraction=1.00 count=1
- front_half: slot 2 fraction=1.00 count=1

### query_entity_index

- predictability: `1.0000`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.5000`
- normalized_mutual_information: `1.0000`

- 0: slot 2 fraction=1.00 count=1
- 2: slot 0 fraction=1.00 count=1
