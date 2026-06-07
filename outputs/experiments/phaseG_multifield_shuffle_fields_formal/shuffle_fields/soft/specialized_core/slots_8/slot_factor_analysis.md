# Slot Routing Factor Analysis

- trace_dir: `outputs\experiments\phaseG_multifield_shuffle_fields_formal\shuffle_fields\soft\specialized_core\slots_8\core_traces`
- trace_files: `2`
- trace_entries: `64`
- slot_count: `8`
- active_write_slots: `[0, 6]`
- active_read_slots: `[0, 6]`
- dominant_write_slot_fractions: `0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.00`
- dominant_read_slot_fractions: `0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.00`

## Write Factors

### query_fact_to_context_distance

- predictability: `0.7656`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2656`
- normalized_mutual_information: `0.1868`

- 14: slot 0 fraction=0.89 count=9
- 28: slot 6 fraction=1.00 count=7
- 8: slot 6 fraction=0.83 count=6
- 12: slot 6 fraction=0.80 count=5
- 34: slot 0 fraction=0.60 count=5
- 36: slot 6 fraction=0.60 count=5
- 32: slot 0 fraction=0.50 count=4
- 38: slot 0 fraction=0.50 count=4

### query_fact_token_end

- predictability: `0.7656`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2656`
- normalized_mutual_information: `0.1868`

- 28: slot 0 fraction=0.89 count=9
- 14: slot 6 fraction=1.00 count=7
- 34: slot 6 fraction=0.83 count=6
- 30: slot 6 fraction=0.80 count=5
- 6: slot 6 fraction=0.60 count=5
- 8: slot 0 fraction=0.60 count=5
- 10: slot 0 fraction=0.50 count=4
- 4: slot 0 fraction=0.50 count=4

### query_fact_token_start

- predictability: `0.7656`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2656`
- normalized_mutual_information: `0.1868`

- 27: slot 0 fraction=0.89 count=9
- 13: slot 6 fraction=1.00 count=7
- 33: slot 6 fraction=0.83 count=6
- 29: slot 6 fraction=0.80 count=5
- 5: slot 6 fraction=0.60 count=5
- 7: slot 0 fraction=0.60 count=5
- 3: slot 0 fraction=0.50 count=4
- 9: slot 0 fraction=0.50 count=4

### query_name

- predictability: `0.7344`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2344`
- normalized_mutual_information: `0.1555`

- judy: slot 0 fraction=0.50 count=10
- erin: slot 6 fraction=0.89 count=9
- dave: slot 0 fraction=0.57 count=7
- grace: slot 0 fraction=0.83 count=6
- mallory: slot 0 fraction=0.83 count=6
- niaj: slot 6 fraction=0.67 count=6
- frank: slot 6 fraction=0.60 count=5
- alice: slot 0 fraction=1.00 count=4

### query_name_initial

- predictability: `0.7344`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2344`
- normalized_mutual_information: `0.1555`

- j: slot 0 fraction=0.50 count=10
- e: slot 6 fraction=0.89 count=9
- d: slot 0 fraction=0.57 count=7
- g: slot 0 fraction=0.83 count=6
- m: slot 0 fraction=0.83 count=6
- n: slot 6 fraction=0.67 count=6
- f: slot 6 fraction=0.60 count=5
- a: slot 0 fraction=1.00 count=4

## Read Factors

### query_fact_to_context_distance

- predictability: `0.7656`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2656`
- normalized_mutual_information: `0.1868`

- 14: slot 0 fraction=0.89 count=9
- 28: slot 6 fraction=1.00 count=7
- 8: slot 6 fraction=0.83 count=6
- 12: slot 6 fraction=0.80 count=5
- 34: slot 0 fraction=0.60 count=5
- 36: slot 6 fraction=0.60 count=5
- 32: slot 0 fraction=0.50 count=4
- 38: slot 0 fraction=0.50 count=4

### query_fact_token_end

- predictability: `0.7656`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2656`
- normalized_mutual_information: `0.1868`

- 28: slot 0 fraction=0.89 count=9
- 14: slot 6 fraction=1.00 count=7
- 34: slot 6 fraction=0.83 count=6
- 30: slot 6 fraction=0.80 count=5
- 6: slot 6 fraction=0.60 count=5
- 8: slot 0 fraction=0.60 count=5
- 10: slot 0 fraction=0.50 count=4
- 4: slot 0 fraction=0.50 count=4

### query_fact_token_start

- predictability: `0.7656`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2656`
- normalized_mutual_information: `0.1868`

- 27: slot 0 fraction=0.89 count=9
- 13: slot 6 fraction=1.00 count=7
- 33: slot 6 fraction=0.83 count=6
- 29: slot 6 fraction=0.80 count=5
- 5: slot 6 fraction=0.60 count=5
- 7: slot 0 fraction=0.60 count=5
- 3: slot 0 fraction=0.50 count=4
- 9: slot 0 fraction=0.50 count=4

### query_name

- predictability: `0.7344`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2344`
- normalized_mutual_information: `0.1555`

- judy: slot 0 fraction=0.50 count=10
- erin: slot 6 fraction=0.89 count=9
- dave: slot 0 fraction=0.57 count=7
- grace: slot 0 fraction=0.83 count=6
- mallory: slot 0 fraction=0.83 count=6
- niaj: slot 6 fraction=0.67 count=6
- frank: slot 6 fraction=0.60 count=5
- alice: slot 0 fraction=1.00 count=4

### query_name_initial

- predictability: `0.7344`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.2344`
- normalized_mutual_information: `0.1555`

- j: slot 0 fraction=0.50 count=10
- e: slot 6 fraction=0.89 count=9
- d: slot 0 fraction=0.57 count=7
- g: slot 0 fraction=0.83 count=6
- m: slot 0 fraction=0.83 count=6
- n: slot 6 fraction=0.67 count=6
- f: slot 6 fraction=0.60 count=5
- a: slot 0 fraction=1.00 count=4
