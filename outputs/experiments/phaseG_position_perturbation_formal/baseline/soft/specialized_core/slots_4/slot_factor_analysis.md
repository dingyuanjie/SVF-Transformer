# Slot Routing Factor Analysis

- trace_dir: `outputs\experiments\phaseG_position_perturbation_formal\baseline\soft\specialized_core\slots_4\core_traces`
- trace_files: `2`
- trace_entries: `64`
- slot_count: `4`
- active_write_slots: `[0, 3]`
- active_read_slots: `[0, 3]`
- dominant_write_slot_fractions: `0.50, 0.00, 0.00, 0.50`
- dominant_read_slot_fractions: `0.50, 0.00, 0.00, 0.50`

## Write Factors

### answer_first_token

- predictability: `0.6250`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.1250`
- normalized_mutual_information: `0.0498`

- 0: slot 0 fraction=0.80 count=10
- 3: slot 3 fraction=0.56 count=9
- 5: slot 0 fraction=0.50 count=8
- 1: slot 3 fraction=0.57 count=7
- 8: slot 3 fraction=0.67 count=6
- 9: slot 0 fraction=0.50 count=6
- 2: slot 3 fraction=0.80 count=5
- 7: slot 3 fraction=0.60 count=5

### answer_value

- predictability: `0.6250`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.1250`
- normalized_mutual_information: `0.0498`

- 0: slot 0 fraction=0.80 count=10
- 3: slot 3 fraction=0.56 count=9
- 5: slot 0 fraction=0.50 count=8
- 1: slot 3 fraction=0.57 count=7
- 8: slot 3 fraction=0.67 count=6
- 9: slot 0 fraction=0.50 count=6
- 2: slot 3 fraction=0.80 count=5
- 7: slot 3 fraction=0.60 count=5

### query_name

- predictability: `0.6094`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.1094`
- normalized_mutual_information: `0.0411`

- frank: slot 0 fraction=0.62 count=8
- niaj: slot 3 fraction=0.62 count=8
- alice: slot 0 fraction=0.57 count=7
- erin: slot 3 fraction=0.57 count=7
- grace: slot 3 fraction=0.71 count=7
- bob: slot 0 fraction=0.50 count=6
- carol: slot 0 fraction=0.50 count=6
- heidi: slot 0 fraction=0.75 count=4

### query_name_initial

- predictability: `0.6094`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.1094`
- normalized_mutual_information: `0.0411`

- f: slot 0 fraction=0.62 count=8
- n: slot 3 fraction=0.62 count=8
- a: slot 0 fraction=0.57 count=7
- e: slot 3 fraction=0.57 count=7
- g: slot 3 fraction=0.71 count=7
- b: slot 0 fraction=0.50 count=6
- c: slot 0 fraction=0.50 count=6
- h: slot 0 fraction=0.75 count=4

### answer_digit_sum_parity

- predictability: `0.5469`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.0469`
- normalized_mutual_information: `0.0064`

- odd: slot 3 fraction=0.54 count=35
- even: slot 0 fraction=0.55 count=29

## Read Factors

### answer_first_token

- predictability: `0.6250`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.1250`
- normalized_mutual_information: `0.0498`

- 0: slot 0 fraction=0.80 count=10
- 3: slot 3 fraction=0.56 count=9
- 5: slot 0 fraction=0.50 count=8
- 1: slot 3 fraction=0.57 count=7
- 8: slot 3 fraction=0.67 count=6
- 9: slot 0 fraction=0.50 count=6
- 2: slot 3 fraction=0.80 count=5
- 7: slot 3 fraction=0.60 count=5

### answer_value

- predictability: `0.6250`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.1250`
- normalized_mutual_information: `0.0498`

- 0: slot 0 fraction=0.80 count=10
- 3: slot 3 fraction=0.56 count=9
- 5: slot 0 fraction=0.50 count=8
- 1: slot 3 fraction=0.57 count=7
- 8: slot 3 fraction=0.67 count=6
- 9: slot 0 fraction=0.50 count=6
- 2: slot 3 fraction=0.80 count=5
- 7: slot 3 fraction=0.60 count=5

### query_name

- predictability: `0.6094`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.1094`
- normalized_mutual_information: `0.0411`

- frank: slot 0 fraction=0.62 count=8
- niaj: slot 3 fraction=0.62 count=8
- alice: slot 0 fraction=0.57 count=7
- erin: slot 3 fraction=0.57 count=7
- grace: slot 3 fraction=0.71 count=7
- bob: slot 0 fraction=0.50 count=6
- carol: slot 0 fraction=0.50 count=6
- heidi: slot 0 fraction=0.75 count=4

### query_name_initial

- predictability: `0.6094`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.1094`
- normalized_mutual_information: `0.0411`

- f: slot 0 fraction=0.62 count=8
- n: slot 3 fraction=0.62 count=8
- a: slot 0 fraction=0.57 count=7
- e: slot 3 fraction=0.57 count=7
- g: slot 3 fraction=0.71 count=7
- b: slot 0 fraction=0.50 count=6
- c: slot 0 fraction=0.50 count=6
- h: slot 0 fraction=0.75 count=4

### answer_digit_sum_parity

- predictability: `0.5469`
- baseline_accuracy: `0.5000`
- gain_over_baseline: `0.0469`
- normalized_mutual_information: `0.0064`

- odd: slot 3 fraction=0.54 count=35
- even: slot 0 fraction=0.55 count=29
