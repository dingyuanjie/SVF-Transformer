# Cheap Routing Mapping Report

- input_root: `outputs\experiments\phaseG_slot_scan\specialized_core`

## Main Question

- Does routing fully use 2 slots, then keep using only 2 slots when capacity grows?

## Verdict

- specialized_core / balance: NO (slot_2_full=False, larger_collapse=True)
  - slots=2: write=[1], read=[1]
  - slots=3: write=[1, 2], read=[2]
  - slots=4: write=[0, 3], read=[0, 3]
  - slots=8: write=[0, 2], read=[0, 2]
- specialized_core / nobalance: NO (slot_2_full=False, larger_collapse=True)
  - slots=2: write=[1], read=[1]
  - slots=3: write=[2], read=[2]
  - slots=4: write=[0, 3], read=[0, 3]
  - slots=8: write=[0, 2], read=[0, 2]

## Routing Groups

### specialized_core / balance / slots_2

- trace_files: `2`
- trace_entries: `64`
- effective_write_slots: `[1]`
- effective_read_slots: `[1]`
- dominant_write_slot_fractions: `0.00, 1.00`
- dominant_read_slot_fractions: `0.00, 1.00`
- mean_write_slot_weights: `0.06, 0.94`
- mean_read_slot_weights: `0.00, 1.00`

#### query_name -> write slot

- alice -> slot 1 (1.00)
- bob -> slot 1 (1.00)
- carol -> slot 1 (1.00)
- dave -> slot 1 (1.00)
- erin -> slot 1 (1.00)
- frank -> slot 1 (1.00)
- grace -> slot 1 (1.00)
- heidi -> slot 1 (1.00)

#### query_field -> write slot

- color -> slot 1 (1.00)

#### query_name -> read slot

- alice -> slot 1 (1.00)
- bob -> slot 1 (1.00)
- carol -> slot 1 (1.00)
- dave -> slot 1 (1.00)
- erin -> slot 1 (1.00)
- frank -> slot 1 (1.00)
- grace -> slot 1 (1.00)
- heidi -> slot 1 (1.00)

#### query_field -> read slot

- color -> slot 1 (1.00)

### specialized_core / balance / slots_3

- trace_files: `2`
- trace_entries: `64`
- effective_write_slots: `[1, 2]`
- effective_read_slots: `[2]`
- dominant_write_slot_fractions: `0.00, 0.00, 1.00`
- dominant_read_slot_fractions: `0.00, 0.00, 1.00`
- mean_write_slot_weights: `0.09, 0.11, 0.80`
- mean_read_slot_weights: `0.00, 0.00, 1.00`

#### query_name -> write slot

- alice -> slot 2 (1.00)
- bob -> slot 2 (1.00)
- carol -> slot 2 (1.00)
- dave -> slot 2 (1.00)
- erin -> slot 2 (1.00)
- frank -> slot 2 (1.00)
- grace -> slot 2 (1.00)
- heidi -> slot 2 (1.00)

#### query_field -> write slot

- color -> slot 2 (1.00)

#### query_name -> read slot

- alice -> slot 2 (1.00)
- bob -> slot 2 (1.00)
- carol -> slot 2 (1.00)
- dave -> slot 2 (1.00)
- erin -> slot 2 (1.00)
- frank -> slot 2 (1.00)
- grace -> slot 2 (1.00)
- heidi -> slot 2 (1.00)

#### query_field -> read slot

- color -> slot 2 (1.00)

### specialized_core / balance / slots_4

- trace_files: `2`
- trace_entries: `64`
- effective_write_slots: `[0, 3]`
- effective_read_slots: `[0, 3]`
- dominant_write_slot_fractions: `0.50, 0.00, 0.00, 0.50`
- dominant_read_slot_fractions: `0.50, 0.00, 0.00, 0.50`
- mean_write_slot_weights: `0.49, 0.08, 0.02, 0.41`
- mean_read_slot_weights: `0.50, 0.00, 0.00, 0.50`

#### query_name -> write slot

- alice -> slot 0 (0.57)
- bob -> slot 0 (0.50)
- carol -> slot 0 (0.50)
- dave -> slot 0 (1.00)
- erin -> slot 3 (0.57)
- frank -> slot 0 (0.62)
- grace -> slot 3 (0.71)
- heidi -> slot 0 (0.75)

#### query_field -> write slot

- color -> slot 0 (0.50)

#### query_name -> read slot

- alice -> slot 0 (0.57)
- bob -> slot 0 (0.50)
- carol -> slot 0 (0.50)
- dave -> slot 0 (1.00)
- erin -> slot 3 (0.57)
- frank -> slot 0 (0.62)
- grace -> slot 3 (0.71)
- heidi -> slot 0 (0.75)

#### query_field -> read slot

- color -> slot 0 (0.50)

### specialized_core / balance / slots_8

- trace_files: `2`
- trace_entries: `64`
- effective_write_slots: `[0, 2]`
- effective_read_slots: `[0, 2]`
- dominant_write_slot_fractions: `0.50, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00`
- dominant_read_slot_fractions: `0.50, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00`
- mean_write_slot_weights: `0.40, 0.07, 0.36, 0.02, 0.05, 0.01, 0.07, 0.02`
- mean_read_slot_weights: `0.50, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00`

#### query_name -> write slot

- alice -> slot 2 (0.57)
- bob -> slot 0 (0.50)
- carol -> slot 0 (0.50)
- dave -> slot 2 (1.00)
- erin -> slot 0 (0.57)
- frank -> slot 2 (0.62)
- grace -> slot 0 (0.71)
- heidi -> slot 2 (0.75)

#### query_field -> write slot

- color -> slot 0 (0.50)

#### query_name -> read slot

- alice -> slot 2 (0.57)
- bob -> slot 0 (0.50)
- carol -> slot 0 (0.50)
- dave -> slot 2 (1.00)
- erin -> slot 0 (0.57)
- frank -> slot 2 (0.62)
- grace -> slot 0 (0.71)
- heidi -> slot 2 (0.75)

#### query_field -> read slot

- color -> slot 0 (0.50)

### specialized_core / nobalance / slots_2

- trace_files: `2`
- trace_entries: `64`
- effective_write_slots: `[1]`
- effective_read_slots: `[1]`
- dominant_write_slot_fractions: `0.00, 1.00`
- dominant_read_slot_fractions: `0.00, 1.00`
- mean_write_slot_weights: `0.00, 1.00`
- mean_read_slot_weights: `0.00, 1.00`

#### query_name -> write slot

- alice -> slot 1 (1.00)
- bob -> slot 1 (1.00)
- carol -> slot 1 (1.00)
- dave -> slot 1 (1.00)
- erin -> slot 1 (1.00)
- frank -> slot 1 (1.00)
- grace -> slot 1 (1.00)
- heidi -> slot 1 (1.00)

#### query_field -> write slot

- color -> slot 1 (1.00)

#### query_name -> read slot

- alice -> slot 1 (1.00)
- bob -> slot 1 (1.00)
- carol -> slot 1 (1.00)
- dave -> slot 1 (1.00)
- erin -> slot 1 (1.00)
- frank -> slot 1 (1.00)
- grace -> slot 1 (1.00)
- heidi -> slot 1 (1.00)

#### query_field -> read slot

- color -> slot 1 (1.00)

### specialized_core / nobalance / slots_3

- trace_files: `2`
- trace_entries: `64`
- effective_write_slots: `[2]`
- effective_read_slots: `[2]`
- dominant_write_slot_fractions: `0.00, 0.00, 1.00`
- dominant_read_slot_fractions: `0.00, 0.00, 1.00`
- mean_write_slot_weights: `0.01, 0.01, 0.98`
- mean_read_slot_weights: `0.00, 0.00, 1.00`

#### query_name -> write slot

- alice -> slot 2 (1.00)
- bob -> slot 2 (1.00)
- carol -> slot 2 (1.00)
- dave -> slot 2 (1.00)
- erin -> slot 2 (1.00)
- frank -> slot 2 (1.00)
- grace -> slot 2 (1.00)
- heidi -> slot 2 (1.00)

#### query_field -> write slot

- color -> slot 2 (1.00)

#### query_name -> read slot

- alice -> slot 2 (1.00)
- bob -> slot 2 (1.00)
- carol -> slot 2 (1.00)
- dave -> slot 2 (1.00)
- erin -> slot 2 (1.00)
- frank -> slot 2 (1.00)
- grace -> slot 2 (1.00)
- heidi -> slot 2 (1.00)

#### query_field -> read slot

- color -> slot 2 (1.00)

### specialized_core / nobalance / slots_4

- trace_files: `2`
- trace_entries: `64`
- effective_write_slots: `[0, 3]`
- effective_read_slots: `[0, 3]`
- dominant_write_slot_fractions: `0.50, 0.00, 0.00, 0.50`
- dominant_read_slot_fractions: `0.50, 0.00, 0.00, 0.50`
- mean_write_slot_weights: `0.49, 0.04, 0.01, 0.47`
- mean_read_slot_weights: `0.50, 0.00, 0.00, 0.50`

#### query_name -> write slot

- alice -> slot 0 (0.57)
- bob -> slot 0 (0.50)
- carol -> slot 0 (0.50)
- dave -> slot 0 (1.00)
- erin -> slot 3 (0.57)
- frank -> slot 0 (0.62)
- grace -> slot 3 (0.71)
- heidi -> slot 0 (0.75)

#### query_field -> write slot

- color -> slot 0 (0.50)

#### query_name -> read slot

- alice -> slot 0 (0.57)
- bob -> slot 0 (0.50)
- carol -> slot 0 (0.50)
- dave -> slot 0 (1.00)
- erin -> slot 3 (0.57)
- frank -> slot 0 (0.62)
- grace -> slot 3 (0.71)
- heidi -> slot 0 (0.75)

#### query_field -> read slot

- color -> slot 0 (0.50)

### specialized_core / nobalance / slots_8

- trace_files: `2`
- trace_entries: `64`
- effective_write_slots: `[0, 2]`
- effective_read_slots: `[0, 2]`
- dominant_write_slot_fractions: `0.50, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00`
- dominant_read_slot_fractions: `0.50, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00`
- mean_write_slot_weights: `0.45, 0.05, 0.40, 0.01, 0.03, 0.00, 0.04, 0.01`
- mean_read_slot_weights: `0.50, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00`

#### query_name -> write slot

- alice -> slot 2 (0.57)
- bob -> slot 0 (0.50)
- carol -> slot 0 (0.50)
- dave -> slot 2 (1.00)
- erin -> slot 0 (0.57)
- frank -> slot 2 (0.62)
- grace -> slot 0 (0.71)
- heidi -> slot 2 (0.75)

#### query_field -> write slot

- color -> slot 0 (0.50)

#### query_name -> read slot

- alice -> slot 2 (0.57)
- bob -> slot 0 (0.50)
- carol -> slot 0 (0.50)
- dave -> slot 2 (1.00)
- erin -> slot 0 (0.57)
- frank -> slot 2 (0.62)
- grace -> slot 0 (0.71)
- heidi -> slot 2 (0.75)

#### query_field -> read slot

- color -> slot 0 (0.50)
