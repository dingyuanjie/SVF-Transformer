# Cheap Routing Mapping Report

- input_root: `outputs\experiments\phaseG_multifield_rank_study_formal\specialized_core`

## Main Question

- Does routing fully use 2 slots, then keep using only 2 slots when capacity grows?

## Verdict

- specialized_core / nobalance_soft: NO (slot_2_full=False, larger_collapse=True)
  - slots=2: write=[1], read=[1]
  - slots=4: write=[1], read=[1]
  - slots=8: write=[0, 6], read=[0, 6]
  - slots=16: write=[0, 3], read=[0, 3]

## Routing Groups

### specialized_core / nobalance_soft / slots_2

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

- age -> slot 1 (1.00)
- city -> slot 1 (1.00)
- color -> slot 1 (1.00)
- job -> slot 1 (1.00)

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

- age -> slot 1 (1.00)
- city -> slot 1 (1.00)
- color -> slot 1 (1.00)
- job -> slot 1 (1.00)

### specialized_core / nobalance_soft / slots_4

- trace_files: `2`
- trace_entries: `64`
- effective_write_slots: `[1]`
- effective_read_slots: `[1]`
- dominant_write_slot_fractions: `0.00, 1.00, 0.00, 0.00`
- dominant_read_slot_fractions: `0.00, 1.00, 0.00, 0.00`
- mean_write_slot_weights: `0.04, 0.93, 0.02, 0.02`
- mean_read_slot_weights: `0.00, 1.00, 0.00, 0.00`

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

- age -> slot 1 (1.00)
- city -> slot 1 (1.00)
- color -> slot 1 (1.00)
- job -> slot 1 (1.00)

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

- age -> slot 1 (1.00)
- city -> slot 1 (1.00)
- color -> slot 1 (1.00)
- job -> slot 1 (1.00)

### specialized_core / nobalance_soft / slots_8

- trace_files: `2`
- trace_entries: `64`
- effective_write_slots: `[0, 6]`
- effective_read_slots: `[0, 6]`
- dominant_write_slot_fractions: `0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.00`
- dominant_read_slot_fractions: `0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.00`
- mean_write_slot_weights: `0.42, 0.05, 0.04, 0.02, 0.00, 0.01, 0.43, 0.02`
- mean_read_slot_weights: `0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.00`

#### query_name -> write slot

- alice -> slot 0 (0.75)
- bob -> slot 6 (0.60)
- carol -> slot 0 (0.75)
- dave -> slot 0 (0.57)
- erin -> slot 0 (0.80)
- frank -> slot 6 (0.57)
- grace -> slot 0 (0.57)
- heidi -> slot 6 (1.00)

#### query_field -> write slot

- age -> slot 0 (0.67)
- city -> slot 0 (0.71)
- color -> slot 6 (0.60)
- job -> slot 6 (0.70)

#### query_name -> read slot

- alice -> slot 0 (0.75)
- bob -> slot 6 (0.60)
- carol -> slot 0 (0.75)
- dave -> slot 0 (0.57)
- erin -> slot 0 (0.80)
- frank -> slot 6 (0.57)
- grace -> slot 0 (0.57)
- heidi -> slot 6 (1.00)

#### query_field -> read slot

- age -> slot 0 (0.67)
- city -> slot 0 (0.71)
- color -> slot 6 (0.60)
- job -> slot 6 (0.70)

### specialized_core / nobalance_soft / slots_16

- trace_files: `2`
- trace_entries: `64`
- effective_write_slots: `[0, 3]`
- effective_read_slots: `[0, 3]`
- dominant_write_slot_fractions: `0.50, 0.00, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00`
- dominant_read_slot_fractions: `0.50, 0.00, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00`
- mean_write_slot_weights: `0.40, 0.00, 0.01, 0.37, 0.02, 0.01, 0.00, 0.04, 0.04, 0.01, 0.01, 0.01, 0.02, 0.00, 0.04, 0.00`
- mean_read_slot_weights: `0.50, 0.00, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00`

#### query_name -> write slot

- alice -> slot 0 (0.75)
- bob -> slot 3 (0.60)
- carol -> slot 0 (0.75)
- dave -> slot 0 (0.57)
- erin -> slot 0 (0.80)
- frank -> slot 3 (0.57)
- grace -> slot 0 (0.57)
- heidi -> slot 3 (1.00)

#### query_field -> write slot

- age -> slot 0 (0.67)
- city -> slot 0 (0.71)
- color -> slot 3 (0.60)
- job -> slot 3 (0.70)

#### query_name -> read slot

- alice -> slot 0 (0.75)
- bob -> slot 3 (0.60)
- carol -> slot 0 (0.75)
- dave -> slot 0 (0.57)
- erin -> slot 0 (0.80)
- frank -> slot 3 (0.57)
- grace -> slot 0 (0.57)
- heidi -> slot 3 (1.00)

#### query_field -> read slot

- age -> slot 0 (0.67)
- city -> slot 0 (0.71)
- color -> slot 3 (0.60)
- job -> slot 3 (0.70)
