# Cheap Routing Mapping Report

- input_root: `outputs\experiments\phaseG_rank_study_formal_top1\specialized_core`

## Main Question

- Does routing fully use 2 slots, then keep using only 2 slots when capacity grows?

## Verdict

- specialized_core / nobalance_top1: NO (slot_2_full=False, larger_collapse=True)
  - slots=2: write=[0], read=[0]
  - slots=4: write=[1, 2], read=[1, 2]
  - slots=8: write=[6, 7], read=[6, 7]
  - slots=16: write=[2, 11], read=[2, 11]

## Routing Groups

### specialized_core / nobalance_top1 / slots_2

- trace_files: `2`
- trace_entries: `64`
- effective_write_slots: `[0]`
- effective_read_slots: `[0]`
- dominant_write_slot_fractions: `1.00, 0.00`
- dominant_read_slot_fractions: `1.00, 0.00`
- mean_write_slot_weights: `1.00, 0.00`
- mean_read_slot_weights: `1.00, 0.00`

#### query_name -> write slot

- alice -> slot 0 (1.00)
- bob -> slot 0 (1.00)
- carol -> slot 0 (1.00)
- dave -> slot 0 (1.00)
- erin -> slot 0 (1.00)
- frank -> slot 0 (1.00)
- grace -> slot 0 (1.00)
- heidi -> slot 0 (1.00)

#### query_field -> write slot

- color -> slot 0 (1.00)

#### query_name -> read slot

- alice -> slot 0 (1.00)
- bob -> slot 0 (1.00)
- carol -> slot 0 (1.00)
- dave -> slot 0 (1.00)
- erin -> slot 0 (1.00)
- frank -> slot 0 (1.00)
- grace -> slot 0 (1.00)
- heidi -> slot 0 (1.00)

#### query_field -> read slot

- color -> slot 0 (1.00)

### specialized_core / nobalance_top1 / slots_4

- trace_files: `2`
- trace_entries: `64`
- effective_write_slots: `[1, 2]`
- effective_read_slots: `[1, 2]`
- dominant_write_slot_fractions: `0.00, 0.50, 0.50, 0.00`
- dominant_read_slot_fractions: `0.00, 0.50, 0.50, 0.00`
- mean_write_slot_weights: `0.00, 0.50, 0.50, 0.00`
- mean_read_slot_weights: `0.00, 0.50, 0.50, 0.00`

#### query_name -> write slot

- alice -> slot 2 (0.57)
- bob -> slot 1 (0.50)
- carol -> slot 1 (0.50)
- dave -> slot 2 (1.00)
- erin -> slot 1 (0.57)
- frank -> slot 2 (0.62)
- grace -> slot 1 (0.71)
- heidi -> slot 2 (0.75)

#### query_field -> write slot

- color -> slot 1 (0.50)

#### query_name -> read slot

- alice -> slot 2 (0.57)
- bob -> slot 1 (0.50)
- carol -> slot 1 (0.50)
- dave -> slot 2 (1.00)
- erin -> slot 1 (0.57)
- frank -> slot 2 (0.62)
- grace -> slot 1 (0.71)
- heidi -> slot 2 (0.75)

#### query_field -> read slot

- color -> slot 1 (0.50)

### specialized_core / nobalance_top1 / slots_8

- trace_files: `2`
- trace_entries: `64`
- effective_write_slots: `[6, 7]`
- effective_read_slots: `[6, 7]`
- dominant_write_slot_fractions: `0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.50`
- dominant_read_slot_fractions: `0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.50`
- mean_write_slot_weights: `0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.50`
- mean_read_slot_weights: `0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.50`

#### query_name -> write slot

- alice -> slot 6 (0.57)
- bob -> slot 6 (0.50)
- carol -> slot 6 (0.50)
- dave -> slot 6 (1.00)
- erin -> slot 7 (0.57)
- frank -> slot 6 (0.62)
- grace -> slot 7 (0.71)
- heidi -> slot 6 (0.75)

#### query_field -> write slot

- color -> slot 6 (0.50)

#### query_name -> read slot

- alice -> slot 6 (0.57)
- bob -> slot 6 (0.50)
- carol -> slot 6 (0.50)
- dave -> slot 6 (1.00)
- erin -> slot 7 (0.57)
- frank -> slot 6 (0.62)
- grace -> slot 7 (0.71)
- heidi -> slot 6 (0.75)

#### query_field -> read slot

- color -> slot 6 (0.50)

### specialized_core / nobalance_top1 / slots_16

- trace_files: `2`
- trace_entries: `64`
- effective_write_slots: `[2, 11]`
- effective_read_slots: `[2, 11]`
- dominant_write_slot_fractions: `0.00, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00`
- dominant_read_slot_fractions: `0.00, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00`
- mean_write_slot_weights: `0.00, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00`
- mean_read_slot_weights: `0.00, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00`

#### query_name -> write slot

- alice -> slot 11 (0.57)
- bob -> slot 2 (0.50)
- carol -> slot 2 (0.50)
- dave -> slot 11 (1.00)
- erin -> slot 2 (0.57)
- frank -> slot 11 (0.62)
- grace -> slot 2 (0.71)
- heidi -> slot 11 (0.75)

#### query_field -> write slot

- color -> slot 2 (0.50)

#### query_name -> read slot

- alice -> slot 11 (0.57)
- bob -> slot 2 (0.50)
- carol -> slot 2 (0.50)
- dave -> slot 11 (1.00)
- erin -> slot 2 (0.57)
- frank -> slot 11 (0.62)
- grace -> slot 2 (0.71)
- heidi -> slot 11 (0.75)

#### query_field -> read slot

- color -> slot 2 (0.50)
