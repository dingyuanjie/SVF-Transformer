# Counterfactual Value-Swap Replay

- checkpoint_path: `outputs\experiments\smoke_counterfactual_replay\checkpoints\delay8_specialized_core_seed42_best.pt`
- variant: `specialized_core`
- delay_tokens: `8`
- samples_evaluated: `8`
- total_counterfactuals: `72`
- write_slot_same_rate: `0.7778`
- read_slot_same_rate: `0.9167`
- write_mean_l1_delta: `0.0002`
- read_mean_l1_delta: `0.0001`

## Per Field

- age: counterfactuals=`9` write_slot_same_rate=`1.0000` read_slot_same_rate=`0.8889`
- city: counterfactuals=`27` write_slot_same_rate=`0.5556` read_slot_same_rate=`1.0000`
- job: counterfactuals=`36` write_slot_same_rate=`0.8889` read_slot_same_rate=`0.8611`

## Example Flips

- idx=1 frank.city: 8 -> 1 | write 0 -> 12 | read 14 -> 14
- idx=1 frank.city: 8 -> 3 | write 0 -> 12 | read 14 -> 14
- idx=1 frank.city: 8 -> 4 | write 0 -> 12 | read 14 -> 14
- idx=1 frank.city: 8 -> 6 | write 0 -> 12 | read 14 -> 14
- idx=1 frank.city: 8 -> 7 | write 0 -> 2 | read 14 -> 14
- idx=1 frank.city: 8 -> 9 | write 0 -> 12 | read 14 -> 14
- idx=2 frank.job: 6 -> 0 | write 12 -> 0 | read 14 -> 14
- idx=2 frank.job: 6 -> 5 | write 12 -> 0 | read 14 -> 14
- idx=2 frank.job: 6 -> 9 | write 12 -> 12 | read 14 -> 5
- idx=3 erin.age: 5 -> 9 | write 0 -> 0 | read 14 -> 10
- idx=4 frank.job: 7 -> 9 | write 0 -> 12 | read 14 -> 14
- idx=5 niaj.job: 8 -> 1 | write 0 -> 0 | read 5 -> 14
