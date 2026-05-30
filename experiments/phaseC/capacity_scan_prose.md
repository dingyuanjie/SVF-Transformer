# Capacity Scan On Prose

- Dataset: `data/svf_prose_20mb.txt`
- Split: `paragraph`, shuffled with fixed split seed
- Variant: `persistent_core`
- Seeds: `42 43 44`
- Scan: `core_slots = 1 2 4 8 16 32`
- Output root: `outputs/experiments/phaseC_capacity_formal/prose/capacity`

## Aggregate Results

- `slots=1`: mean validation CE `0.0341`
- `slots=2`: mean validation CE `0.0296`
- `slots=4`: mean validation CE `0.0323`
- `slots=8`: mean validation CE `0.0321`
- `slots=16`: mean validation CE `0.0297`
- `slots=32`: mean validation CE `0.0317`

## Reading

- The curve is not monotonic.
- `slots=1` is clearly weaker, so a minimal core underuses capacity.
- `slots=2` and `slots=16` are the best settings in this run.
- Larger capacity does not automatically help after that point.

This suggests a useful-range hypothesis rather than a pure bigger-is-better scaling law.
