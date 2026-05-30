# Drift Scan On Prose

- Dataset: `data/svf_prose_20mb.txt`
- Split: `paragraph`, shuffled with fixed split seed
- Variant: `core_dynamics`
- Seeds: `42 43 44`
- Scan: `drift_scale = 0 0.05 0.1 0.2`
- Output root: `outputs/experiments/phaseC_drift_formal/prose/drift`

## Aggregate Results

- `drift=0.0`: mean validation CE `0.0336`
- `drift=0.05`: mean validation CE `0.0315`
- `drift=0.1`: mean validation CE `0.0297`
- `drift=0.2`: mean validation CE `0.0303`

## Reading

- Zero drift is clearly worse than moderate drift.
- The best setting in this run is `0.1`.
- Increasing drift beyond that does not help and begins to degrade performance.

This supports a moderate-dynamics hypothesis rather than either zero-motion or maximum-motion behavior.
