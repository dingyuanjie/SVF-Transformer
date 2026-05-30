# Attractor Scan On Prose

- Dataset: `data/svf_prose_20mb.txt`
- Split: `paragraph`, shuffled with fixed split seed
- Variant: `core_dynamics`
- Seeds: `42 43 44`
- Scan: `attractor_strength = 0 0.01 0.05 0.1 0.2`
- Output root: `outputs/experiments/phaseC_attractor_formal/prose/attractor`

## Aggregate Results

- `attractor=0.0`: mean validation CE `0.0289`
- `attractor=0.01`: mean validation CE `0.0284`
- `attractor=0.05`: mean validation CE `0.0293`
- `attractor=0.1`: mean validation CE `0.0306`
- `attractor=0.2`: mean validation CE `0.0321`

## Reading

- The best setting is small but non-zero.
- `0.01` outperforms `0.0`, so a weak attractor helps.
- Stronger pull progressively hurts performance.
- This supports a gentle-stabilization hypothesis rather than a strong-restoration hypothesis.
