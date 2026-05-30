# Phase C Core Scans

- Goal: study why `Persistent Core` helps, instead of adding new speculative modules.
- Focus: `Persistent Core` capacity and dynamics.
- Dataset default: `prose`, because continuity should matter more on narrative-style text.
- Split protocol: `paragraph`, shuffled with a fixed split seed.

## Scans

- `capacity`: variant `persistent_core`, scan `core_slots = 1 2 4 8 16 32`
- `attractor`: variant `core_dynamics`, scan `attractor_strength = 0 0.01 0.05 0.1 0.2`
- `drift`: variant `core_dynamics`, scan `drift_scale = 0 0.05 0.1 0.2`

## Hypotheses

- Capacity hypothesis: larger persistent state improves validation CE up to a useful range.
- Attractor hypothesis: no attractor is worse, moderate attractor is best, overly strong attractor hurts adaptation.
- Drift hypothesis: zero drift underuses the state, moderate drift helps, excessive drift destabilizes performance.
