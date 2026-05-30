# Formal Phase 2

- Goal: ablate `memory`, `persistent_core`, and the full SVF combination.
- Variants: `baseline`, `memory`, `persistent_core`, `memory_core`, `svf`
- Seeds: `42 43 44`
- Data: `data/svf_mixed_bpe_clean_20mb.txt`
- Tokenizer: `bpe`
- Output dir: `outputs/experiments/phase2_formal`
- Summary: `outputs/experiments/phase2_formal/aggregate_summary_20260530_144428.json`
- Checkpoints: `outputs/experiments/phase2_formal/checkpoints`

## Outcome

- `memory_core`: mean validation CE `0.0428`
- `persistent_core`: mean validation CE `0.0444`
- `svf`: mean validation CE `0.0450`
- `baseline`: mean validation CE `0.1350`
- `memory`: mean validation CE `0.1539`

Conclusion: `persistent_core` drives the gain, `memory` alone does not.
