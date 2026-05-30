# Formal Phase 1

- Goal: compare `baseline` vs `svf` under one tokenizer, one split, and multiple seeds.
- Variants: `baseline`, `svf`
- Seeds: `42 43 44`
- Data: `data/svf_mixed_bpe_clean_20mb.txt`
- Tokenizer: `bpe`
- Output dir: `outputs/experiments/phase1_formal`
- Summary: `outputs/experiments/phase1_formal/aggregate_summary_20260530_140241.json`
- Checkpoints: `outputs/experiments/phase1_formal/checkpoints`

## Outcome

- `svf`: mean validation CE `0.0450`
- `baseline`: mean validation CE `0.1350`
- Conclusion: full SVF clearly outperforms the plain Transformer baseline in the initial formal run.
