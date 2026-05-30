# Strict Reproduction Phase 1

- Goal: rerun `baseline` vs `svf` with stricter protocol checks.
- Variants: `baseline`, `svf`
- Seeds: `42 43 44`
- Data: `data/svf_mixed_bpe_clean_20mb.txt`
- Tokenizer: `bpe`, fit on the train split only
- Eval protocol: independent batches, no memory writeback
- Train protocol: no cross-batch carried state by default
- Parameter control: `baseline` auto-matched to `svf`
- Output dir: `outputs/experiments/phase3_strict_phase1_parammatch`
- Summary: `outputs/experiments/phase3_strict_phase1_parammatch/aggregate_summary_20260530_150830.json`
- Checkpoints: `outputs/experiments/phase3_strict_phase1_parammatch/checkpoints`

## Outcome

- `svf`: mean validation CE `0.0434`
- `baseline`: mean validation CE `0.1356`
- Parameter counts: `svf=1292928`, `baseline=1315440`

Conclusion: the advantage survives stricter reproduction and parameter matching, so the gain is unlikely to be explained by obvious leakage or model size alone.
