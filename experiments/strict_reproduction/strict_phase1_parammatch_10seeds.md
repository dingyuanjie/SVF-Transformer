# Strict Reproduction Phase 1, 10 Seeds

- Goal: test whether the `baseline` vs `svf` advantage survives a broader seed sweep under the stricter protocol.
- Variants: `baseline`, `svf`
- Seeds: `42 43 44 45 46 47 48 49 50 51`
- Data: `data/svf_mixed_bpe_clean_20mb.txt`
- Tokenizer: `bpe`, fit on the train split only
- Eval protocol: independent batches, no memory writeback
- Train protocol: no cross-batch carried state by default
- Parameter control: `baseline` auto-matched to `svf`
- Planned output dir: `outputs/experiments/phaseA_strict_phase1_10seeds`

## Question

Is the current `svf` advantage robust across a wider random seed sweep, or was the previous 3-seed result partly due to random luck?
