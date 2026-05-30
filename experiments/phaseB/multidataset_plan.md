# Phase B Multidataset Plan

- Goal: test whether the current advantage survives across different corpora, not just across random seeds.
- Variants: `baseline`, `persistent_core`, `memory_core`, `svf`
- Datasets:
  - `technical`: `data/svf_corpus_technical_clean_10mb.txt`
  - `instruction`: `data/svf_instruction_10mb.txt`
  - `prose`: `data/svf_prose_20mb.txt`
- Seeds: default `42 43 44`, expandable to `5` if needed
- Tokenizer: `bpe`, fit on the train split only
- Split protocol: default `paragraph` units, optional shuffle with fixed split seed
- Baseline control: auto-match baseline parameters to `svf`
- Planned output root: `outputs/experiments/phaseB_multidataset`

## Question

Does the advantage remain when the corpus style changes, or is the current win specific to one dataset family?
