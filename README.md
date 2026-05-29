# SVF-Transformer

SVF-Transformer is a minimal, trainable baseline for a Transformer with structural dynamics constraints.

The project keeps the v1 design intentionally small:

- Transformer backbone for sequence modeling.
- Persistent Core for long-term structural state.
- Structural Drift for bounded state evolution.
- Attractor Dynamics for stability.
- Structural Conservation loss to discourage runaway energy growth.
- Ring Buffer memory with attention-based compression.

It is not an artificial consciousness simulator or an infinite recursive system. The goal of v1 is a stable research baseline that can run, train, and be extended.

## Project Structure

```text
SVF-Transformer/
  models/
    __init__.py
    svf_transformer.py
  tests/
    test_svf_transformer.py
  tools/
    make_svf_corpus.py
  generate_word.py
  generate_bpe.py
  eval_bpe.py
  generate.py
  train_word.py
  train_bpe.py
  train.py
  requirements.txt
  pyproject.toml
  README.md
```

## Install

CPU-only PyTorch:

```bash
pip install -r requirements.txt
```

NVIDIA GPU with CUDA 12.8:

```bash
pip install -r requirements-cu128.txt
```

Verify CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## Train

Run a tiny character-level language-modeling experiment:

```bash
python train.py --steps 200 --batch-size 16 --seq-len 64 --device cpu
```

Run with CUDA:

```bash
python train.py --data data/svf_corpus_10mb.txt --steps 3000 --batch-size 8 --seq-len 128 --device cuda
```

Use your own text file:

```bash
python train.py --data path/to/text.txt --steps 1000
```

Generate a synthetic SVF corpus:

```bash
python tools/make_svf_corpus.py --target-kb 1024 --out data/svf_corpus.txt
```

Generate a more coherent technical corpus:

```bash
python tools/make_svf_corpus.py --style technical --target-kb 10240 --out data/svf_corpus_technical_10mb.txt
```

Generate a technical corpus without note numbers:

```bash
python tools/make_svf_corpus.py --style technical --no-note-numbers --target-kb 10240 --out data/svf_corpus_technical_clean_10mb.txt
```

Generate an instruction-style corpus for cleaner prompt completions:

```bash
python tools/make_svf_instruction_corpus.py --target-kb 10240 --out data/svf_instruction_10mb.txt
```

Mix corpora for balanced BPE training:

```bash
python tools/mix_corpora.py --inputs data/svf_instruction_10mb.txt data/svf_corpus_technical_clean_10mb.txt data/svf_corpus_10mb.txt --weights 0.50 0.35 0.15 --target-kb 20480 --out data/svf_mixed_bpe_20mb.txt
```

For a cleaner mix without synthetic note numbers:

```bash
python tools/mix_corpora.py --inputs data/svf_instruction_10mb.txt data/svf_corpus_technical_clean_10mb.txt --weights 0.60 0.40 --target-kb 20480 --out data/svf_mixed_bpe_clean_20mb.txt
```

Generate continuous prose for fewer format jumps:

```bash
python tools/make_svf_prose_corpus.py --target-kb 20480 --out data/svf_prose_20mb.txt
```

## Generate

After training, load the checkpoint and generate from a prompt:

```bash
python generate.py --checkpoint svf_transformer.pt --prompt "The persistent core" --max-new-tokens 300 --device cpu
```

Reduce repetition during generation:

```bash
python generate.py --checkpoint svf_transformer.pt --prompt "Attractor dynamics" --max-new-tokens 500 --temperature 0.8 --top-k 30 --repetition-penalty 1.05 --no-repeat-ngram-size 24 --device cuda
```

## Word-Level Training

The character-level scripts are useful for smoke tests, but word-level training produces cleaner spelling.
Pure numeric tokens are normalized so synthetic paragraph numbers do not dominate generation.

```bash
python train_word.py --data data/svf_corpus_technical_10mb.txt --steps 10000 --batch-size 16 --seq-len 128 --device cuda
```

Generate from the word-level checkpoint:

```bash
python generate_word.py --checkpoint svf_word_transformer.pt --prompt "Attractor dynamics" --max-new-tokens 150 --temperature 0.8 --top-k 40 --device cuda
```

## BPE/Subword Training

BPE is the recommended tokenizer once the basic pipeline works. It reduces spelling errors without forcing every unknown phrase into whole-word vocabulary items.

```bash
python train_bpe.py --data data/svf_corpus_technical_clean_10mb.txt --steps 10000 --batch-size 16 --seq-len 128 --vocab-size 4096 --device cuda
```

For cleaner prompt completion behavior, train on the instruction-style corpus:

```bash
python train_bpe.py --data data/svf_instruction_10mb.txt --steps 10000 --batch-size 16 --seq-len 128 --vocab-size 4096 --device cuda
```

For the best balanced behavior, train on the mixed corpus:

```bash
python train_bpe.py --data data/svf_mixed_bpe_clean_20mb.txt --steps 15000 --batch-size 16 --seq-len 128 --vocab-size 4096 --device cuda --eval-interval 500 --save-best svf_bpe_best.pt
```

For prose-style generation, train the larger BPE model on continuous prose:

```bash
python train_bpe.py --data data/svf_prose_20mb.txt --steps 20000 --batch-size 8 --seq-len 256 --d-model 256 --layers 6 --heads 8 --vocab-size 4096 --device cuda --eval-interval 500 --save-best svf_bpe_best.pt
```

Generate from the BPE checkpoint:

```bash
python generate_bpe.py --checkpoint svf_bpe_transformer.pt --prompt "In this experiment, structural conservation is evaluated by" --max-new-tokens 180 --temperature 0.8 --top-k 50 --repetition-penalty 1.08 --frequency-penalty 0.10 --presence-penalty 0.03 --no-repeat-ngram-size 8 --device cuda
```

Run a fixed-prompt BPE evaluation and save the report:

```bash
python eval_bpe.py --checkpoint svf_bpe_transformer.pt --device cuda
```

Evaluate the best validation checkpoint:

```bash
python eval_bpe.py --checkpoint svf_bpe_best.pt --device cuda
```

## Test

```bash
pytest
```

## Minimal Usage

```python
import torch
from models import SVFTransformer, SVFTransformerConfig

config = SVFTransformerConfig(vocab_size=128, d_model=128, n_layers=4)
model = SVFTransformer(config)

tokens = torch.randint(0, 128, (2, 32))
out = model(tokens, targets=tokens)

print(out.logits.shape)
print(out.loss.item())
```
email：911260800@qq.com
技术支持：chatGPT codex
## License

MIT
