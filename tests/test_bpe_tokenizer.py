import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("tokenizers") is None,
    reason="tokenizers package is not installed",
)


def test_train_bpe_tokenizer_round_trip():
    from train_bpe import train_tokenizer

    tokenizer = train_tokenizer(
        "Attractor dynamics provide a stabilizing reference for structural state.",
        vocab_size=128,
        min_frequency=1,
    )
    ids = tokenizer.encode("Attractor dynamics").ids
    assert ids
    assert "Attractor dynamics" in tokenizer.decode(ids)
