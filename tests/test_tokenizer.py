from tokenizer import WordTokenizer


def test_word_tokenizer_round_trip_shape():
    tokenizer = WordTokenizer.train("Attractor dynamics pull the core state.")
    ids = tokenizer.encode("Attractor dynamics.")
    text = tokenizer.decode(ids)
    assert ids
    assert "Attractor" in text
    assert text.endswith(".")


def test_word_tokenizer_normalizes_numbers():
    tokenizer = WordTokenizer.train("Metric note 123: causal structure. Metric note 456.")
    text = tokenizer.decode(tokenizer.encode("Metric note 789: causal structure."))
    assert "789" not in text
    assert "Metric note:" in text
