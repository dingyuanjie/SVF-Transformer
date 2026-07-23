from counterfactual_value_swap_replay import (
    apply_counterfactual_value,
    candidate_counterfactual_values,
)


def test_candidate_counterfactual_values_excludes_original_digit():
    values = candidate_counterfactual_values(["5"], max_counterfactuals=9, rng=__import__("random").Random(0))
    assert len(values) == 9
    assert ["5"] not in values
    assert values[0] == ["0"]
    assert values[-1] == ["9"]


def test_apply_counterfactual_value_only_rewrites_target_fact_and_answer():
    tokens = [
        "<bos>",
        "remember",
        "alice",
        "age",
        "1",
        "sep",
        "bob",
        "city",
        "7",
        "context",
        "question",
        "alice",
        "age",
        "answer",
        "1",
        "<eos>",
    ]
    metadata = {
        "query_fact_token_start": 3,
        "query_fact_token_end": 4,
        "answer_token_start": 14,
        "answer_token_end": 14,
    }
    updated = apply_counterfactual_value(tokens, metadata, new_value_tokens=["9"])
    assert updated[4] == "9"
    assert updated[14] == "9"
    assert updated[:4] == tokens[:4]
    assert updated[5:14] == tokens[5:14]
    assert updated[15:] == tokens[15:]
