import json

from analyze_slot_factors import analyze_trace_dir
from train_delayed_recall import DelayedRecallDataset, SyntheticTokenizer, TaskSpec


def test_dataset_emits_fact_span_and_remember_position_metadata():
    task_spec = TaskSpec(
        task_type="multi_entity",
        entities_per_sample=3,
        fields=["age", "city", "color"],
        value_length=1,
    )
    tokenizer = SyntheticTokenizer(noise_vocab_size=8, fields=task_spec.fields)
    dataset = DelayedRecallDataset(
        num_samples=1,
        delay_tokens=6,
        noise_vocab_size=8,
        seed=123,
        tokenizer=tokenizer,
        task_spec=task_spec,
        field_order_mode="shuffled",
        remember_position_mode="middle",
    )

    metadata = dataset.get_metadata(0)
    assert metadata["field_order_mode"] == "shuffled"
    assert metadata["remember_position_mode"] == "middle"
    assert metadata["prefix_noise_length"] + metadata["suffix_noise_length"] == 6
    assert metadata["remember_token_index"] == 1 + metadata["prefix_noise_length"]
    assert metadata["fact_tokens_start"] == metadata["remember_token_index"] + 1
    assert metadata["query_entity_token_start"] <= metadata["query_fact_token_start"] <= metadata["query_fact_token_end"]
    assert metadata["query_entity_token_end"] >= metadata["query_fact_token_end"]
    assert metadata["query_fact_position_bucket"] in {"front", "middle", "back"}
    assert len(metadata["field_orders_by_entity"][metadata["query_name"]]) == 3
    query_key = f"{metadata['query_name']}.{metadata['query_field']}"
    assert query_key in metadata["fact_spans"]
    assert metadata["fact_spans"][query_key]["token_start"] == metadata["query_fact_token_start"]


def test_analyze_slot_factors_can_surface_remember_position_signal(tmp_path):
    trace_dir = tmp_path / "core_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for sample_index in range(8):
        front_mode = sample_index < 4
        if front_mode:
            write_weights = [0.90, 0.10, 0.00, 0.00]
            read_weights = [0.88, 0.12, 0.00, 0.00]
            remember_position_mode = "front"
            prefix_noise_length = 0
            suffix_noise_length = 6
        else:
            write_weights = [0.00, 0.00, 0.10, 0.90]
            read_weights = [0.00, 0.00, 0.12, 0.88]
            remember_position_mode = "back"
            prefix_noise_length = 6
            suffix_noise_length = 0
        entries.append(
            {
                "query_name": "alice",
                "query_field": "color",
                "query_entity_index": 0,
                "query_entity_bucket": "front_half",
                "query_field_index": 0,
                "answer_value": ["1"],
                "answer_first_token": "1",
                "slot_norms": write_weights,
                "slot_routing_weights": write_weights,
                "slot_read_weights": read_weights,
                "remember_position_mode": remember_position_mode,
                "prefix_noise_length": prefix_noise_length,
                "suffix_noise_length": suffix_noise_length,
            }
        )
    (trace_dir / "delay32_specialized_core_seed42.json").write_text(
        json.dumps({"trace_entries": entries}),
        encoding="utf-8",
    )

    summary = analyze_trace_dir(
        trace_dir,
        min_count_fraction=0.10,
        min_mean_weight=0.10,
        top_k_factors=5,
        top_k_values=8,
    )

    factor_names = {item["name"] for item in summary["write_top_factors"]}
    assert "remember_position_mode" in factor_names or "prefix_noise_length" in factor_names
