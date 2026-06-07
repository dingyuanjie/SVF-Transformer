from analyze_value_swap_probe import analyze_trace_dir


def make_entry(
    *,
    query_name: str,
    query_field: str,
    query_entity_index: int,
    query_field_index: int,
    query_fact_token_start: int,
    query_fact_token_end: int,
    context_token_index: int,
    answer_value: str,
    write_slot: int,
    read_slot: int | None = None,
):
    if read_slot is None:
        read_slot = write_slot
    slot_count = 4
    write_weights = [0.0] * slot_count
    read_weights = [0.0] * slot_count
    write_weights[write_slot] = 1.0
    read_weights[read_slot] = 1.0
    return {
        "query_name": query_name,
        "query_field": query_field,
        "query_entity_index": query_entity_index,
        "query_field_index": query_field_index,
        "query_fact_token_start": query_fact_token_start,
        "query_fact_token_end": query_fact_token_end,
        "query_fact_position_bucket": "middle",
        "context_token_index": context_token_index,
        "remember_position_mode": "front",
        "field_orders_by_entity": {query_name: ["color", "city", "age", "job"]},
        "answer_value": [answer_value],
        "slot_routing_weights": write_weights,
        "slot_read_weights": read_weights,
        "slot_norms": write_weights,
    }


def test_value_swap_probe_detects_structure_dominant_pattern(tmp_path):
    trace_dir = tmp_path / "core_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    entries = [
        make_entry(
            query_name="alice",
            query_field="color",
            query_entity_index=0,
            query_field_index=0,
            query_fact_token_start=10,
            query_fact_token_end=11,
            context_token_index=30,
            answer_value="1",
            write_slot=0,
        ),
        make_entry(
            query_name="alice",
            query_field="color",
            query_entity_index=0,
            query_field_index=0,
            query_fact_token_start=10,
            query_fact_token_end=11,
            context_token_index=30,
            answer_value="9",
            write_slot=0,
        ),
        make_entry(
            query_name="bob",
            query_field="color",
            query_entity_index=1,
            query_field_index=0,
            query_fact_token_start=20,
            query_fact_token_end=21,
            context_token_index=30,
            answer_value="1",
            write_slot=3,
        ),
        make_entry(
            query_name="bob",
            query_field="color",
            query_entity_index=1,
            query_field_index=0,
            query_fact_token_start=20,
            query_fact_token_end=21,
            context_token_index=30,
            answer_value="9",
            write_slot=3,
        ),
    ]
    (trace_dir / "trace.json").write_text(
        __import__("json").dumps({"trace_entries": entries}),
        encoding="utf-8",
    )

    summary = analyze_trace_dir(trace_dir, max_examples=4)
    strict_write = summary["write_probe"]["strict"]
    assert strict_write["same_structure_diff_value"]["slot_match_rate"] == 1.0
    assert strict_write["same_value_diff_structure"]["slot_match_rate"] == 0.0
    assert strict_write["structure_minus_value_match_rate"] == 1.0
