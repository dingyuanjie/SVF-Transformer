import json

from analyze_slot_factors import analyze_trace_dir


def test_analyze_slot_factors_surfaces_position_like_split(tmp_path):
    trace_dir = tmp_path / "core_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    names = ["alice", "bob", "carol", "dave"]
    entries = []
    for offset in range(8):
        entity_names = names[offset % 4 :] + names[: offset % 4]
        query_entity_index = offset % 4
        query_name = entity_names[query_entity_index]
        if query_entity_index < 2:
            write_weights = [0.90, 0.02, 0.02, 0.06]
            read_weights = [0.88, 0.02, 0.02, 0.08]
            query_entity_bucket = "front_half"
        else:
            write_weights = [0.08, 0.02, 0.02, 0.88]
            read_weights = [0.10, 0.02, 0.02, 0.86]
            query_entity_bucket = "back_half"
        entries.append(
            {
                "entity_names": entity_names,
                "query_name": query_name,
                "query_name_initial": query_name[0],
                "query_entity_index": query_entity_index,
                "query_entity_bucket": query_entity_bucket,
                "query_field": "color",
                "query_field_index": 0,
                "answer_value": ["1"],
                "answer_first_token": "1",
                "slot_norms": write_weights,
                "slot_routing_weights": write_weights,
                "slot_read_weights": read_weights,
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

    assert summary["active_write_slots"] == [0, 3]
    assert summary["active_read_slots"] == [0, 3]
    top_write_factor = summary["write_top_factors"][0]["name"]
    top_read_factor = summary["read_top_factors"][0]["name"]
    assert top_write_factor in {"query_entity_bucket", "query_entity_index"}
    assert top_read_factor in {"query_entity_bucket", "query_entity_index"}
